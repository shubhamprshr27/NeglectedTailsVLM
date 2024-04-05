import os
import requests
import pickle
from laion_parser import LaionParser
from multiprocessing import Pool
from img2dataset import download
from concurrent.futures import ThreadPoolExecutor
import shutil
import pandas as pd
from PIL import Image
from io import BytesIO
import argparse
import time
import random

def validate_and_save_image(base_path,response_content, img_count):
    valid_image = False
    try:
        image = Image.open(BytesIO(response_content))

        width, height = image.size
        if max(width,height) > 80: # Avoid very small images, they tend to be empty.
            img_path = os.path.join(base_path,f'{img_count}.JPEG')
            image.convert('RGB').save(img_path, 'JPEG')
            img_count+=1
            valid_image = True
    except Exception as e:
        s = "error downloading"
    return img_count, valid_image

"""
    Worker Function to download images, without using img2dataset.
    Benefits: Allows us to control exactly the amount of images to be downloaded.
    Disadvantage: Slow!
"""
def worker(args):
    laion_parser = LaionParser('LAION400M.db', source='./')
    item, download_folder = args
    key, metadata =  item
    folder_path = os.path.join(download_folder, str(key))
    os.makedirs(folder_path, exist_ok=True)
    downloads_dict = {key: []}

    img_count = 0
    start = time.time()
    # randomly download images.
    random.shuffle(metadata)
    for sample in metadata:
        shard, rowid, text, relevance_score = sample
        if relevance_score == 'yes' or isinstance(relevance_score, float): # LLAMA based or similarity based.
            result = laion_parser.find_by_id(rowid=rowid, shard=shard, column='URL')
            url,nsfw = result
            if nsfw == 'NSFW':
                continue
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200: # Successfully mined images.
                    img_count, valid_image = validate_and_save_image(base_path=folder_path, 
                                                        response_content=response.content,
                                                        img_count=img_count)
                    if valid_image:
                        downloads_dict[key].append((shard, rowid, text, url, relevance_score))
            except Exception as e:
                continue
        if img_count == 1000:
            break
    print(f'Downloaded for {key} - {img_count} -', time.time()-start)
    laion_parser.conn.close()
    return downloads_dict


def download_and_save_imgs(retrieved_captions: dict, download_folder: str, dataset: str, pre_training_corpus:str ,name_type:str, ):
    os.makedirs(download_folder, exist_ok=True)
    params = [((key,value), download_folder) for (key,value) in retrieved_captions.items()]
    
    with Pool(32) as pool:
        downloads = pool.map(worker, params)

    all_downloads = {}
    for item in downloads:
        all_downloads.update(item)  
    
    meta_data_dir = os.path.join(dataset, pre_training_corpus, name_type)
    if not os.path.exists(meta_data_dir):
        os.makedirs(meta_data_dir)
    with open(f'./{meta_data_dir}/download-meta-data.pkl', 'wb') as f:
        pickle.dump(all_downloads, f)

def move_file(root_folder,child_folder,filename, file_count, cls):
    file_path = os.path.join(child_folder, filename)
    dest_path = os.path.join(root_folder, cls,f'{str(file_count)}.{ext}')
    if os.path.isfile(file_path) and not filename.endswith('.json'):
        shutil.move(file_path, dest_path)

def img2dataset_download(parquet_path: str, download_dir):
    print('Downloading -', parquet_path)
    if os.path.exists(download_dir):
        print('Already Downloaded images')
    else:
        os.makedirs(download_dir, exist_ok=True)
        download(
            processes_count=32,
            thread_count=64,
            url_list=parquet_path,
            resize_mode='no',
            encode_quality=100,
            input_format='parquet',
            output_format='files',
            min_image_size=85,
            number_sample_per_shard=2000000,
            output_folder=download_dir
        )   
        print(f'Downloaded - {parquet_path}')

def create_parquet(
        dataset,
        retrieved_captions, 
        sampling = 'ranked', 
        pre_training_corpus = 'laion400m',
        max_images = 1000
        ):
    metadata_path = os.path.join(f'./{dataset}', f'metadata-{sampling}-{pre_training_corpus}.meta')
    urls_path = os.path.join(f'./{dataset}',f'urls-{sampling}-{pre_training_corpus}.parquet')
    
    if os.path.exists(urls_path):
        print("URL parquet exists.")
        return urls_path, metadata_path
    
    dataset_urls = []
    download_metadata = {}
    for i, key in enumerate(retrieved_captions.keys()):
        # Randomly sample image urls.
        if sampling == 'random':
            retrieved_captions[key] = list(retrieved_captions[key])
            random.shuffle(retrieved_captions[key])

            # In img2dataset we can't account for data loss.
            # Therefore one easy way to do ensure that we get enough images is to download more images
            # A factor of 2 is generally good.
            upper_limit = max_images*2
            metadata = retrieved_captions[key][:upper_limit]
        else:
            metadata = list(sorted(retrieved_captions[key][:upper_limit], reverse=True, key= lambda x: x[-1]))
        with ThreadPoolExecutor(16) as executor:
            metadata = list(executor.map(process_sample, metadata))
        
        urls = [{'class': key,'url': metadata_i[-2]} for metadata_i in metadata]
        download_metadata[key] = metadata
        dataset_urls.extend(urls)
        print(f'{i}, key: {key}-{time.time() - start}')
    df = pd.DataFrame(dataset_urls)
    df.to_parquet(urls_path, index=False)
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(download_metadata, f)
    return urls_path, metadata_path

def process_sample(sample):
    laion_parser = LaionParser('LAION400M.db', source='./')

    # Hotfix for ImageNet metadata.
    if len(sample) == 3:
        shard = sample[0]
        rowid = sample[1]
    else:
        shard = sample[1]
        rowid = sample[2]
    url, nsfw = laion_parser.find_by_id(rowid=rowid, shard=shard, column='URL')
    laion_parser.conn.close()
    del laion_parser
    return (*sample, url, nsfw)

def restructure_download(parquet_path, download_dir, metadata_path=''):
    df = pd.read_parquet(parquet_path)

    classes = df['class'].unique()
    child_folder = os.path.join(download_dir, '00000')

    metadata = pickle.load(open(metadata_path,'rb'))

    for cls in classes:
        class_path = os.path.join(download_dir, str(cls))
        os.makedirs(class_path, exist_ok=True)
        row_ids = df[df['class'] == cls].index.tolist()
        file_count = 0
        for i, row_id in enumerate(row_ids):
            fomatted_rowid = "{:012}".format(row_id)
            if os.path.exists(os.path.join(download_dir, child_folder,f'{fomatted_rowid}.jpg')):
                metadata[str(cls)][i]=(*metadata[str(cls)][i],file_count)
                move_file(download_dir, child_folder, f'{fomatted_rowid}.jpg', file_count, cls)
                file_count +=1
        print('Completed for: -', cls) 
    shutil.rmtree(child_folder)
    with open(metadata_path,'wb') as f:
        pickle.dump(metadata, f)
    print('Restructuring completed.')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for script.')
    parser.add_argument('--arch', type=str, default='ViT-B/32', help='ViT Transformer arch.')
    parser.add_argument('--pre_training_corpus', type=str, default='laion400m')
    parser.add_argument('--dataset', type=str, default='imagenet_1k', help='Dataset that is to be used.')
    parser.add_argument('--name_type', type=str, default='alternates', choices=['name', 'most_common_name', 'alternates'])
    parser.add_argument('--sampling', type=str, default='ranked', choices=['ranked', 'random'])
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--max_images', type=int, default=500)
    parser.add_argument('--use_img2dataset', type=bool, default=True)

    args = parser.parse_args()
    random.seed(0)
    print('Starting loading captions.')
    caption_file = f'./{args.dataset}/mined_captions-{args.pre_training_corpus.upper()}{f"-{args.tag}" if args.tag is not None else ""}'
    retrieved_captions = pickle.load(open(caption_file,'rb'))
    download_dir = f'/data3/sparasha/cliptail/research/data/{args.dataset}/retrieved_1m-{args.pre_training_corpus}-{args.name_type}-{args.sampling}{f"-{args.tag}" if args.tag is not None else ""}'
    start = time.time()
    parquet_path, metadata_path = create_parquet(args.dataset,retrieved_captions=retrieved_captions,sampling=args.sampling, sampling_threshold=args.sampling_threshold, pre_training_corpus=args.pre_training_corpus)
    
    # Download Images using Img2Dataset or not.
    # Preferred to download via Img2Dataset
    if args.use_img2dataset:
        img2dataset_download(parquet_path=parquet_path, download_dir=download_dir)
        restructure_download(parquet_path=parquet_path, download_dir=download_dir, metadata_path=metadata_path)
    else:
        download_and_save_imgs(retrieved_captions=retrieved_captions, 
                           download_folder=download_dir,
                           dataset=args.dataset, 
                           pre_training_corpus=args.pre_training_corpus,
                           name_type=args.name_type,
                           )
