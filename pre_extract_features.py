from utils.datasets.dataset_utils import SPLIT_NAMES, get_dataset, DatasetWrapper
import torch
from torch.utils.data import DataLoader
from utils.extras import get_engine, OPENCLIP_MODEL_DIC
import os
import json
import argparse
import time
import pickle
import sys
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def extract_feats(model, dataloader):
    img_feats_store = None
    labels_store = None
    model.cuda()
    for i, data in enumerate(dataloader):
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.long()

        with torch.no_grad():
            img_feats = model.encode_image(imgs)

        if img_feats_store == None:
            img_feats_store = img_feats.cpu()
            labels_store = labels
        else:
            img_feats_store = torch.cat([img_feats_store, img_feats.cpu()], dim=0)
            labels_store = torch.cat([labels_store, labels], dim=0)
    return {'image_features': img_feats_store, 'labels': labels_store}

def save_dataset_feats(dataset_dict, 
                       dataset_name, 
                       split, 
                       pre_training_corpus, 
                       arch, 
                       save_dir: str='./data/test_data'): # 'pre_extracted'
    destination = os.path.join(save_dir,dataset_name)


    save_name = f"{dataset_name}_{split}_{pre_training_corpus}_{arch}"
    

    final_save_path = os.path.join(destination, f'{save_name}.pkl')
    torch.save(dataset_dict, final_save_path)
    return save_name


def pre_extract_test_data(
        device=0,
        batch_size=512):
    val_datasets = ['imagenet_1k',
                    'flowers102',
                    'stanford_cars',
                    'fgvc_aircraft', 
                    'imagenet_v2', 
                    'dtd',
                    'food101',
                    'oxford_pets',
                    'eurosat'
                    'cub2011'
                    ]
    torch.cuda.set_device(device)

    # Extract all testing data.
    for i, dataset_name in enumerate(val_datasets):
        split_name = SPLIT_NAMES['test'][dataset_name]
        print(f'-{i}-{dataset_name}')
        for (pre_training_corpus, archs) in OPENCLIP_MODEL_DIC.items():
            for (arch_key, config) in archs.items():
                _, arch = config
                model, preprocess, _ = get_engine(arch=arch_key, 
                                                  corpus=pre_training_corpus)
                dataset = get_dataset(dataset=dataset_name, 
                                      dataset_root =f'./data/{dataset_name}/' , 
                                      split= 'test', 
                                      preprocess=preprocess)
                val_dataloader = DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False, 
                                            num_workers=16, 
                                            drop_last=False)
                model = model.cuda()

                dataset_dict = extract_feats(model, dataloader=val_dataloader)
                save_dataset_feats(
                    dataset_dict=dataset_dict,
                    dataset_name=dataset_name,
                    split=split_name,
                    pre_training_corpus=pre_training_corpus,
                    arch=arch
                )
                print('done', dataset_name, arch_key, pre_training_corpus)

""" Helper function to load and preprocess an image. """
def load_and_preprocess_image(file_path, preprocess):
    img_tensor = preprocess(Image.open(file_path))
    return img_tensor

def pre_extract_directory(mined_folder, 
                          arch_key, 
                          pre_training_corpus, 
                          dataset='imagenet_1k', 
                          batch_size=512):
    dataset_dir=f'./data/{dataset}'
    if not os.path.exists(dataset_dir):
        sys.exit(f'Datset: {dataset} was not found.')
    mined_folder = os.path.join(dataset_dir,mined_folder)
    if not os.path.exists(mined_folder):
        sys.exit(f'Mined Data for {dataset} not found')
    classes = [cls for cls in os.listdir(mined_folder) if os.path.isdir(os.path.join(mined_folder, cls))]
    
    # Load OpenCLIP model.
    model, preprocess, tokenizer = get_engine(arch=arch_key, corpus=pre_training_corpus)
    model = model.cuda()
    start = time.time()
    extracted_feats = {}
    
    # metadata file for retrieved data.
    metadata = pickle.load(open(f'./analysis/laion/{dataset}/metadata-random-0.0-{pre_training_corpus}.meta', 'rb'))
    
    # Extract for each class.
    for i, cls in enumerate(classes):
        source_folder = os.path.join(mined_folder, cls)
        files_in_folder = os.listdir(source_folder)
        extracted_feats[cls] = {'feats': None, 'file_paths': None}
        download_metadata = [tup for tup in metadata[cls] if type(tup[-1]) == int]
        files_in_folder = [f'{tup[-1]}.jpg' for tup in download_metadata]
        if not files_in_folder:
            print('Empty Dir for class:', cls)
            print(f'{(i+1)*100/len(classes)}% - Done - class: {cls} - time: {time.time() - start}')
            continue

        file_list = sorted(files_in_folder, key=lambda file_name: int(file_name.split('.')[0]))
        start = time.time()
        file_paths = [os.path.join(source_folder, file) for file in file_list]
        
        # A hotfix for ImageNet metadata.
        caption_idx = -4
        if type(download_metadata[0][caption_idx]) != str:
            caption_idx = -5

        caption_tokens = tokenizer([tup[caption_idx] for tup in download_metadata]).cuda()

        # Load files.
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_and_preprocess_image, [os.path.join(source_folder, file) for file in file_list], [preprocess for _ in file_list]))
        
        class_tensors = torch.stack(results).cuda()
        img_embeddings = None
        caption_embeddings = None
        with torch.no_grad():
            for j in range(0, len(class_tensors), batch_size):
                batch_tensors = class_tensors[j:j+batch_size]
                text_batch_tensors = caption_tokens[j:j+batch_size]
                batch_embeddings = model.encode_image(batch_tensors)
                text_batch_embeddings = model.encode_text(text_batch_tensors)
                
                if img_embeddings is None:
                    img_embeddings = batch_embeddings
                    caption_embeddings = text_batch_embeddings
                else:
                    img_embeddings = torch.cat((img_embeddings, batch_embeddings), dim=0)
                    caption_embeddings = torch.cat((caption_embeddings, text_batch_embeddings), dim=0)
        img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
        caption_embeddings /= caption_embeddings.norm(dim=-1, keepdim=True)

        extracted_feats[cls]['feats'] = img_embeddings.cpu()
        extracted_feats[cls]['caption_feats'] = caption_embeddings.cpu()
        extracted_feats[cls]['file_paths'] = file_paths

        print(f'{(i+1)*100/len(classes)}% - Done - class: {cls} - time: {time.time() - start}')
    torch.save(extracted_feats,os.path.join(dataset_dir,f'{mined_folder.replace(f"{pre_training_corpus}-","")}-{arch_key.replace("/","-")}-{pre_training_corpus}.pth'))
    return extracted_feats


if __name__ == '__main__':

    if not os.path.exists('./data'):
        sys.exit('./data was not initialized, please intialize datasets first.')

    parser = argparse.ArgumentParser(description='Arguments for script.')
    parser.add_argument('--dataset', type=str, default='imagenet_1k', help='Dataset that is to be used.')
    parser.add_argument('--arch', type=str, default='ViT-B/32', help='Arch to extract features for.')
    parser.add_argument('--pre_training_corpus', type=str, default='laion400m', help='Pre-Training Corpus.')
    parser.add_argument('--device', type=int, default=0, help='GPU device id.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch Size.')
    args = parser.parse_args()
    
    # Pre-extract test sets.
    if not os.path.exists('./data/test_data'):
        os.makedirs('./data/test_data')
        pre_extract_test_data(
            device=args.device,
            batch_size=args.batch_size
        )

    
    torch.cuda.set_device(args.device)
    print(f'Loaded - {args.arch}')
    
    pre_extract_directory(
        mined_folder=f'retrieved_1m-{args.pre_training_corpus}-alternates-random', 
        arch_key=args.arch, pre_training_corpus=args.pre_training_corpus,
        dataset={args.dataset}, 
        args=args
    )