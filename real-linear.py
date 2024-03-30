from utils.extras import get_engine
from utils.prompt_templates import make_per_class_prompts
from PIL import Image
import numpy as np
import os
import torch
import time
from utils import features
import json
import random
import argparse
from pre_extract_features import pre_extract_directory, save_dataset_feats
from utils.prompt_templates import prompt_maker
from clip_cross_modal import train
from analysis.tail_analysis import calculate_head_tail_acc
import copy

"""
    Function to calculate image . prompt similarity.
"""
def calc_similarity(model, preprocess, file_list, class_prompt, embeddings = None):
    if embeddings is None:
        img_tensors = [preprocess(Image.open(file)).unsqueeze(0) for file in file_list]
        img_tensors = torch.cat(img_tensors, dim=0).cuda() 
        with torch.no_grad():
            embeddings = model.encode_image(img_tensors)
        embeddings /=embeddings.norm(dim=-1, keepdim=True)
    similarity = embeddings.cuda() @ class_prompt.t()
    # Calculate the average similarity across alternate names.
    if similarity.shape[-1] > 1:
        similarity = torch.mean(similarity, dim=-1)
    return similarity.squeeze().cpu().tolist()

"""
    Mode can be name, most_common_name and all alternate labels.
"""
def get_class_prompts(metrics, class_idx, name_type='name', dataset='imagenet_1k'):
    class_prompts = make_per_class_prompts(metrics=metrics, class_idx=class_idx, name_type='alternates', dataset=args.dataset)
    prompt_tensors = features.get_text_features(model, class_prompts, logger=None, data_source = 'All')
    prompt_embeddings = features.prompt_sampler(prompt_tensors, logger=None, sample_by='mean')
    return prompt_embeddings

def add_to_split(cls:int, mined_split: dict, 
                 imgpaths_sim_zip: list, num_samples: int, 
                 label_name:str,):
    sampled_files = 0
    feature_list = []
    label_list = []
    for i, (file_path, similarity, embedding) in enumerate(imgpaths_sim_zip):
        if sampled_files == num_samples:
            break
        # Sample images above 0 CLIP score(prompt,image)
        if similarity >= 0:
            sampled_files +=1 
            feature_list.append(embedding)
            label_list.append(int(cls))
            mined_split['train']['data'].append({'impath': file_path, 'label': int(cls), 'classname': label_name}) 
    return mined_split, feature_list, label_list       

"""
    This function is used to sample data used for REAL-Linear.
    The idea is to sample images based on their T2I similarity to downstream prompts, containing REAL-Prompt labels.
"""
def real_sampler(root_folder, 
                   metrics, 
                   num_samples=100, 
                   model=None, 
                   preprocess=None, 
                   name_type='name', 
                   pre_extracted_feats = None):
    classes = pre_extracted_feats.keys()
    if model is not None:
        model = model.cuda()
    mined_split = {'train': {'data': []}}
    start = time.time()
    all_features = []
    all_labels = []
    for cls in classes:
        img_embeddings = None

        # No data for this class, just go forward.
        if cls.endswith('parquet') or cls.endswith('json'):
            continue
        if pre_extracted_feats[cls]['feats'] is None:
            continue

        file_list = pre_extracted_feats[cls]['file_paths'] 
        img_embeddings = pre_extracted_feats[cls]['feats']
        caption_embeddings = pre_extracted_feats[cls]['caption_feats']
        class_prompt = get_class_prompts(metrics=metrics, class_idx=int(cls), name_type=name_type, dataset=args.dataset)
        similarity = calc_similarity(model, preprocess,file_list, class_prompt, caption_embeddings)

        if isinstance(similarity, float):
            similarity = [similarity]

        embedding_list = [img_embeddings[i] for i in range(len(img_embeddings))]
        path_sim_zip = sorted(list(zip(file_list, similarity, embedding_list)), key=lambda x: x[1], reverse=True)

        label_name = metrics[cls]['most_common_name']

        mined_split, feature_list, label_list  = add_to_split(int(cls),
                                                              mined_split, 
                                                              path_sim_zip, 
                                                              num_samples, 
                                                              label_name)
        all_features.extend(feature_list)
        all_labels.extend(torch.tensor(label_list))
    print('Time taken for random sampling:', time.time()-start)
    return mined_split, all_features, all_labels 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for script.')
    parser.add_argument('--dataset', type=str, default='imagenet_1k', help='dataset')
    parser.add_argument('--pre_training_dataset', type=str, default='laion400m', help='Pre-training dataset to fetch images from.')
    parser.add_argument('--arch', type=str, default='ViT-B/32', help='OpenCLIP Architecture.')
    parser.add_argument('--image_label_type', type=str, default='most_common_name', choices=['most_common_name', 'alternates', 'name'], help='What label to use for ranking images.')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of images that are to be sampled.')
    parser.add_argument('--case_num', type=int, default=0, help='Ablation study case.')
    parser.add_argument('--max_iters', type=int, default=32000, help='Max number of iterations.')
    parser.add_argument('--prompt_name_type', type=str, default='most_common_name')
    parser.add_argument('--gpu', type=int, default=0)
   
    args = parser.parse_args()
    arch_name = args.arch.replace("/", "-")
    retrieved_folder_name = f'retrieved_1m-alternates-random'
    dataset_root = f'./data/{args.dataset}'
    root_folder = f'./data/{args.dataset}/{retrieved_folder_name}'
    metrics = json.load(open(f'./analysis/laion/{args.dataset}/metrics-{args.pre_training_dataset.upper()}.json', 'r'))
    orgingal_metrics = copy.deepcopy(metrics)
    text_prompts, _ = prompt_maker(metrics=metrics, dataset_name=args.dataset, name_type=args.prompt_name_type)
    
    device = 'cuda'
    torch.cuda.set_device(args.gpu)
    model, train_preprocess, preprocess, tokenizer = get_engine(arch=args.arch, corpus=args.pre_training_dataset, mode='train')


    # random.seed()
    # torch.manual_seed()

    # Check whether pre_extracted_feats exist.
    pre_extracted_feats = None
    print(os.path.join(dataset_root, f'{retrieved_folder_name}-{args.arch.replace("/", "-")}-{args.pre_training_dataset}.pth'))
    if os.path.exists(os.path.join(dataset_root, f'{retrieved_folder_name}-{args.arch.replace("/", "-")}-{args.pre_training_dataset}.pth')):
        print('Pre-Extracted Features found.')
        pre_extracted_feats = torch.load(os.path.join(dataset_root, f'{retrieved_folder_name}-{args.arch.replace("/", "-")}-{args.pre_training_dataset}.pth'))
    else:
        print('Pre-Extracted Features not found, extracting first')
        pre_extracted_feats = pre_extract_directory(retrieved_folder_name, args.arch, args.pre_training_dataset, f'./data/{args.dataset}')
    
    # Sample images for training.
    mined_split, features_list, labels_list = real_sampler(root_folder=root_folder, 
                                                            metrics=metrics, 
                                                            num_samples=args.num_samples, 
                                                            model=model, 
                                                            preprocess=preprocess, 
                                                            name_type=args.image_label_type, 
                                                            pre_extracted_feats = pre_extracted_feats)

    # Evaluation
    # Split Pre-extraction + Cross Modal Probing.
    filename = f'case_{args.case_num}_{args.dataset}_{args.pre_training_dataset}_{args.arch.replace("/","")}'\
                f'_{args.image_label_type}_{args.num_samples}'

    img_tensor = torch.stack(features_list)
    labels_tensor = torch.stack(labels_list)
    dataset_dict={'image_features': img_tensor, 'labels': labels_tensor}
    print(img_tensor.shape, labels_tensor.shape)

    pre_extracted_path = save_dataset_feats(dataset_dict=dataset_dict,
                        dataset_name=f'{args.dataset}_mined',
                        split='mined',
                        pre_training_corpus=args.pre_training_dataset,
                        arch=arch_name,
                        shots=args.num_samples)

    # Not more than 10 epochs. 32000 iters for im1k is ~10 epochs.
    num_iters = min(32000, args.num_samples * len(orgingal_metrics.items()) * 10/ 32)
    print('iters',num_iters)
    
    # Cross-Modal training.
    val_acc, best_head, confusion_matrices = train(
        arch=args.arch,
        pre_training_corpus=args.pre_training_dataset,
        prompts=text_prompts,
        shots=args.num_samples,
        wise_ft_alpha=.5,
        logit_scale=4.60517, # soft-max logit scale
        bsz=32, # fixed zero-shot hyper-parameter from previous literature.
        lr=1e-4, # fixed zero-shot hyper-parameter from previous literature.
        wd=0.01, # fixed zero-shot hyper-parameter from previous literature.
        dataset=args.dataset,
        extracted_feats_path=pre_extracted_path,
        max_iters=int(num_iters)
    )

    tail_ratio = 0.2
    head_acc = 0.
    tail_acc = 0.
    head_acc, tail_acc = calculate_head_tail_acc(dataset=args.dataset, pretrained_dataset=args.pre_training_dataset, 
                                                 confusion_matrix=confusion_matrices[1], method_name=f'REAL', tail_ratio=tail_ratio)
    torch.save(best_head.state_dict(), f'{filename}.pkl')

    model_name = f'{args.pre_training_dataset}_{arch_name}'
    print(f"{args.dataset},{args.num_samples},{args.prompt_name_type},{args.pre_training_dataset},{args.image_label_type},{args.arch},{round(val_acc,1)},{tail_acc},{head_acc}")

