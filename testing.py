import torch
import time
import argparse
from utils.logger import get_logger
from utils.extras import get_engine, OPENCLIP_MODEL_DIC
from torch.utils.data import DataLoader
from models import MyLinear
from utils import features 
from utils.datasets.dataset_utils import get_dataset
from utils.prompt_templates import prompt_maker
from torchmetrics import ConfusionMatrix
import numpy as np
import pickle
import json
import os
from analysis.tail_analysis import calculate_head_tail_acc



def validate(data_loader, 
             model, 
             logger=None, 
             classifier_head = None, 
             Epoch=None, 
             show_confusion_matrix = False, 
             device='cuda', 
             grnd_truth = 'label',  
             pre_extracted=False):
    model.eval()
    if classifier_head is not None: # For later. When will run zero shot metrics.
        classifier_head.eval()
    val_acc = 0.
    val_count = 0
    num_classes = classifier_head.num_classes
    confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    
    for i, val_data in enumerate(data_loader):
        if grnd_truth == 'label':
            inputs, labels = val_data
        
        images = inputs.to(device)

        labels = labels.to(device).long()
        with torch.no_grad():
            if classifier_head:
                if not pre_extracted:
                    image_features = model.encode_image(images)
                else:
                    image_features = images
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logit = classifier_head(image_features) 
            else:
                raise ValueError("Classifier Head is None.")
        pred = torch.argmax(logit, dim=1)

        val_acc += torch.sum(pred == labels).item()
        val_count += labels.size(0)
        
        if show_confusion_matrix:
            preds = pred.cpu()
            labels = labels.cpu()
            confusion_matrix.update(preds, labels)
            
        images.cpu()
    val_acc = (val_acc/val_count)*100
    confusion_matrix = confusion_matrix.compute().numpy()

    if logger:
        logger.info(f'Top 1 validation accuracy: {val_acc} - epoch: {Epoch}')

    if show_confusion_matrix:
        return val_acc, confusion_matrix
    return val_acc

TEST_SPLIT = {
    'imagenet_1k': 'val',
    'imagenet_v2': 'val',
    'flowers102': 'test',
    'stanford_cars': 'test',
    'dtd': 'test',
    'fgvc_aircraft': 'test',
    'eurosat': 'test',
    'food101': 'test',
    'dtd': 'test',
    'caltech101': 'test',
    'oxford_pets': 'test',
    'sun397': 'test',
    'cub2011': 'test'
}    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for script.')
    parser.add_argument('--model_path', type=str, default='', help='learning rate for optimizer')
    parser.add_argument('--data_source', type=str, default='All', help='Data Source for Prompts.')
    parser.add_argument('--log_mode', type=str, default='stream', help='Out to Stream or FIle.')
    parser.add_argument('--sample_by', type=str, default='mean', help='Sampling for Text Prompts.')
    parser.add_argument('--mode', type=str, default='translated', help='Mode of Prompts.')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch Size.') # Validation only.
    parser.add_argument('--grnd_truth', type=str, default='label', help='Ground Truth - Labels / Text')
    parser.add_argument('--save_confusion_mat', action="store_true", help='Decide whether to save particular confusion matrix or not.')
    parser.add_argument('--tau', type=float, default=0, help='Tau for Normalization.')
    parser.add_argument('--cosine_sim', action="store_true", help='Use Cosine Similarity for checking.')
    parser.add_argument('--log_prefix', type=str, default='', help='Prefix for Log file Name.')
    parser.add_argument('--model_folder', type=str, default='finetuned', help='Model Folder Name')
    parser.add_argument('--use_react', action='store_true', help='use react')
    parser.add_argument('--arch', type=str, default='ViT-B/32', help='ViT Transformer arch.')
    parser.add_argument('--dataset', type=str, default='semi-inat-2021', help='Dataset that is to be used.')
    parser.add_argument('--topk', action='store_true', help='Top K for attributes.')
    parser.add_argument('--pretrained_corpus', type=str, default='laion400m', choices=['laion400m', 'laion2b', 'openai'], help='Pre-training corpus for OpenCLIP.')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu to use for testing.')
    parser.add_argument('--name_type', type=str, default='name', help='Choosing most common name or name.')
    parser.add_argument('--prompt_style', type=str, default='hand_engineered', choices=['hand_engineered','vanilla','only_name', 'dclip', 'cupl'])
    parser.add_argument('--ablate_prompts', action='store_true', help='Ablate all prompt techniques.')
    parser.add_argument('--use_metaclip', action='store_true', help='Ablate MetaCLIP')

    args = parser.parse_args()
    logger = get_logger(f'{args.log_prefix}_testing_{args.model_path}_{time.time()}_{args.dataset}', args.log_mode, True)
    dataset_root = f'/data3/sparasha/cliptail/research/data/{args.dataset}/'
    
    torch.cuda.empty_cache()
    torch.cuda.set_device(args.gpu)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    react_config= None

    if args.use_react:
        react_config = {
                'base_ckpt': f'openclip_laion400m',
                'retrieval_set': 'laion400m',
                'finetuning_mode': 'gated-image'
        }     
    model, preprocess, tokenizer = get_engine(arch=args.arch, corpus=args.pretrained_corpus, react_config=react_config)
    model.cuda()
    
    if args.use_metaclip:
        logger.info('Loading MetaCLIP')
        ckpt = torch.load('./METACLIP_b32_400m.pt')
        model.load_state_dict(ckpt['state_dict'])
    dataset_name = args.dataset
    
    if args.dataset.startswith('imagenet'):
        dataset_name = 'imagenet_1k'
    if args.use_metaclip or args.pretrained_corpus == 'openai':
        metrics = json.load(open(os.path.join('./analysis/laion', dataset_name, f'metrics-LAION400M.json')))
    else:
        metrics = json.load(open(os.path.join('./analysis/laion', dataset_name, f'metrics-{args.pretrained_corpus.upper()}.json')))

    text_prompts, label_map = prompt_maker(metrics=metrics, dataset_name=args.dataset, name_type=args.name_type, prompt_style=args.prompt_style)# json.load(open(dataset_root +'/prompts/' +args.prompts_file))
    
    prompt_tensors = features.get_text_features(model, text_prompts, logger=None, data_source = args.data_source, tokenize = tokenizer)
    weights = features.prompt_sampler(prompt_tensors, logger=None, sample_by=args.sample_by)
    classifier_head = MyLinear(weights = weights, bias=False,label_map=None) # Bias False when we want to use text-embeddings as weight.

    if args.tau !=0:
        classifier_head.linear.weight.data /= torch.pow(classifier_head.linear.weight.data.norm(dim=-1, keepdim=True), args.tau)

    classifier_head.to(device)

    split = TEST_SPLIT[args.dataset]
    _ ,model_arch = OPENCLIP_MODEL_DIC[args.pretrained_corpus][args.arch]
    pre_extracted_config = None
    if not args.use_react and not args.use_metaclip:
        pre_extracted_config = {'arch': model_arch, 'pre_trained_corpus': args.pretrained_corpus}
    val_dataset = get_dataset(dataset=args.dataset, dataset_root=dataset_root, preprocess=preprocess, 
                              split=split, pre_extracted_config=pre_extracted_config)
    
    arch_name = args.arch.replace('/','-')
    if args.ablate_prompts:
        num_workers = 0
    
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    val_acc, confusion_matrix = validate(val_dataloader, model, logger=None, classifier_head= classifier_head, show_confusion_matrix = True, 
                                         grnd_truth = args.grnd_truth, dataset=args.dataset, pre_extracted=(not args.use_react and not args.use_metaclip))
    model_name = f'{args.pretrained_corpus.upper()}_{arch_name}'

    head_acc, tail_acc = calculate_head_tail_acc(dataset=args.dataset, pretrained_dataset='LAION400M', 
                                                 confusion_matrix=confusion_matrix, method_name=f'{args.name_type}-{args.prompt_style}',
                                                 tail_ratio=0.2)
    print(f'{args.arch},{args.pretrained_corpus},{args.dataset},{args.prompt_style},{args.name_type},{round(val_acc,1)},{round(head_acc,1)},{round(tail_acc,1)}')

    if args.use_metaclip:
        model_name = f'MetaCLIP_{arch_name}'

    if not os.path.exists(f'./analysis/confusion_matrices/{args.dataset}'):
        os.makedirs(f'./analysis/confusion_matrices/{args.dataset}')
    if args.save_confusion_mat:
        with open(f'./analysis/confusion_matrices/{args.dataset}/{model_name}.pkl', 'wb') as f:
           pickle.dump(confusion_matrix, f)