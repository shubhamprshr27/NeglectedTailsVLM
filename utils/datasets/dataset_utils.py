from torchvision.datasets import (
    Flowers102, 
    FGVCAircraft, 
    DTD,
    Food101,
    OxfordIIITPet,
    EuroSAT,
)
import os
import torch
from .CUB200 import Cub2011
from .GenericDataset import GenericDataset
from .inat_dataset import iNatDataset
from .imagenet_1k import ImageNet1K
from imagenetv2_pytorch import ImageNetV2Dataset
from torchvision.datasets.folder import default_loader

SPLIT_NAMES = {
    'val': {
        'flowers102': 'val',
        'cub2011': 'val',
        'imagenet_1k': 'val',
        'imagenet_v2': 'val',
        'semi-inat-2021': 'val',
        'semi-aves': 'train_val',
        'fgvc_aircraft': 'val',
        'stanford_cars': 'val',
        'dtd': 'val'
    },
    'test': {
        'flowers102': 'test',
        'cub2011': 'test',
        'imagenet_1k': 'val',
        'imagenet_v2': 'val',
        'imagenet_sketch': 'val',
        'semi-inat-2021': 'val',
        'semi-aves': 'train_val',
        'fgvc_aircraft': 'test',
        'stanford_cars': 'test',
        'eurosat': 'test',
        'caltech101': 'test',
        'dtd': 'test',
        'food101': 'test',
        'oxford_pets': 'test',
        'sun397': 'test'
    },
    'mined':{
        'imagenet_1k_mined': 'mined',
        'imagenet_1k_zs_mined': 'mined',
        'flowers102_mined': 'mined',
        'cub2011_mined': 'mined',
        'fgvc_aircraft_mined': 'mined',
        'food101_mined': 'mined',
        'oxford_pets_mined': 'mined',
        'stanford_cars_mined': 'mined',
        'dtd_mined': 'mined',
        'eurosat_mined': 'mined'
    },
    'train': {
        'imagenet_1k': 'train',
    }
}

from .tensor_dataset import TensorDataset

def get_dataset(dataset, 
                dataset_root, 
                preprocess, 
                split, 
                pre_extracted_config: dict = None):
    if split in SPLIT_NAMES:
        split = SPLIT_NAMES[split][dataset]
    # Caltech101(root=dataset_root, transform=preprocess, split=split)
    if pre_extracted_config is not None:
        torch_dataset = TensorDataset(dataset_root='./data', split='test', dataset=dataset, 
                                    arch=pre_extracted_config['arch'].replace('/','-'), 
                                    pre_trained_corpus=pre_extracted_config['pre_trained_corpus'])
        return torch_dataset    
    
    if not os.path.exists(dataset_root): 
        os.makedirs(dataset_root)
    if dataset == 'flowers102':
        torch_dataset = Flowers102(root=dataset_root, split=split, transform=preprocess, download=True)
    elif dataset == 'cub2011':
        torch_dataset = Cub2011(root=dataset_root, train=False, transform=preprocess, download=True)
    elif dataset == 'imagenet_1k':
        torch_dataset = ImageNet1K(root=dataset_root,split=split, transform=preprocess)
    elif dataset == 'imagenet_v2':
        torch_dataset = ImageNetV2Dataset(transform=preprocess, location=dataset_root)
    elif dataset == 'fgvc_aircraft':
        torch_dataset = FGVCAircraft(root=dataset_root, split=split, transform=preprocess, download=True)
    elif dataset in ['stanford_cars', 'caltech101', 'eurosat', 'sun397', 'imagenet_sketch']:
        torch_dataset = GenericDataset(root=dataset_root, split=split, transform=preprocess, dataset_name=dataset)
    elif dataset == 'dtd':
        torch_dataset = DTD(root=dataset_root, split=split, transform=preprocess, download=True) 
    elif dataset == 'food101':
        torch_dataset = Food101(root=dataset_root, split=split, transform=preprocess, download=True)
    elif dataset == 'oxford_pets':
        torch_dataset = OxfordIIITPet(root=dataset_root, split=split, transform = preprocess, download=True)  
    elif dataset == 'eurosat':
        torch_dataset = EuroSAT(root=dataset_root, transform = preprocess, download=True)   
    else:
        torch_dataset = iNatDataset(dataset_root, split, dataset, transform=preprocess)
    # quit()
    return torch_dataset

class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, data_source, transform):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        img = self.transform(default_loader(item['impath']))

        output = {
            "img": img,
            "label": item['label'],
            "classname": item['classname'],
            "impath": item['impath'],
        }

        return output['img'], output['label']
