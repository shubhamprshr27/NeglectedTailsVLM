import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os

class GenericDataset(Dataset):
    def __init__(self, root, transform=None, split='test', dataset_name = 'caltech101'):
        self.dataset_name = dataset_name
        self.split = split
        self.root_dir = root
        self.split_file = json.load(open(os.path.join(self.root_dir, 'split.json')))
        self.annotations = self.split_file[self.split]
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # img_name, label, label_name = self.annotations[idx]
        img_name = self.annotations[idx][0]
        label = self.annotations[idx][1]
        img_name = os.path.join(self.root_dir, img_name)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        
        return image, label
        