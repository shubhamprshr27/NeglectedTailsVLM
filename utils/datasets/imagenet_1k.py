import os
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import random

class ImageNet1K(Dataset):
    def __init__(self, root, split='val', transform=None):
        self.dataset_root = root
        self.split = split
        self.data_dict = json.load(open(f'./data/imagenet_1k/{split}_annotations.json', 'r'))
        self.image_names = list(self.data_dict.keys())
        self.image_file_paths = [os.path.join(self.dataset_root, self.split, self.data_dict[file_path],file_path) for file_path in self.image_names]
        self.transform = transform
        print(f'Loaded {len(self.image_names)} files.')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = self.image_file_paths[idx]
        name = self.image_names[idx]
        """self.split == 'mined' or self.split == 'cross_modal_lp'"""
        label = self.label_dict[self.data_dict[name]]
        image = default_loader(image_path)

        if self.transform:
            image = self.transform(image)

        if self.split == 'mined':
            return image, label, self.hfl_prompts[str(label)]["corpus"][0], image_path
        return image, label
