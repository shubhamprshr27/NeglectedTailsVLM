import torch
import os
from .dataset_utils import SPLIT_NAMES
import json

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, dataset, arch, pre_trained_corpus, split='test', shots=None, tags=None, base_path = None):
        self.split = SPLIT_NAMES[split][dataset]
        if base_path is None:
            base_path = f'{dataset}_{self.split}_{pre_trained_corpus}_{arch}'
            if shots is not None:
                base_path = base_path + f'_{shots}'
            if tags is not None:
                base_path = base_path + f'_{tags}'
        print(base_path)
        pre_extracted_path = os.path.join(dataset_root, 'pre_extracted', dataset, f'{base_path}.pkl')
        self.dataset = torch.load(pre_extracted_path)
        self.input_tensor = self.dataset['image_features']
        self.label_tensor = self.dataset['labels']
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index]
    
    def __len__(self):
        return self.input_tensor.size(0)

class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, model, tokenizer, prompts):
        self.prompts = prompts
        labels_flat = None
        prompt_tensors = None
        if torch.cuda.is_available():
            model.cuda()
        for i, key in enumerate(self.prompts.keys()):
            labels = torch.Tensor([i for _ in range(len(self.prompts[key]['corpus']))]).long()
            prompt_embeddings = model.encode_text(tokenizer([prompt for prompt in self.prompts[key]['corpus']]).cuda()).cpu()
            with torch.no_grad():
                if prompt_tensors is None:
                    prompt_tensors = prompt_embeddings
                    labels_flat = labels
                else:
                    prompt_tensors = torch.cat((prompt_tensors, prompt_embeddings), dim=0)
                    labels_flat = torch.cat((labels_flat, labels))
        self.labels = labels_flat
        self.prompt_tensors = prompt_tensors
        print('Loaded Text Tensor Augmentations - ', self.prompt_tensors.shape, self.labels.shape)
    
    def __getitem__(self, index):
        return self.prompt_tensors[index], self.labels[index]
    
    def __len__(self):
        return self.prompt_tensors.size(0)

