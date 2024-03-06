import torch
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip, RandomCrop
from PIL import Image
import open_clip
from react_open_clip import create_model_and_transforms as react_create_model_and_transforms


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def flip_transform(n_px , mode='train'):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if mode == 'train':
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            # Add Center Crop. 
            RandomHorizontalFlip(),
            _convert_image_to_rgb,
            ToTensor(),
            normalize
        ])

OPENCLIP_MODEL_DIC = {
    'laion400m': {
        'ViT-B/32': ('laion400m_e32','ViT-B-32-quickgelu'),
        'ViT-B/16': ('laion400m_e32','ViT-B-16'),
        'ViT-L/14': ('laion400m_e32','ViT-L-14'),
    },
    'openai': {
        'ViT-B/32': ('openai','ViT-B-32-quickgelu'),
        'ViT-B/16': ('openai','ViT-B-16'),
        'ViT-L/14': ('openai','ViT-L-14')
    },
    'laion2b': {
        'ViT-B/32': ('laion2b_s34b_b79k','ViT-B-32'),
        'ViT-B/16': ('laion2b_s34b_b88k','ViT-B-16'),
        'ViT-L/14': ('laion2b_s32b_b82k','ViT-L-14')
    }
}


def get_engine(arch, mode='val', corpus='laion400m', react_config: dict = None):
    corpus_config ,model_arch = OPENCLIP_MODEL_DIC[corpus][arch]
    if react_config is not None:
        print('here')
        model, train_preprocess, preprocess = get_react_models(model_arch, react_config = react_config)
    else:
        model, train_preprocess, preprocess = open_clip.create_model_and_transforms(model_name = model_arch, pretrained=corpus_config)
        
    tokenizer = open_clip.get_tokenizer(model_arch)
    model = model.float() # Removes the mixed precision stuff.
    if mode == 'train':
        return model, train_preprocess, preprocess, tokenizer
    return model, preprocess, tokenizer

def get_react_models(arch_name, react_config):
    base_ckpt, retrieval_set, finetuning_mode = react_config.values()
    arch_name = arch_name.strip('-quickgelu') 
    if finetuning_mode == 'gated-image':
        arch_name = 'react_' + arch_name
    react_name = f'react_{base_ckpt}_ret_{retrieval_set}'
    model, train_preprocess, val_preprocess = react_create_model_and_transforms(model_name = arch_name, pretrained=react_name)
    return model, train_preprocess, val_preprocess


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model
    
    def forward(self, text):
        return self.model.encode_text(text)
    
class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model
    
    def forward(self, image):
        return self.model.encode_image(image)