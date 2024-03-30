import clip
import torch
import random
import time

def prompt_sampler(prompt_tensors, logger = None, sample_by='mean'):
    sampled_prompts = []
    # print(f'Sampling as per: {sample_by}')
    if logger is not None:
        logger.info(f'Sampling as per: {sample_by}')
    for i in range(len(prompt_tensors)):
        if sample_by == 'mean':
            sampled_prompts.append(prompt_tensors[i]['mean'])
        elif sample_by == 'random':
            sampled_prompts.append(random.choice(prompt_tensors[i]['all']))
    stacked_prompts = torch.stack(sampled_prompts)
    return stacked_prompts

def operate_on_prompt(model, text, operation, tokenize = clip.tokenize):
    if operation=='encode':
        with torch.no_grad():
            text_tokens = tokenize(text).cuda()
            features = model.encode_text(tokenize(text).cuda())
            text_tokens.cpu()
        features /= features.norm(dim=-1, keepdim=True) # Normalization.
        return features
    elif operation == 'tokenize':
        tokens = tokenize(text)
        return tokens

def get_text_features(model, prompt_dict, logger=None, operation='encode', tokenize=clip.tokenize):
    tensor_list = []
    with torch.no_grad():
        start = time.time()
        for i, species in enumerate(prompt_dict):
            obj = prompt_dict[species]
            source = {}
            if i % 100 == 0:
                if logger: logger.info(f'Starting feature extraction for class: {i}')
            prompts = []
            for prompt in obj['corpus']:
                prompts.append(prompt)
            stacked_tensor = operate_on_prompt(model, prompts, operation, tokenize)
            source['all'] = stacked_tensor
            if operation == 'encode':
                mean_tensor = torch.mean(stacked_tensor, dim=0)
                mean_tensor /= mean_tensor.norm(dim=-1, keepdim=True) # torch.linalg.vector_norm(mean_tensor, dim=-1, keepdim=True)
                source['mean'] = mean_tensor
            tensor_list.append(source)
        return tensor_list
    