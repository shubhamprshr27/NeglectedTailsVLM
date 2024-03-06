import json
from collections import defaultdict
caltech101_templates = [
    'a photo of a {}.',
    'a painting of a {}.',
    'a plastic {}.',
    'a sculpture of a {}.',
    'a sketch of a {}.',
    'a tattoo of a {}.',
    'a toy {}.',
    'a rendition of a {}.',
    'a embroidered {}.',
    'a cartoon {}.',
    'a {} in a video game.',
    'a plushie {}.',
    'a origami {}.',
    'art of a {}.',
    'graffiti of a {}.',
    'a drawing of a {}.',
    'a doodle of a {}.',
    'a photo of the {}.',
    'a painting of the {}.',
    'the plastic {}.',
    'a sculpture of the {}.',
    'a sketch of the {}.',
    'a tattoo of the {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'the embroidered {}.',
    'the cartoon {}.',
    'the {} in a video game.',
    'the plushie {}.',
    'the origami {}.',
    'art of the {}.',
    'graffiti of the {}.',
    'a drawing of the {}.',
    'a doodle of the {}.',
]

describabletextures_templates = [
    'a photo of a {} texture.',
    'a photo of a {} pattern.',
    'a photo of a {} thing.',
    'a photo of a {} object.',
    'a photo of the {} texture.',
    'a photo of the {} pattern.',
    'a photo of the {} thing.',
    'a photo of the {} object.',
]

eurosat_templates = [
    'a centered satellite photo of {}.',
    'a centered satellite photo of a {}.',
    'a centered satellite photo of the {}.',
]

fgvcaircraft_templates = [
    'a photo of a {}, a type of aircraft.',
    'a photo of the {}, a type of aircraft.',
]

flowers102_templates = [
    'a photo of a {}, a type of flower.',
]

food101_templates = [
    'a photo of {}, a type of food.',
]

oxfordpets_templates = [
    'a photo of a {}, a type of pet.',
]

semi_aves_templates = [
    'a photo of a {}, a type of bird.'
]

cub2011_templates = [
    'a photo of a {}, a type of bird.',
    'a photo of a {}, in the wild.'
]

sun397_templates = [
    'a photo of a {}.',
    'a photo of the {}.',
]

stanfordcars_templates = [
    'a photo of a {}.',
    'a photo of the {}.',
    'a photo of my {}.',
    'i love my {}!',
    'a photo of my dirty {}.',
    'a photo of my clean {}.',
    'a photo of my new {}.',
    'a photo of my old {}.',
]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

caltech101_templates = [
    'a photo of a {}.',
    'a painting of a {}.',
    'a plastic {}.',
    'a sculpture of a {}.',
    'a sketch of a {}.',
    'a tattoo of a {}.',
    'a toy {}.',
    'a rendition of a {}.',
    'a embroidered {}.',
    'a cartoon {}.',
    'a {} in a video game.',
    'a plushie {}.',
    'a origami {}.',
    'art of a {}.',
    'graffiti of a {}.',
    'a drawing of a {}.',
    'a doodle of a {}.',
    'a photo of the {}.',
    'a painting of the {}.',
    'the plastic {}.',
    'a sculpture of the {}.',
    'a sketch of the {}.',
    'a tattoo of the {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'the embroidered {}.',
    'the cartoon {}.',
    'the {} in a video game.',
    'the plushie {}.',
    'the origami {}.',
    'art of the {}.',
    'graffiti of the {}.',
    'a drawing of the {}.',
    'a doodle of the {}.',
]



hand_engineered_templates = {
    'imagenet_1k': imagenet_templates,
    'flowers102': flowers102_templates,
    'food101': food101_templates,
    'stanford_cars': stanfordcars_templates,
    'fgvc_aircraft': fgvcaircraft_templates,
    'oxford_pets': oxfordpets_templates,
    'imagenet_v2': imagenet_templates,
    'dtd': describabletextures_templates,
    'semi-aves': semi_aves_templates,
    'caltech101': caltech101_templates,
    'eurosat': eurosat_templates,
    'sun397': sun397_templates,
    'cub2011': cub2011_templates
}

only_name_templates = defaultdict(lambda: ['{}'])
vanilla_templates = defaultdict(lambda: ['A photo of a {}'])

all_templates = {
    'only_name': only_name_templates,
    'vanilla': vanilla_templates,
    'hand_engineered': hand_engineered_templates
}


def prompt_maker(metrics: dict, dataset_name: str, name_type='name', prompt_style='hand_engineered'):
    prompts = {}
    label_map = {}
    if prompt_style in ['only_name', 'vanilla', 'hand_engineered']:
        prompt_templates = all_templates[prompt_style][dataset_name]
        if name_type == 'alternates':
            for i, key in enumerate(metrics.keys()):
                label = metrics[key][name_type]
                most_common_name = metrics[key]['most_common_name']
                
                if name_type == 'alternates':
                    prompt_lst = []
                    for alt_name, ct in label.items():
                        if alt_name != most_common_name: 
                            formatted_alt_name = alt_name  
                        else:
                            formatted_alt_name = most_common_name             
                        pt = [template.format(formatted_alt_name) for template in prompt_templates]
                        prompt_lst.extend(pt)
                    prompts[key] = {'corpus': prompt_lst}
        else:
            for i, key in enumerate(metrics.keys()):
                label = metrics[key][name_type]
                prompts[key] = {'corpus': [template.format(label) for template in prompt_templates]}
    else:
        prompts = json.load(open(f'./zero_shot_prompts/{prompt_style}/{dataset_name}.json'))
        for i, key in enumerate(prompts):
            if len(prompts[key]['corpus'])!=0:
                prompts[key]['corpus'] = [prompt.replace(prompts[key]['name'], metrics[key][name_type]) for prompt in prompts[key]['corpus']]
            else:
                print(metrics[key][name_type])
                prompts[key]['corpus'] = [f'A photo of a {metrics[key][name_type]}']
    prompts = dict(sorted(prompts.items(), key= lambda x: int(x[0])))
    return prompts, label_map


def make_per_class_prompts(metrics: dict, class_idx: int, name_type: str, dataset: str):
    class_metrics = metrics[str(class_idx)]
    prompt_templates = hand_engineered_templates[dataset]
    prompts = {}
    if name_type == 'alternates':
        for i, key in enumerate(class_metrics['alternates']):
            prompts[i] = {'corpus': [template.format(key) for template in prompt_templates]}
    else:
        label = class_metrics[name_type]
        prompts[0] = {'corpus': [template.format(label) for template in prompt_templates]}
    return prompts
        

def make_alt_name_prompts(metrics: dict, prompt_templates):
    prompts = {}
    alt_counter = 0
    label_map = {}
    for i, key in enumerate(metrics.keys()):
        label_map[i] = []
        for j, alt_name in enumerate(metrics[str(i)]['alternates']):
            prompts[alt_counter] = {'corpus': [template.format(alt_name) for template in prompt_templates]}
            label_map[i].append(alt_counter)
            alt_counter +=1
    return prompts, label_map