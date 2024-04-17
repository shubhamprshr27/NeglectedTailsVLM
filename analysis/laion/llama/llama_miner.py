from typing import Optional
import json
import pickle
import time
import torch
import os
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp 
from llama import Llama

CKPT_DIR = 'llama-2-7b-chat/'
TOKENIZER_PATH = 'tokenizer.model'
TEMPERATURE = 0.6
TOP_P = 0.9
MAX_SEQ_LEN = 1024
MAX_GEN_LEN = 100
MAX_BATCH_SIZE = 64
LIMIT = 10000

LLAMA_PROMPT_TEMPLATES = {
    'imagenet_1k': 'Does the "{name}" in "{caption}", refer to the visual concept: "{name}" - "{definition}"?',
    'flowers102': 'Does the "{name}" in "{caption}", refer to the flower: "{name}" - "{definition}"?',
    'oxford_pets': 'Does the "{name}" in "{caption}", refer to the pet: "{name}"?',
    'fgvc_aircraft': 'Does the "{name}" in "{caption}", refer to the aircraft: "{name}" ?',
    'stanford_cars': 'Does the "{name}" in "{caption}", refer to the car: "{name}" ?',
    "food101": 'Does the "{name}" in "{caption}", refer to the food dish: "{name}" - "{definition}"?',
    'dtd': 'Does the "{name}" in "{caption}" describe a texture, pattern or item?',
    'eurosat': 'Does the "{caption}" refer to a satellite image of the {name} ?',
    'cub2011': 'Does the "{name}" in "{caption}", refer to the bird: "{name}"'
}


def divide_data(data, 
                k, 
                rank):
    """
    Divide a dictionary into k sets based on process rank.

    Parameters:
    - data: The dictionary to be divided.
    - k: The total number of sets or processes.
    - rank: The rank of the current process (0-indexed).

    Returns:
    - A sub-dictionary for the given rank.
    """
    
    keys = list(data.keys())
    chunk_size = len(keys) // k
    remainder = len(keys) % k

    # Calculate start and end indices for slicing
    start_idx = rank * chunk_size + min(rank, remainder)
    end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)

    # Extract keys for this chunk
    chunk_keys = keys[start_idx:end_idx]
    # Create a sub-dictionary based on the chunk keys
    return {key: data[key] for key in chunk_keys}

def filter_captions(generator, 
                    hfl_name, 
                    captions_data, 
                    prompt_template: str, 
                    definition: str=None, 
                    limit=10000, 
                    temperature: float = 0.6,
                    top_p: float = 0.9,
                    max_seq_len: int = 512, 
                    max_gen_len: Optional[int] = None):
    dialogs = []
    name = hfl_name
    for data in captions_data:
        # if len(data) == 3:
        #     _,_,caption = data
        # else:
        name,_,_,caption = data
        if len(caption.split()) > 100: # To avoid the noisy captions.
            caption = 'Too long a caption.'
        if definition is not None:
            prompt = prompt_template.format(name=name, caption=caption, definition=definition)
        else:
            prompt = prompt_template.format(name=name, caption=caption)
        dialogs.append([
            {"role": "system", "content": "Just answer Yes or No."},
            # {"role": "user", "content": f'Given the sentence "{caption.lower()}", can it be used for an image of one or more - "{hfl_name}", defined as - "{definition.strip()}"?'}
            {"role": "user", "content": prompt}
        ])
    final_results = []
    for d_i in range(0, len(dialogs), MAX_BATCH_SIZE): # Specific to Batch Size
        chunks = dialogs[d_i:d_i+MAX_BATCH_SIZE]

        # This piece of code is a last ditch exception handling.
        # Such errors can happen due to some problem with large batch sizes and LLAMA-2 weights
        # https://github.com/facebookresearch/llama/issues/380
        llama_exception = False
        for batch_index in range(MAX_BATCH_SIZE):
            if llama_exception: # Reduce batch size to 1, incase of exception.
                chunks = [dialogs[d_i+batch_index]]
            try:
                results = generator.chat_completion(
                    chunks,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                parsed_results = [result['generation']['content'].strip().strip('.').lower() for result in results]
                final_results = final_results + parsed_results
                if not llama_exception:
                    break
            except Exception as e:
                llama_exception = True
                print(f'Exception happened in inference for {hfl_name} - ', e)
        if (len(final_results) >= 1000 and final_results.count('yes') >= 200) or d_i > limit: # Atleast 1000 seen or limit.
            break
    captions_data_final = [(data[0],data[1],data[2],result) for data, result in zip(captions_data[:len(final_results)], final_results)]
    if len(final_results) == 0:
        return {'data': captions_data_final, 'relevance': 0}

    return {'data': captions_data_final, 'relevance': final_results.count('yes')/len(final_results)}

def run_inference_worker(
    rank: int  = 0,
    world_size: int = 3,
    pre_trained_corpus = '',
    root_dir:str = '',
    dataset: str = ''
):  
    dist.init_process_group(rank=rank, world_size=world_size)
    
    generator = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        local_rank=rank
    )

    print('LLAMA Model Load - Complete', rank, torch.cuda.current_device(), flush=True) 
    METRICS = json.load(open(os.path.join(root_dir, dataset,f'metrics-{pre_trained_corpus}.json'), 'r'))
    if os.path.exists(os.path.join('definitions', f'{args.dataset}.json')):
        defs = json.load(open(os.path.join('definitions', f'{dataset}.json')))
    
    start = time.time()
    MINED_CAPTIONS = pickle.load(open(os.path.join(root_dir, dataset, f'mined_captions-{pre_trained_corpus}{"-satellite" if dataset == "eurosat" else ""}'), 'rb'))
    
    # Sort to maintain order.
    METRICS = dict(sorted(METRICS.items(), key = lambda x: int(x[0])))
    MINED_CAPTIONS = dict(sorted(MINED_CAPTIONS.items(), key = lambda x: int(x[0])))
    
    relevant_captions = {}
    metrics_dist = divide_data(METRICS, world_size, rank)
    mined_captions_dist = divide_data(MINED_CAPTIONS, world_size, rank)
    print(f'Files loaded - {time.time() - start} - {rank}')
    
    prompt_template = LLAMA_PROMPT_TEMPLATES[dataset]
    best_def = None

    for i, key in enumerate(metrics_dist):
        hfl_name = metrics_dist[key]['most_common_name']
        captions_data = list(mined_captions_dist[key])
        best_def = defs[key]['definition']

        relevant_caption_data = filter_captions(generator, hfl_name, captions_data, prompt_template, best_def, limit=10000)
        print(i,key,hfl_name,f'{time.time()-start}', 'relevance', relevant_caption_data['relevance'], 'Process rank:', rank, flush=True)
        
        relevant_captions[key] = relevant_caption_data
        with open(os.path.join(root_dir,dataset,f'./analysis-llama-{pre_trained_corpus}-{rank}'),'wb') as f:
            pickle.dump(relevant_captions, file=f)


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Arguments for script.')
    parser.add_argument('--root_dir', type=str, default='..', help='Root Dir, where all metrics are present')
    parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs to parallelize on.')
    parser.add_argument('--pre_trained_corpus', type=str, default='LAION400M')
    parser.add_argument('--dataset', type=str, default='imagenet_1k', help='Dataset to analyze for.')
    
    args = parser.parse_args()

    print(f'Estimating frequency for: {args.dataset}')

    mp.spawn(run_inference_worker, 
             args=(args.world_size,'mine', args.pre_trained_corpus, args.root_dir, args.dataset), 
             nprocs=args.world_size, join=True)
    retrieval_llama = {}
    
    for rank in range(args.world_size):
        chunk = pickle.load(open(os.path.join(args.root_dir,args.dataset,f'./analysis-llama-{args.pre_trained_corpus}-{rank}'), 'rb'))
        retrieval_llama.update(chunk)
    with open(os.path.join(args.root_dir,args.dataset,f'analysis-llama-{args.pre_trained_corpus}'),'wb') as f:
        pickle.dump(retrieval_llama, file=f)
