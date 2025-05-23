# NeglectedTailsVLM
This repository houses the code for the CVPR 2024 paper - "The Neglected Tails of Vision Language Models".

# Updates
2025-02-26: Our work [SWAT](https://tian1327.github.io/SWAT/) on applying Retrieval-Augmented Learning (RAL) for Few-shot Recognition (FSR) has been accepted to CVPR'25. Please check it out.

2024-08-10: Please check out our latest work [SWAT](https://tian1327.github.io/SWAT/) which explores retrieval-augmented learning for few-shot recognition!! See [RETRIEVAL.md](https://github.com/tian1327/SWAT/blob/master/retrieval/RETRIEVAL.md) for step-by-step instructions for retrieving pretraining data.

2024-04-20: Analysis code has been released. Documentation being updated along with release of data and captions.

2024-04-16: Code released for REAL-Linear and REAL-prompt, analysis code is on the way.

2024-02-26: Our paper has been accepted to CVPR 2024! Thanks for your patience, please expect the code to be released in upcoming weeks!

# Setup

This repository uses uses Python v3.11.4. Please create a new conda/miniconda environment using the `requirements.txt` file. Note that the CUDNN version used was 12.2.

# Dataset Setup

Coming soon!

# Execution

## Analyzing LAION400M and LAION2B
After downloading LAION400M and LAION2B from huggingface, the following steps need to be done to analyze the captions.
1. Retrieve captions containing class names. The code for doing this is in the file `/analysis/laion/laion_parser.py`, for e.g. LAION400M and ImageNet analysis can be run as:
```
python laion_parser.py --database LAION400M --downstream imagenet_1k --datasource {location_of_hf_download} --max_threads 16
```
2. After retrieving all the captions based on exact string matching we need to filter them using LLAMA. The code for this can be found in `/analysis/laion/llama/llama_miner.py`
   We use LLaMA-2-7B-chat for our analysis and the official LLaMA repo was adapted for our needs. Please refer to LLaMA's official repo to download the weights. Note for faster 
   inference it is adviceable to increase the world size to 8, but the code should work on a single GPU machine as well.
```
torchrun llama_miner.py --dataset imagenet_1k --world_size 8 --pre_trained_corpus LAION400M
```
&nbsp;&nbsp;&nbsp;The output of this step should be pickle files, each file is an ordered dict according to class IDs. The ordered dict contains relevant captions to the downstream concept names and filters out irrelevant ones, giving us a relevance metric which is defined as `(number of captions relevant to downstream task) / (total number of captions)` 
  
3. Now having filtered captions, we can estimate the frequency by running the file: `/analysis/tail_analysis.py`
```
python tail_analysis.py --dataset imagenet_1k --pre_trained_corpus LAION400M
```
4. If you would just like to access the frequency of a given concept in a given dataset, you can directly refer to `/analysis/laion/{dataset}/metrics-{pre_trained_corpus}.json` the `relevant_total` is the estimated frequency obtained after analyzing using LLaMA.


## Citation

If you find our project useful, please consider citing:

```bibtex
@inproceedings{liu2025few,
  title={Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning},
  author={Liu, Tian and Zhang, Huixin and Parashar, Shubham and Kong, Shu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@inproceedings{parashar2024neglected,
  title={The Neglected Tails in Vision-Language Models},
  author={Parashar, Shubham and Lin, Zhiqiu and Liu, Tian and Dong, Xiangjue and Li, Yanan and Ramanan, Deva and Caverlee, James and Kong, Shu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

```
