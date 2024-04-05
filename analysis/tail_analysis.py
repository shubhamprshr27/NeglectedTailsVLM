import pickle
import json
import numpy as np
import os

def calculate_true_freq(metrics, llama_stats = {}):
    for i, key in enumerate(metrics):
        relevance = llama_stats[key]['relevance']
        metrics[key]['relevant_total'] = relevance * metrics[key]['actual_freq']

    return dict(sorted(metrics.items(), key=lambda item: item[1]['relevant_total'], reverse=False))

ANALYSIS_PATH = './'

def tail_hypothesis(metrics, confusion_matrices, tail_ratio=0.2):
    head_acc = [[] for _ in enumerate(confusion_matrices)]
    tail_acc = [[] for _ in enumerate(confusion_matrices)]

    for i, key in enumerate(metrics):
        for cm_idx, confusion_matrix_data in enumerate(confusion_matrices):
            _, confusion_matrix = confusion_matrix_data['name'], confusion_matrix_data['data']
            class_acc: float = confusion_matrix[int(key)][int(key)]/ np.sum(confusion_matrix[int(key)])
            if i <= int(tail_ratio*len(metrics.items())):
                tail_acc[cm_idx].append(class_acc)
            else:
                head_acc[cm_idx].append(class_acc)
    return head_acc, tail_acc

def calculate_head_tail_acc(dataset, pretrained_dataset, confusion_matrix, method_name='zero_shot', tail_ratio=0.2):
    metrics_path = os.path.join(ANALYSIS_PATH,'laion', dataset, f'metrics-{pretrained_dataset.upper()}.json') 
    metrics = json.load(open(metrics_path, 'r'))

    llama_stats = pickle.load(open(os.path.join(ANALYSIS_PATH,'laion', dataset,f'analysis-llama-{pretrained_dataset.upper()}'), 'rb'))
    metrics = calculate_true_freq(metrics=metrics, llama_stats=llama_stats)
    
    head_acc, tail_acc = tail_hypothesis(metrics=metrics, confusion_matrices=[{'name':method_name,'data':confusion_matrix}], tail_ratio=tail_ratio)
    return sum(head_acc[0])*100/len(head_acc[0]), sum(tail_acc[0])*100/len(tail_acc[0])