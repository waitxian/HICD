import os
import argparse
import pickle
import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  

parser = argparse.ArgumentParser()
parser.add_argument("--saved_head_importance_path_ac", type=str, default=None)
parser.add_argument("--saved_head_importance_path_ad", type=str, default=None)
parser.add_argument("--save_file_path", type=str, default=None)
parser.add_argument("--shot", type=str, default="0-shot")
parser.add_argument("--scale", type=int, default=20)
parser.add_argument("--topk", type=int, default=30)

args = parser.parse_args()

with open(args.saved_head_importance_path_ac, 'rb') as handle_ac:
    results_ac = pickle.load(handle_ac)
with open(args.saved_head_importance_path_ad, 'rb') as handle_ad:
    results_ad = pickle.load(handle_ad) 

results_ac = results_ac.view(32, 32)
results_ac = (results_ac - results_ac.min()) / (results_ac.max() - results_ac.min())
results_ad = results_ad.view(32, 32)
results_ad = (results_ad - results_ad.min()) / (results_ad.max() - results_ad.min())

scale = args.scale
data=results_ac+(results_ac-results_ad)*scale

layers, heads = data.shape

top_k =args.topk
flattened_indices = torch.flatten(data).topk(top_k).indices
sorted_indices = [(idx // heads , idx % heads) for idx in flattened_indices]

top_k_dict = {}
for layer, head in sorted_indices:
    if str(layer.item()) not in top_k_dict:
        top_k_dict[str(layer.item())] = []
    top_k_dict[str(layer.item())].append(head.item())


output_file = args.save_file_path
with open(output_file, 'w') as json_file:
    json.dump(top_k_dict, json_file, indent=4)
    print(f'Top {top_k} heads saved in {output_file}')
