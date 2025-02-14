import re
import os
import json
import transformers
import torch
from tqdm import tqdm, trange
import argparse
import pandas as pd

import ssl
import urllib.request
import zipfile
from datasets import load_dataset
import json
from modeling import HICD
import ipdb


def load_hellaswag(data_path=None):
    if data_path:
        dataset = load_dataset(data_path, trust_remote_code=True)
    else:
        dataset = load_dataset("hellaswag", trust_remote_code=True)

    data = dataset["validation"]
    list_data_dict = []
    for idx, item in enumerate(data):
        label = int(item["label"]) if isinstance(item["label"], str) else item["label"]

        contradictions = [ending for i, ending in enumerate(item["endings"]) if i != item["label"]]
        
        formatted_item = {
            "prefix": item["ctx"],  
            "completion": item["endings"][label],
            "contradiction_0": contradictions[0],
            "contradiction_1": contradictions[1],
            "contradiction_2": contradictions[2],
        }
        
        list_data_dict.append(formatted_item)
    
    return list_data_dict

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
parser.add_argument("--num-gpus", type=str, default="1")
parser.add_argument("--max_gpu_memory", type=int, default=27)
parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
parser.add_argument("--data-path", type=str, default=None)
parser.add_argument("--output-path", type=str, default="./keys")
parser.add_argument("--output-token-path", type=str, default="./tokens")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--key-num", type=int, default=3)

args = parser.parse_args()
model_name = args.model_name
num_gpus = args.num_gpus
device = args.device

list_data_dict = load_hellaswag(args.data_path)

llm = HICD(model_name, device, num_gpus, args.max_gpu_memory)
llm.set_stop_words(["Q:", "\end{code}"])

key_words =[]
token_ranges_list =[]
cnt = 0
with torch.no_grad():
    for sample in tqdm(list_data_dict):
        
        decoded_text, token_ranges=llm.key_words(sample['prefix'], key_num=args.key_num)
        key_words.append(decoded_text)
        token_ranges_list.append(token_ranges)
        cnt += 1
        if cnt <= 10:
            print(cnt - 1)
            print("prefix:")
            print(sample['prefix'])
            print("key word:")
            print(key_words[-1])

output_file = args.output_path
output_file_token = args.output_token_path
with open(output_file, 'w') as f:
    json.dump(key_words, f)
with open(output_file_token, 'w') as f:
    json.dump(token_ranges_list, f)