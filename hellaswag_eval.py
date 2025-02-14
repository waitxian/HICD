# Ref: https://github.com/LUMIA-Group/SH2

import re
import os
import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse

import ssl
import urllib.request

from modeling import HICD
import ipdb
import json
import copy
from datasets import load_dataset
from itertools import combinations

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = True
ANSWER_TRIGGER = "The answer is"

def load_hellaswag(data_path=None, pondering=None, keys_path=None):
    if keys_path is not None:
        with open(keys_path, "r", encoding="utf-8") as f:
            key_words = json.load(f)
    
    if data_path:
        dataset = load_dataset(data_path, trust_remote_code=True)
    else:
        dataset = load_dataset("Rowan/hellaswag", trust_remote_code=True)
    
    data = dataset["validation"]
    cnt = 0
    list_data_dict = []
    completion_lens = []
    for idx, item in enumerate(data):
        
        # Convert label to integer if it is a string
        label = int(item["label"]) if isinstance(item["label"], str) else item["label"]

        # Exclude the correct label from contradictions
        contradictions = [ending for i, ending in enumerate(item["endings"]) if i != item["label"]]
        
        formatted_item = {
            "prefix": item['activity_label']+':'+item["ctx"],  # Combine ctx_a and ctx_b if needed
            "completion": item["endings"][label],
            "contradiction_0": contradictions[0],
            "contradiction_1": contradictions[1],
            "contradiction_2": contradictions[2],
        }
        
        correct_length = len(item["endings"][label])
        incorrect_lengths = [len(ending) for i, ending in enumerate(item["endings"]) if i != label]

        completion_lengths = [correct_length] + incorrect_lengths
        completion_lens.append(completion_lengths)
        
        if pondering == "hard" and keys_path is not None:
            formatted_item["prefix"] = "Pondering: " + key_words[idx % len(key_words)] + "\n\nContext:" + formatted_item["prefix"]
        
        list_data_dict.append(formatted_item)
    
    return list_data_dict


def download_url(url: str, folder='folder'):
    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_token_ranges(filepath=None):
    try:
        with open(filepath, "r") as file:
            token_ranges_all = json.load(file)
            return token_ranges_all
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except json.JSONDecodeError:
        print("Error: JSON file could not be decoded.")
    return None

def load_top_k_heads(json_path):
    with open(json_path, 'r') as f:
        head_config_list = json.load(f)
    return head_config_list

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="./results")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--keys-path", type=str, default=None)
    parser.add_argument("--pondering", type=str, default=None)
    parser.add_argument("--pause-num", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.1)
    parser.add_argument("--token_ranges", type=str, default=None)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--task", type=str, default='hellaswag')
    parser.add_argument("--flag", type=str, default=None)
    parser.add_argument("--file_path", type=str, default=None)
    parser.add_argument("--file_path_inner", type=str, default=None)
    
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    file_path =args.file_path
    file_path_inner =args.file_path_inner
    
    list_data_dict = load_hellaswag(args.data_path)

    if args.token_ranges:
        token_ranges_all =load_token_ranges(filepath=args.token_ranges)
    output_file = args.output_path
    
    file_paths = []
    if file_path is None:
        file_path=None
    file_paths.append(file_path)
    
    if file_path_inner is None:
        file_path_inner=None
    file_paths.append(file_path_inner)

    head_config =file_paths
   
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]

    if args.debug:
        list_data_dict = list_data_dict[:10]
    
    if args.pondering is not None:
        list_data_dict_keys = load_hellaswag(args.data_path,pondering=args.pondering, keys_path=args.keys_path)

        if args.parallel:
            chunk_size = len(list_data_dict_keys) // args.total_shard
            list_data_dict_keys = list_data_dict_keys[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
   
    llm = HICD(model_name, device, num_gpus, args.max_gpu_memory)
    llm.set_stop_words(["Q:", "\end{code}"])
    answers = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            temperature=args.temperature,
                            repetition_penalty=args.repetition_penalty,
                            relative_top=args.relative_top,
                            relative_top_value=args.relative_top_value,
                            pondering=args.pondering,
                            alpha=args.alpha,
                            scale=args.scale,
                            task=args.task,
                            flag=args.flag
                            )
    
    for idx in tqdm(range(len(list_data_dict))):
        sample = list_data_dict[idx]
        if args.token_ranges:
            token_ranges =token_ranges_all[idx]
        else:
            token_ranges=None
        if args.pondering is None:
            input_text_keys = None
        else:
            sample_keys = list_data_dict_keys[idx]
            input_text_keys = sample_keys['prefix']
        
        context = sample['prefix']
        answer_true = ' ' + sample['completion']
        answers_false = []
        for i in range(3):
            answers_false.append(' ' + sample[f'contradiction_{i}'])
        
        completion_len = []
        answer_true_log_prob ,correct_len= llm.lm_score(context, answer_true,head_config,token_ranges=token_ranges,input_text1_keys=input_text_keys,**generate_kwargs)
        completion_len.append(correct_len)
        
        answer_false_log_probs = []
        for answer_false in answers_false:
            answer_false_log_prob ,incorrect_len= llm.lm_score(context, answer_false,head_config,token_ranges=token_ranges,input_text1_keys=input_text_keys,**generate_kwargs)
            completion_len.append(incorrect_len)
            answer_false_log_probs.append(answer_false_log_prob)
        if args.debug:
            print(f'log prob of answers: {answer_true_log_prob}', end=' ')
            for answer_false_log_prob in answer_false_log_probs:
                print(f'{answer_false_log_prob}', end=' ')
            print()
        is_cor = True

        log_probs = [answer_true_log_prob] + answer_false_log_probs 
        normalized_log_probs = log_probs / np.array(completion_len)
        predicted_answer_idx = np.argmax(normalized_log_probs)
        if predicted_answer_idx == 0: 
            is_cor = True
        else:
            is_cor = False
        
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_completion'].append([answer_true_log_prob] + answer_false_log_probs)

        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.')

    with open(output_file, "a") as file:
        result_info = (f"Num of total questions: {len(answers)}, "
                    f"correct num: {sum(answers)}, "
                    f"correct rate: {float(sum(answers)) / len(answers):.5f}.")

        file.write(f"--scale: {args.scale}, alpha: {args.alpha}, head_config:{head_config},{result_info}\n")
