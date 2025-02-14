# Ref: https://github.com/LUMIA-Group/SH2
import re
import os
import json
import random
import torch
import numpy as np
import transformers
from tqdm import tqdm, trange
import argparse
from collections import defaultdict, Counter
import glob
import sys

import ssl
import urllib.request
import zipfile
import tiktoken
from modeling import HICD
transformers.logging.set_verbosity(40)

DEBUG = False

def num_tokens_from_message(message, model="davinci"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(message))
    return num_tokens


def truncate_message(prompt1, prompt2, model="davinci"):
    if num_tokens_from_message(prompt1 + prompt2, model) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split()[:-1])
    prompt = prompt1 + prompt2
    return prompt


demo_keys = []
def load_jsonl(file_path):
    global demo_keys
    list_data_dict = {}
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        data = data[:1000]

        candicates = ["hallucinated_summary", "right_summary"]
        ground_truths = ["Yes", "No"]
        for j in range(len(candicates)):
            list_data_dict[j] = []
            for idx in range(len(data)):
                response = "\n#Summary#: " + data[idx][candicates[j]] + "\n#Your Judgement#:"
                ground_truth = ground_truths[j]

                new_item = dict(
                    context="#Document#: " + data[idx]["document"],
                    response=response,
                    answer=ground_truth
                )
                list_data_dict[j].append(new_item)

    return list_data_dict


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


def create_demo_text():
    prompt, context, response, answer = [], [], [], []

    prompt.append("You are trying to determine if the summary is factual but some information cannot be directly inferred or entailed from the document.")
    context.append("#Document#: The panther chameleon was found on Monday by a dog walker in the wooded area at Marl Park. It had to be put down after X-rays showed all of its legs were broken and it had a deformed spine. RSPCA Cymru said it was an \"extremely sad example of an abandoned and neglected exotic pet\". Inspector Selina Chan said: \"It is a possibility that the owners took on this animal but were unable to provide the care he needs and decided to release him to the wild. \"We are urging potential owners of exotic animals to thoroughly research what is required in the care of the particular species before taking one on. \"Potential owners need to make sure they can give their animal the environment it needs and they have the facilities, time, financial means and long-term commitment to maintain a good standard of care, as required under the Animal Welfare Act 2006.\" She added it was illegal to release non-native species into the wild.")
    response.append("#Summary#: A chameleon that was found in a Cardiff park has been put down after being abandoned and neglected by its owners.")
    answer.append("#Your Judgement#: Yes")

    prompt.append("You are trying to determine if there exists some non-factual and incorrect information in the summary.  ")
    context.append("#Document#: The city was brought to a standstill on 15 December last year when a gunman held 18 hostages for 17 hours. Family members of victims Tori Johnson and Katrina Dawson were in attendance. Images of the floral tributes that filled the city centre in the wake of the siege were projected on to the cafe and surrounding buildings in an emotional twilight ceremony. Prime Minister Malcolm Turnbull gave an address saying a \"whole nation resolved to answer hatred with love\". \"Testament to the spirit of Australians is that with such unnecessary, thoughtless tragedy, an amazing birth of mateship, unity and love occurs. Proud to be Australian,\" he said. How the Sydney siege unfolded New South Wales Premier Mike Baird has also announced plans for a permanent memorial to be built into the pavement in Martin Place. Clear cubes containing flowers will be embedded into the concrete and will shine with specialised lighting. It is a project inspired by the massive floral tributes that were left in the days after the siege. \"Something remarkable happened here. As a city we were drawn to Martin Place. We came in shock and in sorrow but every step we took was with purpose,\" he said on Tuesday.")
    response.append("#Summary#: Crowds have gathered in Sydney's Martin Place to honour the victims of the Lindt cafe siege, one year on.")
    answer.append("#Your Judgement#: No")

    prompt.append("You are trying to determine if there is a factual contradiction between the summary and the document.")
    context.append("#Document#: Christopher Huxtable, 34, from Swansea, had been missing since the collapse in February. His body was found on Wednesday and workers who carried out the search formed a guard of honour as it was driven from the site in the early hours of the morning. Ken Cresswell, 57, and John Shaw, 61, both from Rotherham, remain missing. The body of a fourth man, Michael Collings, 53, from Brotton, Teesside, was previously recovered from the site. Swansea East MP Carolyn Harris, who has been involved with the family since the incident, said they still did not know all the facts about the collapse. She said: \"I feel very sad. My heart and my prayers go out to the family who have waited desperately for Christopher's body to be found. They can finally have closure, and say goodbye to him and grieve his loss. \"But let's not forget that there's two other families who are still waiting for their loved ones to be returned.\" The building was due for demolition when it partially collapsed in February.")
    response.append("#Summary#: The body of a man whose body was found at the site of the Swansea Bay Power Station collapse has been removed from the site.")
    answer.append("#Your Judgement#: Yes")

    # Concatenate demonstration examples ...
    demo_text = "I want you act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.\n\n"
    for i in range(len(context)):
        demo_text += prompt[i] + "\n" + context[i] + "\n" + \
                        response[i] + "\n" + answer[i] + "\n\n"

    return demo_text


def build_prompt(context, response):
    demo = create_demo_text()
    prompt = demo + context
    input_text_prompt = truncate_message(prompt, response)
    return input_text_prompt


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./strqa")
    parser.add_argument("--output-path", type=str, default="./halueval_sum")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--keys-path", type=str, default=None)
    parser.add_argument("--pause-num", type=int, default=3)
    parser.add_argument("--alpha", type=str, default='2')
    parser.add_argument("--task", type=str, default='halusum')
    parser.add_argument("--flag", type=str, default=None)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--token_ranges", type=str, default=None)
    parser.add_argument("--file_path", type=str, default=None)
    parser.add_argument("--file_path_inner", type=str, default=None)

    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    keys_path = args.keys_path
    if args.token_ranges:
        token_ranges_all =load_token_ranges(filepath=args.token_ranges)
    
    file_path =args.file_path
    file_path_inner =args.file_path_inner
    # set seed
    set_seed(args.seed)

    fp = args.data_path
    if not os.path.exists(fp):
        raise ValueError(f"Test file {fp} does not exist.")

    file_paths = []
    if file_path is None:
        file_path=None
    file_paths.append(file_path)
    
    if file_path_inner is None:
        file_path_inner=None
    file_paths.append(file_path_inner)
    

    head_config =file_paths
    
    list_data_dict = load_jsonl(fp)

    llm = HICD(model_name, device, num_gpus, args.max_gpu_memory)
    stop_word_list = ["#Document#:"]
    llm.set_stop_words(stop_word_list)
    
    
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, 
                            do_sample=args.do_sample, 
                            top_p=args.top_p, 
                            top_k=args.top_k, 
                            temperature=args.temperature,  
                            task=args.task, 
                            alphas=args.alpha,
                            scale=args.scale,
                            flag=args.flag)
    
    output_path = args.output_path
    candicates = ["hallucinated_summary", "right_summary"]
    alpha_list = list(map(float, args.alpha.split(',')))
    corrects  = [[] for _ in range(len(alpha_list))]
    incorrects =[[] for _ in range(len(alpha_list))]
    
    
   
    for j in range(len(candicates)):
        print("="*20 + candicates[j] + "="*20)
        correct = [0] * len(alpha_list)
        incorrect = [0] * len(alpha_list)
        for idx in tqdm(range(len(list_data_dict[j]))):
            sample = list_data_dict[j][idx]

            toekn_ranges=None
            if args.token_ranges:
                token_ranges =token_ranges_all[idx]

            input_text = build_prompt(sample['context'], sample['response'])
            
            model_completion_all = llm.generate(input_text,head_config, token_ranges=token_ranges, **generate_kwargs)
                

            for i in range(len(model_completion_all)):
                for stop_word in stop_word_list:
                    length_to_remove = len(stop_word)
                    if model_completion_all[i][-length_to_remove:] == stop_word:
                        model_completion_all[i]= model_completion_all[i][:-length_to_remove]
                model_completion_all[i] = model_completion_all[i].strip()
                ans = model_completion_all[i].replace(".", "")

                
                if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                    gen = {"document": sample['context'], "summary": sample['response'], "ground_truth": sample['answer'], "judgement": "failed!"}
                    dump_jsonl(gen, output_path, append=True)
                    incorrect[i] += 1
                    print('sample {} fails......'.format(idx))
                    continue
                elif "Yes" in ans:
                    if ans != "Yes":
                        ans = "Yes"
                    gen = {"document": sample['context'], "summary": sample['response'], "ground_truth": sample['answer'], "judgement": ans}
                elif "No" in ans:
                    if ans != "No":
                        ans = "No"
                    gen = {"document": sample['context'], "summary": sample['response'], "ground_truth": sample['answer'], "judgement": ans}
                else:
                    gen = None
                assert (gen is not None)

                if sample['answer'] == ans:
                    correct[i] += 1
                else:
                    incorrect[i] += 1

                print('sample {} success {}......'.format(idx,i))
                dump_jsonl(gen, output_path, append=True)
        
        
        for i in range(len(alpha_list)):
            print(alpha_list[i])
            print('{}: {} correct samples, {} incorrect samples, Accuracy: {}'.format(candicates[j], correct[i], incorrect[i], correct[i] / len(list_data_dict[j])))
            corrects[i].append(correct[i])
            incorrects[i].append(incorrect[i])
    
    
    print("=" * 50)
    with open(output_path, 'w') as f:
        for i in range(len(alpha_list)):
            f.write("Alpha {}: {}\n".format(i, alpha_list[i]))

            correct, incorrect, total = 0, 0, 0
            for j in range(len(candicates)):
                result_str = '{}: {} correct samples, {} incorrect samples, Accuracy: {}\n'.format(
                    candicates[j], corrects[i][j], incorrects[i][j], corrects[i][j] / len(list_data_dict[j])
                )
                print(result_str) 
                f.write(result_str) 
                correct += corrects[i][j]
                incorrect += incorrects[i][j]
                total += len(list_data_dict[j])
            
            total_result_str = 'Total: {} correct samples, {} incorrect samples, acc_H {}, acc_A: {}\n'.format(
                correct, incorrect, 2 * corrects[i][0] * corrects[i][1] / (correct * len(list_data_dict[0])), correct / total
            )
            print(total_result_str)  
            f.write(total_result_str) 

            precision = corrects[i][0] / (corrects[i][0] + incorrects[i][1])
            recall = corrects[i][0] / len(list_data_dict[0])
            F1 = (2 * precision * recall) / (precision + recall)
            
            f1_result_str = 'Precision: {}, recall: {}, F1: {}\n'.format(precision, recall, F1)
            print(f1_result_str) 
            f.write(f1_result_str) 

