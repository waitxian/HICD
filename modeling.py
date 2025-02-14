import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np

from scripts.run_hicd import hicd_config
import ipdb
import pickle
from torch.nn.functional import softmax

class HICD:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)
        

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text,head_config, token_ranges=None,
                 max_new_tokens=256, 
                 top_p=0.95, 
                 top_k=0,
                 temperature=0.8, 
                 task='halusum', 
                 remove_stop_words=False, 
                 alphas='2',
                 scale=0.01,
                 flag='None',
                 **kwargs):
        with torch.no_grad():
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens
            
            outputs = self.model.generate(input_ids,max_length=max_len, num_return_sequences=1,
                                output_scores=True, return_dict_in_generate=True,top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria,**kwargs)
            induce_outputs = hicd_config(self.model,self.tokenizer,input_text,head_config,token_ranges=token_ranges,scale=scale,max_length=max_len,task=task,flag=flag)
            
            induce_sequences, induce_scores = induce_outputs.sequences, induce_outputs.scores
            sequences, scores = outputs.sequences, outputs.scores
            
            
            output_str_list =[]
            gen_probs = [score for score in scores]  
            induce_probs = [score for score in induce_scores]
            
            alphas = list(map(float, alphas.split(',')))
            for alpha in alphas:
                print(alpha)
                amplified_probs = [
                    softmax(gen_prob + alpha * (gen_prob - induce_prob),dim=-1)
                    for gen_prob, induce_prob in zip(gen_probs, induce_probs)
                ]

                amplified_tokens = [
                    torch.argmax(amplified_prob, dim=-1) for amplified_prob in amplified_probs
                ]

                amplified_sequences = torch.cat(amplified_tokens).cpu().numpy() if isinstance(amplified_tokens[0], torch.Tensor) else amplified_tokens
                amplified_output = self.tokenizer.decode(amplified_sequences, skip_special_tokens=True)
                output_str =amplified_output

                if remove_stop_words:
                    for stop_word in self.stop_words:
                        length_to_remove = len(stop_word)
                        if output_str[-length_to_remove:] == stop_word:
                            output_str = output_str[:-length_to_remove]
                    output_str = output_str.strip()
                output_str_list.append(output_str)

        if self.device:
            torch.cuda.empty_cache()

        return output_str_list
    
    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def lm_score(self, input_text1, input_text2,head_config,task='hellaswag',flag='None',
                input_text1_keys=None,
                pmi=False, 
                max_new_tokens=256, 
                top_p=0.95, 
                top_k=0, 
                temperature=0.8, 
                verbose=True, 
                remove_stop_words=False, 
                relative_top=0.1, 
                pondering =None,
                relative_top_value=-1000.0, 
                post_softmax=True, 
                alpha=1,
                scale=0.01,
                token_ranges=None,
                **kwargs,
                ):
        with torch.no_grad():
            
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            completion_len =len(continue_ids)
                
            if pondering is None:
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  
                logits = outputs[prefix_ids.shape[-1] - 1: -1, :]
                
                induce_outputs = hicd_config(self.model,self.tokenizer,input_text,head_config,token_ranges=token_ranges,scale=scale,task=task,flag=flag)
                induce_outputs = induce_outputs[0].squeeze(0)
                
                induce_outputs = induce_outputs.log_softmax(-1) 
                induce_logits = induce_outputs[prefix_ids.shape[-1] - 1: -1, :]

                final_logits = logits-induce_logits
                final_logits = logits + alpha * final_logits 
                
                if alpha == 0:
                    final_logits = logits
                else:
                    final_logits = final_logits.log_softmax(dim=-1)
            
                log_probs = final_logits[range(logits.shape[0]), continue_ids].sum().item()
            else:
                input_text_keys = input_text1_keys + input_text2
                input_keys_ids = self.tokenizer(input_text_keys, return_tensors="pt").input_ids.to(self.device)
                prefix_keys_ids = self.tokenizer(input_text1_keys, return_tensors="pt").input_ids.to(self.device)

                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  

                logits = outputs[prefix_ids.shape[-1] - 1: -1, :]

                ponder_outputs = self.model(input_keys_ids)[0].squeeze(0)
                ponder_outputs = ponder_outputs.log_softmax(-1)  
                ponder_logits = ponder_outputs[prefix_keys_ids.shape[-1] - 1: -1, :]

                final_logits = logits-ponder_logits
                
                final_logits = logits + alpha * final_logits
                
                if alpha == 0:
                    final_logits = logits
                else:
                    final_logits = final_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(ponder_logits, relative_top)
                    final_logits = torch.where(relative_top_mask, relative_top_value, final_logits)
                
                log_probs = final_logits[range(logits.shape[0]), continue_ids].sum().item()        
            
        return log_probs, completion_len

    def key_words(self, input_text, key_num=1):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model(input_ids)[0].squeeze(0)[:-1, :]
        outputs = outputs.log_softmax(-1)
        shift_text = input_ids[0, 1:]
        

        text_probs = outputs[range(outputs.shape[0]), shift_text]
       
        _, indices = torch.topk(text_probs, max(int(outputs.shape[0] * 0.015 * key_num), 1), largest=False, sorted=True)

        indices = indices[:int(outputs.shape[0] * 0.01 * key_num)]
        
        indices, _ = torch.sort(indices, descending=False)
        #indices, _ = torch.sort(indices, descending=True)

        sample_ids = shift_text[indices]
        decoded_text = self.tokenizer.decode(sample_ids)

        offset_mapping = self.tokenizer(input_text, return_offsets_mapping=True)["offset_mapping"]
        token_ranges = [(offset_mapping[i+1][0], offset_mapping[i+1][1]) for i in indices]
        
        for i, (token, (start, end)) in enumerate(zip(decoded_text.split(), token_ranges)):
            extracted_text = input_text[start:end]
            print(f"Token {i}: '{token}' - Range: ({start}, {end}), Extracted: '{extracted_text}'")

        return decoded_text, token_ranges

    

