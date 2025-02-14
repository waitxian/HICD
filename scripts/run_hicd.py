import argparse
import json
import logging
import random
from functools import partial
from pathlib import Path
from typing import cast

from evaluation import  models
from pastalib import pastamodel
from typing import Any, Literal, Optional, Sequence, cast, overload,Union, Tuple

import torch
import torch.utils.data
from tqdm.auto import tqdm
import json
from torch.utils.tensorboard import SummaryWriter
from evaluation.utils.typing import (
    Dataset,
    Device,
    ModelInput,
    ModelOutput,
    StrSequence,
    Tokenizer,
    TokenizerOffsetMapping,
)

class ModelAndTokenizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

def induce_heads(head_config, num_heads_per_layer, head_mask):
    global_head_indices = []
    for layer_idx, head_list in head_config.items():
        layer_number = int(layer_idx.replace("layer", ""))
        for head in head_list:
            global_index = layer_number * num_heads_per_layer + head
            global_head_indices.append(global_index)
    
    induce_indices = global_head_indices
    return induce_indices

def read_head_config(head_config):
    if '.json' in head_config:
        with open(head_config, 'r') as handle:
            head_config = json.load(handle) 
    elif '{' in head_config and '}' in head_config:
        head_config = json.loads(head_config)
    else:
        # head_config = [int(h) for h in pasta_head_config.strip().split(',')]
        raise ValueError("Incorrect format of head config.")
    return head_config

def inputs_from_batch(
    mt: models.ModelAndTokenizer,
    text: str | StrSequence,
    device: Optional[Device] = None,
) -> tuple[ModelInput, Sequence[TokenizerOffsetMapping]]:
    """Precompute model inputs."""
    
    inputs = mt.tokenizer(
        text,
        return_tensors="pt",
    )
    if device is not None:
        inputs = inputs.to(device)
    return inputs
    
def hicd_config(model,tokenizer,prompts,head_config,token_ranges=None,scale=0.01,scale_position="include",max_length=2048,task='hellaswag',flag='None'):
    mt = ModelAndTokenizer(model=model, tokenizer=tokenizer)

    hicd_head_config=head_config
    head_config = read_head_config(hicd_head_config[0])
   
    if hicd_head_config[1]==None:
        head_config_inner =None
    else:
        head_config_inner =read_head_config(hicd_head_config[1])
    
    mt.tokenizer.pad_token =mt.tokenizer.eos_token
    with models.set_padding_side(mt.tokenizer, padding_side="left"):
                inputs =inputs_from_batch(mt, prompts, device='cuda')
    
    attention_mask = inputs["attention_mask"]
    input_ids = inputs["input_ids"]
    
    num_layers = model.config.num_hidden_layers  
    num_heads = model.config.num_attention_heads  
    head_mask = torch.ones((num_layers, num_heads), device=input_ids.device)
    indices = induce_heads(head_config, num_heads, head_mask)
    head_mask.view(-1)[indices] = 0.
    head_mask = head_mask.view(num_layers, num_heads).contiguous()

    if task=='halusum':
        max_new_tokens=max_length-inputs["input_ids"].shape[-1]
        generate_kwargs = dict(
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_scores=True,
                            max_length=max_length,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id,
                        )
    prompts=[prompts]
   
    if head_config_inner ==None and task!='halusum':
        outputs = model(input_ids=input_ids, head_mask=head_mask,flag=flag)
    elif head_config_inner ==None and task =='halusum':
        outputs = model.generate(input_ids=input_ids, head_mask=head_mask,flag=flag,**generate_kwargs)
    else:
        pasta_steerer = pastamodel.PASTA_induce(
                mt.model, 
                mt.tokenizer,
                head_config_inner=head_config_inner, 
                alpha=scale, 
                scale_position=scale_position,
            )

        if pasta_steerer is not None and token_ranges!=None:
            with pasta_steerer.apply_steering(
                        model=model, 
                        strings=prompts, 
                        token_ranges=token_ranges, 
                        model_input=inputs, 
                    ) as steered_model: 
                        if task =='hellaswag' or task == 'race' or task == 'truthfulqa' or task=='openbookqa' or task=='factor':
                            if head_config!=None:
                                outputs = steered_model(input_ids=inputs["input_ids"],head_mask=head_mask,flag=flag)
                            else:
                                outputs = steered_model(input_ids=inputs["input_ids"])
                        elif task =='halusum':
                            if head_config!=None:
                                outputs = steered_model.generate(**inputs,head_mask=head_mask,flag=flag,**generate_kwargs)
                            else:
                                outputs = steered_model.generate(**inputs,**generate_kwargs)
                        else:
                            print("no task!")
        else:
            outputs =model(input_ids=inputs["input_ids"])
    return outputs



