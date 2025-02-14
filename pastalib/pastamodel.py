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
from pastalib.pasta import PASTA
import ipdb
from pastalib.utils.typing import (
    Model,
    Dataset,
    Device,
    ModelInput,
    ModelOutput,
    StrSequence,
    Tokenizer,
    TokenizerOffsetMapping,
)
from contextlib import contextmanager
from functools import partial

class PASTA_induce(PASTA):
    def __init__(
        self, 
        model: Model, 
        tokenizer: Tokenizer, 
        head_config: dict|list|None = None, 
        alpha: float = 0.01, 
        scale_position: str = "exclude", 
        head_config_inner: dict|list|None = None, 
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.setup_model(model)

        self.alpha = alpha
        self.scale_position = scale_position
        #self.setup_head_config(head_config)
        self.setup_head_config_inner(head_config_inner)

        assert self.scale_position in ['include', 'exclude', 'generation']
        assert self.alpha > 0
    
    def setup_head_config_inner(self, head_config):
        """Configures the attention heads to be steered."""

        if isinstance(head_config, dict):
            self.head_config_inner = {int(k): v for k, v in head_config.items()}
            self.all_layers_idx_inner = [int(key) for key in head_config]
        elif isinstance(head_config, list):
            self.all_layers_idx_inner = [int(v) for v in head_config]
            self.head_config_inner = {
                idx: list(range(self.num_attn_head)) for idx in self.all_layers_idx_inner
            }
        else:
            self.head_config_inner = None
            self.all_layers_idx_inner = None

    def edit_multisection_attention(
        self, 
        module: torch.nn.Module, 
        input_args: tuple,
        input_kwargs: dict, 
        head_idx: list[int] = None,
        head_idx_inner: list[int] = None,
        token_ranges: list[torch.Tensor] = None, 
        input_len: int = None, 
        layer_idx: int = None
    ):
        # Ensure at least one head_idx is provided
        if head_idx is None and head_idx_inner is None:
            raise ValueError(f"No valid head configuration found for layer {layer_idx}. Provide at least one of `head_idx` or `head_idx_inner`.")
        
        # Extract or clone the attention mask
        if "attention_mask" in input_kwargs:
            attention_mask = input_kwargs['attention_mask'].clone()
        elif input_args is not None:
            arg_idx = self.ATTENTION_MASK_ARGIDX[self.model_name]
            attention_mask = input_args[arg_idx].clone()
        else:
            raise ValueError(f"Attention mask not found in {str(module)}")
        
        #attention_mask = attention_mask[..., :-1]
        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        dtype, device = attention_mask.dtype, attention_mask.device

        if head_dim != self.num_attn_head:
            attention_mask = attention_mask.expand(
                bsz, self.num_attn_head, tgt_len, src_len
            ).clone()
        
        # Scale constant for token scaling
        scale_constant = torch.Tensor([self.alpha]).to(dtype).to(device).log()
        
        
        # Pruning logic if `head_idx` is provided
        if head_idx is not None:
            pruning_value = torch.tensor(float("-65504")).to(dtype).to(device)
            attention_mask[:, head_idx, :, :] = pruning_value
        
        if head_idx_inner is not None and token_ranges is not None:
            for token_range in token_ranges:
                ti, tj = token_range[0], token_range[1]
                if self.scale_position == "include":
                    attention_mask[0, head_idx_inner, :, ti:tj] -= scale_constant
                elif self.scale_position == "exclude":
                    attention_mask[0, head_idx_inner, :, :ti] += scale_constant
                    attention_mask[0, head_idx_inner, :, tj:input_len] += scale_constant
                elif self.scale_position == "generation":
                    attention_mask[0, head_idx_inner, :, :input_len] += scale_constant
                else:
                    raise ValueError(f"Unexpected scale_position: {self.scale_position}")

        if self.model_name in ["llama", "mistral", "phi3mini"]:
            attention_mask.old_size = attention_mask.size 
            attention_mask.size = lambda: (bsz, 1, tgt_len, src_len)

        if "attention_mask" in input_kwargs:
            input_kwargs['attention_mask'] = attention_mask 
            return input_args, input_kwargs
        else:
            return (input_args[:arg_idx], attention_mask, input_args[arg_idx+1:]), input_kwargs

    @contextmanager
    def apply_steering(
        self, 
        model: Model, 
        strings: list, 
        token_ranges: list, 
        model_input: ModelInput,  
        occurrence: int = 0,
    ):
        if isinstance(token_ranges[0], str):
            token_ranges = [token_ranges]
        registered_hooks = []

        # Combine all layers from `self.all_layers_idx` and `self.all_layers_idx_inner` if they exist
        all_layers_idx = self.all_layers_idx_inner 
        
        for layer_idx in all_layers_idx:
            name = self.ATTN_MODULE_NAME[self.model_name].format(layer_idx)
            module = model.get_submodule(name)
            
            # Prepare hook function with appropriate head indices
            #head_idx = self.head_config.get(layer_idx) if self.head_config else None
            head_idx = None
            head_idx_inner = self.head_config_inner.get(layer_idx) if self.head_config_inner else None

            # Ensure at least one configuration is valid
            if head_idx is None and head_idx_inner is None:
                raise ValueError(f"No valid head configuration found for layer {layer_idx}.")
            
            hook_func = partial(
                self.edit_multisection_attention, 
                head_idx=head_idx,
                head_idx_inner=head_idx_inner,
                token_ranges=token_ranges, 
                input_len=model_input['input_ids'].size(-1),
                layer_idx=layer_idx
            )

            registered_hook = module.register_forward_pre_hook(hook_func, with_kwargs=True)
            registered_hooks.append(registered_hook)
        
        try:
            yield model
        except Exception as error:
            raise error
        finally:
            for registered_hook in registered_hooks:
                registered_hook.remove()
