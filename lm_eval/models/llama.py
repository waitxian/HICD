import transformers
import torch
import os
import random
import pickle
from lm_eval.base import BaseLlamaLM
# from transformers.deepspeed import HfDeepSpeedConfig
# import deepspeed


class HFLM(BaseLlamaLM):
    def __init__(
        self,
        device="cuda",
        pretrained="facebook/llama-7b",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        model_cache_dir=None,
        tokenizer_cache_dir=None,
        mask_single_head=0,
        mask_heads=0,
        mask_fc=0,
        mask_iterative_fc=0,
        head_percent_mask=0,
        head_importance_path=None,
        fc_percent_mask=0,
        fc_importance_path=None,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        device_map = 'auto'
        self.llama = transformers.LlamaForCausalLM.from_pretrained(
            pretrained,
            device_map=device_map,
            cache_dir=model_cache_dir,
            torch_dtype=torch.float16
        )
        
        self.llama.model.embed_tokens.weight.requires_grad = False
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
            pretrained,
            cache_dir=tokenizer_cache_dir if tokenizer_cache_dir else 'tokenizer_cache/',
            use_fast=False
        ) if tokenizer is None else tokenizer

        assert isinstance(
            self.tokenizer,
            transformers.LlamaTokenizer,
        ), "Tokenizer must be an instance of LlamaTokenizer for compatibility!"

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size

        num_hidden_layers = self.llama.config.num_hidden_layers
        num_heads = self.llama.config.num_attention_heads
        self.head_mask = torch.ones(num_hidden_layers, num_heads, dtype=torch.half)
        self.fc_mask = torch.ones(num_hidden_layers, dtype=torch.half)

        if int(mask_heads):
            with open(head_importance_path, 'rb') as f:
                importance = pickle.load(f)
            _, head_indices = torch.sort(importance.view(-1))
            head_indices = list(head_indices.numpy())
            head_indices = head_indices[: int(head_percent_mask) * len(head_indices) // 100]
            self.head_mask.view(-1)[head_indices] = 0.
        elif int(mask_single_head):
            layer_idx = (int(mask_single_head) - 1) // num_heads
            head_idx = (int(mask_single_head) - 1) % num_heads
            self.head_mask[layer_idx, head_idx] = 0.

        if mask_fc:
            self.fc_mask[int(mask_fc)] = 0.
        elif int(mask_iterative_fc):
            with open(fc_importance_path, 'rb') as f:
                fc_indices = list(pickle.load(f))
            fc_indices = fc_indices[: int(fc_percent_mask) * len(fc_indices) // 100] 
            self.fc_mask[fc_indices] = 0.

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.llama.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def get_tokenizer(self):
        return self.tokenizer

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        if labels is None:
            with torch.no_grad():
                output = self.llama(input_ids=inps, attention_mask=attn_mask, head_mask=self.head_mask, fc_mask=self.fc_mask)
                return output.logits
        else:
            return self.llama(input_ids=inps, attention_mask=attn_mask, labels=labels).loss

    def _model_generate(self, context, max_length, eos_token_id):
        return self.llama.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


# for backwards compatibility
LlamaLM = HFLM