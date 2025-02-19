from lm_eval.base import Task
from lm_eval.utils import create_dataloader
import re
import os
import json
import random
from tqdm import tqdm
import pandas as pd

def load_csv(file_path):
    list_data_dict = []
    df = pd.read_csv(file_path)
    prefix_type = 'turncated_prefixes'
    for idx in range(len(df)):
        item = dict(
            prefix=df[prefix_type][idx],
            completion=df['completion'][idx],
            contradiction_0=df['contradiction_0'][idx],
            contradiction_1=df['contradiction_1'][idx],
            contradiction_2=df['contradiction_2'][idx],
        )
        list_data_dict.append(item)
    return list_data_dict

class Factor(Task):
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        out_doc = {
            "id": doc["id"],
            "query": doc["question_stem"],
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D"].index(doc["answerKey"].strip()),
        }
        return out_doc
    
    def get_dataloader(self, tokenizer, split='train', subset_size=None, batch_size=1, num_fewshot=0):
        """
        Create a dataloader for the specified dataset split.

        Args:
            tokenizer (Tokenizer): The tokenizer used for text processing.
            split (str): The data split to use ('train', 'validation', 'test').
            subset_size (int): The size of the subset to use (optional).
            batch_size (int): The batch size for dataloader.
            num_fewshot (int): Number of few-shot examples to include.

        Returns:
            Dataloader: A dataloader for the processed dataset.
        """
        converted_data = []
        choice = 'news'  # Dataset choice (can be adjusted as needed)
        file_path = f'/path/to/{choice}_factor.csv'  # Update path accordingly
        data_dict_list = load_csv(file_path)

        adv = False  # Toggle for adversarial setup (optional)
        
        for idx in tqdm(range(len(data_dict_list))):
            sample = data_dict_list[idx]

            prefix = sample["prefix"]
            completion = sample["completion"]
            contradictions = [sample[key] for key in sample if key.startswith("contradiction")]
            
            if adv:
                new_sample = {
                    "id": str(idx),
                    "query": prefix,
                    "choices": [completion] + contradictions,
                    "gold": 1  # Adversarial label (optional)
                }
            else:
                new_sample = {
                    "id": str(idx),
                    "query": prefix,
                    "choices": [completion] + contradictions,
                    "gold": 0  # Non-adversarial label
                }

            converted_data.append(new_sample)

        # Prepare the documents for dataloader
        docs = converted_data

        return create_dataloader(
            tokenizer, docs, 
            fewshot_context=self.fewshot_context, 
            doc_to_cont=self.doc_to_cont,
            subset_size=subset_size, 
            batch_size=batch_size, 
            num_fewshot=num_fewshot
        )
    
    
    def doc_to_cont(self, doc):
        return doc['choices'][doc['gold']]

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]