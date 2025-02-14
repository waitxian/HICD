"""
Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering
https://arxiv.org/pdf/1809.02789.pdf

OpenBookQA is a question-answering dataset modeled after open book exams for
assessing human understanding of a subject. It consists of 5,957 multiple-choice
elementary-level science questions (4,957 train, 500 dev, 500 test), which probe
the understanding of a small “book” of 1,326 core science facts and the application
of these facts to novel situations. For training, the dataset includes a mapping
from each question to the core science fact it was designed to probe. Answering
OpenBookQA questions requires additional broad common knowledge, not contained
in the book. The questions, by design, are answered incorrectly by both a retrieval-
based algorithm and a word co-occurrence algorithm.

Homepage: https://allenai.org/data/open-book-qa
"""
from lm_eval.base import Task
from lm_eval.utils import create_dataloader
import re
import os
import json
import random
from tqdm import tqdm
import pandas as pd
_CITATION = """
@inproceedings{OpenBookQA2018,
    title={Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering},
    author={Todor Mihaylov and Peter Clark and Tushar Khot and Ashish Sabharwal},
    booktitle={EMNLP},
    year={2018}
}
"""
def load_csv(file_path, pondering=None, keys_path=None):
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
        if pondering == 'pause':
            item['prefix'] = "\nPondering: " + "." * args.pause_num + "\n\nContext:" + item['prefix']
        elif pondering == 'repeat':
            item['prefix'] = "\nPondering: " + item['prefix'] + "\n\nContext:" + item['prefix']
        elif pondering == 'hard':
            if keys_path is not None:
                item['prefix'] = "Pondering: " + key_words[idx] + "\n\nContext:" + item['prefix']
        list_data_dict.append(item)
    return list_data_dict

def load_jsonl(file_path, pondering=None, keys_path=None):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        list_data_dict = []
        for idx in range(len(data)):
            new_item = dict(
                document=data[idx]["document"],
                right_summary = data[idx]['right_summary'],
                hallucinated_summary = data[idx]['hallucinated_summary'],
            )
    
            list_data_dict.append(new_item)
    
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
    

    def get_dataloader(self, tokenizer, split = 'train', subset_size = None, batch_size = 1, num_fewshot = 0):
        converted_data = []
        choice = 'news'
        fp =f'/root/dataset/{choice}_factor.csv'
        list_data_dict = load_csv(fp)
        for idx in tqdm(range(len(list_data_dict))):
            sample = list_data_dict[idx]
            
            prefix = sample["prefix"]
            completion = sample["completion"]
            contradictions = [sample[key] for key in sample if key.startswith("contradiction")]
            
            new_sample = {
                "id": str(idx),
                "query": prefix,
                "choices": [completion] + contradictions,
                "gold": 0  
            }
            
            converted_data.append(new_sample)
        
        docs =converted_data
        
        return create_dataloader(tokenizer, docs, self.fewshot_context, self.doc_to_cont, subset_size = subset_size, batch_size = batch_size, num_fewshot = num_fewshot)
    
    def doc_to_cont(self, doc):
        return doc['choices'][doc['gold']]

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]