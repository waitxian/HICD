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
from lm_eval.base import MultipleChoiceTask
from lm_eval.utils import create_dataloader

_CITATION = """
@inproceedings{OpenBookQA2018,
    title={Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering},
    author={Todor Mihaylov and Peter Clark and Tushar Khot and Ashish Sabharwal},
    booktitle={EMNLP},
    year={2018}
}
"""


class OpenBookQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "openbookqa"
    DATASET_NAME = "additional"

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
        return  list(map(self._process_doc, self.dataset["test"]))
        
    def _process_doc(self, doc):
        out_doc = {
            "id": doc["id"],
            "query": doc["fact1"]+'\n'+doc["question_stem"]+'\nAnswer:',
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D"].index(doc["answerKey"].strip()),
        }
        return out_doc

    def get_dataloader(self, tokenizer, split = 'train', subset_size = None, batch_size = 1, num_fewshot = 0):
        split ='test'  
        docs = self.training_docs() if split == 'train' else self.validation_docs() if split == 'validation' else self.test_docs()
        
        adv=False
        # Adversarial modification of the gold label
        if adv==True:
            for idx in range(len(docs)):
                original_gold = docs[idx]['gold']
                new_gold = (original_gold + 1) % 4  # Rotate to the next answer
                docs[idx]['gold'] = new_gold
        
        return create_dataloader(tokenizer, docs, self.fewshot_context, self.doc_to_cont, subset_size = subset_size, batch_size = batch_size, num_fewshot = num_fewshot)
    
    def doc_to_cont(self, doc):
        return doc['choices'][doc['gold']]

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]