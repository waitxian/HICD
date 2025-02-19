
import numpy as np
from lm_eval.base import Task
from lm_eval.utils import create_dataloader
import json
from tqdm import tqdm
from pathlib import Path

import tiktoken

def num_tokens_from_message(message, tokenizer, model_max_length=2048):
    """
    Calculate the number of tokens in a message using the tokenizer.
    """
    tokens = tokenizer.encode(message)
    return len(tokens), model_max_length - len(tokens)

def truncate_message(prompt1, prompt2, tokenizer, model_max_length=2048):
    """
    Efficiently truncate `prompt1` to ensure the combined length of `prompt1` and `prompt2`
    fits within `model_max_length`.
    """
    prompt1_tokens = tokenizer.encode(prompt1, add_special_tokens=False)
    prompt2_tokens = tokenizer.encode(prompt2, add_special_tokens=False)

    max_prompt1_length = model_max_length - len(prompt2_tokens)

    if len(prompt1_tokens) > max_prompt1_length:
        prompt1_tokens = prompt1_tokens[:max_prompt1_length]

    truncated_prompt1 = tokenizer.decode(prompt1_tokens, clean_up_tokenization_spaces=True)
    
    return truncated_prompt1 + prompt2



class HaluEvalSumTask(Task):
    DATASET_NAME = "halu_eval_sum"
    DEFAULT_FILE_PATH = "/path/to/summarization_data.json"

    def __init__(self, file_path=None):
        """
        Initialize the HaluEvalSumTask with the dataset file path.
        Args:
            file_path: Path to the dataset in JSONL format.
        """
        self.file_path = file_path or self.DEFAULT_FILE_PATH  
        super().__init__()
        self.data = self._load_data()
    
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        if not self.file_path:
            raise ValueError("file_path is not set.")
        
        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        print(f"Loaded dataset from {self.file_path}")


    def _load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        return data

    def training_docs(self):
        return self._format_docs()

    def validation_docs(self):
        return []

    def test_docs(self):
        return self._format_docs()

    def has_training_docs(self):
        return True  

    def has_validation_docs(self):
        return False  

    def has_test_docs(self):
        return True  

    def _format_docs(self, tokenizer, max_doc_length=2048):
        formatted_docs = []
        adv=False

        for idx, item in tqdm(enumerate(self.data), desc="Formatting documents", total=len(self.data), unit="doc", miniters=1):
            try:

                document = item["document"]
                right_summary = item["right_summary"]
                hallucinated_summary = item["hallucinated_summary"]

                # Construct queries and contexts
                query = "Given the document below, determine whether the summary is factual.\n #Document#: " + document + "\n"
                context_right = '#Summary#: ' + right_summary + '\n'
                context_hallucinated = '#Summary#: ' + hallucinated_summary + '\n'

                # Truncate the #Document# part
                truncated_document_right = truncate_message(query, context_right, tokenizer, model_max_length=max_doc_length)
                truncated_document_hallucinated = truncate_message(query, context_hallucinated, tokenizer, model_max_length=max_doc_length)

                # Correct summary
                formatted_docs.append({
                    "id": f"{idx}_right",
                    "query": f"{truncated_document_right}\n"
                            "#Your Judgement#:",
                    "choices": [" Yes", " No"],
                    "gold": 0  # Correct summary is the first choice ("Yes")
                })

                # Hallucinated summary
                formatted_docs.append({
                    "id": f"{idx}_hallucinated",
                    "query": f"{truncated_document_hallucinated}\n"
                            "#Your Judgement#:",
                    "choices": [" Yes", " No"],
                    "gold": 1 # Hallucinated summary is the second choice ("No")
                })
                if adv:
                    formatted_docs[-2]["gold"] = 1
                    formatted_docs[-1]["gold"] = 0

            except Exception as e:
                print(f"Error processing document {idx}: {e}")
                continue

        
        return formatted_docs

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def doc_to_text(self, doc):
        return doc["query"]


    def construct_requests(self, doc, ctx):
        return [rf.loglikelihood(ctx, choice)[0] for choice in doc["choices"]]

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0

        completion_len = np.array([len(choice) for choice in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
        }

    def aggregation(self):
        """
        Define the aggregation methods for metrics.
        Returns:
            A dictionary with metric names and aggregation functions.
        """
        return {
            "acc": np.mean,
            "acc_norm": np.mean,
        }

    def get_dataloader(self, tokenizer, split=None,subset_size=None, batch_size=1, num_fewshot=0):
        """
        Create a DataLoader for the task.
        Args:
            tokenizer: Tokenizer to process the text data.
            subset_size: Size of the dataset subset (for sampling).
            batch_size: Batch size for the DataLoader.
            num_fewshot: Number of few-shot examples to include.
        Returns:
            A DataLoader object.
        """
        docs = self._format_docs(tokenizer)
        
        self.tokenizer = tokenizer  

        return create_dataloader(
            tokenizer,
            docs,
            self.fewshot_context,
            self.doc_to_target,
            subset_size=subset_size,
            batch_size=batch_size,
            num_fewshot=num_fewshot
        )

