import json
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from ..utils import Task

PARTITIONS = {
    # "code_debug": 
    #   {"filename": "code_debug.jsonl", "type": "qa", "anno_type": "real"},
    # "code_run":  
    #   {"filename": "code_run.jsonl", "type": "qa", "anno_type": "synth"},
    # "kv_retrieval":  
    #   {"filename": "kv_retrieval.jsonl", "type": "qa", "anno_type": "synth"},
    "longbook_choice_eng":  
      {"filename": "/longbook_choice_eng.jsonl", "type": "qa", "anno_type": "real"},
    "longbook_qa_eng":  
      {"filename": "/longbook_qa_eng.jsonl", "type": "qa", "anno_type": "real"},
    "longbook_sum_eng":  
      {"filename": "/longbook_sum_eng.jsonl", "type": "summary", "anno_type": "real"},
    "longdialogue_qa_eng":  
      {"filename": "/longdialogue_qa_eng.jsonl", "type": "qa", "anno_type": "synth"},
    # "math_calc":  
    #   {"filename": "math_calc.jsonl", "type": "qa", "anno_type": "synth"},
    # "math_find":  
    #   {"filename": "math_find.jsonl", "type": "qa", "anno_type": "synth"},
    # "number_string":  
    #   {"filename": "number_string.jsonl", "type": "qa", "anno_type": "synth"},
    # "passkey":  
    #   {"filename": "passkey.jsonl", "type": "qa", "anno_type": "synth"},
}

class LocalSetInfinity(Dataset):
    def __init__(self, path, tokenizer, length = -1, min_context_len = -1, max_context_len = 1e7, type = "qa", anno_type = "real", seed = 52):
        super().__init__()
        self.length = length
        self.min_context_len = min_context_len
        self.max_context_len = max_context_len
        self.type = type
        self.anno_type = anno_type
        self.tokenizer = tokenizer

        np.random.seed(seed)
        self._load_data(path)

    def name(self):
        return 'inf'

    def _load_data(self, path):
        self.tasks = []

        raw_tasks = []
        for name, part in PARTITIONS.items():
            with open(path + part["filename"], 'r') as json_file:
                json_list = list(json_file)
                raw_tasks += [(json.loads(json_str), name, part["type"], part["anno_type"]) for json_str in json_list]

        for task in tqdm(raw_tasks, "InfiniteBench load"):
            context = task[0]["context"]
            context_len = len(self.tokenizer(context)["input_ids"])
            if context_len > self.max_context_len or context_len < self.min_context_len:
                continue
            if (self.type != "any" and task[2] != self.type) or (self.anno_type != "any" and task[3] != self.anno_type):
                continue
            self.tasks.append(Task(task[2], task[3], context_len, context, task[0]["answer"][0], task[0]["input"], "InfiniteBench", task[1], task[0]["options"]))

        self.tasks = np.random.permutation(self.tasks)
        if self.length >= 0:
            self.tasks = self.tasks[:self.length]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

    
    