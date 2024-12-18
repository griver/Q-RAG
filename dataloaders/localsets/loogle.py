import json
import torch
import numpy as np
from ast import literal_eval
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from ..utils import Task

class LocalSetLoogle(Dataset):
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
        return 'loogle'

    def _load_data(self, path):
        self.tasks = []

        with open(path + '/longdep_qa.jsonl', 'r') as json_file:
            json_list = list(json_file)
            raw_tasks = [(json.loads(json_str), "real", "qa", "longdep_qa") for json_str in json_list]

        with open(path + '/longdep_summarization.jsonl', 'r') as json_file:
            json_list = list(json_file)
            raw_tasks += [(json.loads(json_str), "synth", "summary", "longdep_summarization") for json_str in json_list]

        with open(path + '/shortdep_qa.jsonl', 'r') as json_file:
            json_list = list(json_file)
            raw_tasks += [(json.loads(json_str), "synth", "qa", "shortdep_qa") for json_str in json_list]

        with open(path + '/shortdep_cloze.jsonl', 'r') as json_file:
            json_list = list(json_file)
            raw_tasks += [(json.loads(json_str), "synth", "qa", "shortdep_cloze") for json_str in json_list]

        for task, anno_type, task_type, partition in tqdm(raw_tasks, "Loogle load"):
            context = task["input"]
            context_len = len(self.tokenizer(context)["input_ids"])
            if context_len > self.max_context_len or context_len < self.min_context_len:
                continue
            
            if isinstance(task["qa_pairs"], str):
                try:
                    qa_pairs = literal_eval(task["qa_pairs"])
                except:
                    continue
            elif isinstance(task["qa_pairs"], list):
                qa_pairs = task["qa_pairs"]
            else:
                continue

            for qa_pair in qa_pairs:
                if task_type == "qa" and (self.type == "any" or self.type == "qa"):
                    self.tasks.append(Task("qa", anno_type, context_len, context, qa_pair["A"], qa_pair["Q"], "Loogle", partition))
                if self.type == "any" or self.type == "summary":
                    self.tasks.append(Task("summary", "synth", context_len, context, qa_pair["S"], "Your task is to summarize the following context", "Loogle", partition))

        self.tasks = np.random.permutation(self.tasks)
        if self.length >= 0:
            self.tasks = self.tasks[:self.length]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

    
    