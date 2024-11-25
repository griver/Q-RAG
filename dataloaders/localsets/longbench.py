import os
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from utils import Task

class LocalSetLongbench(Dataset):
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

    def _load_data(self, path):
        self.tasks = []

        raw_tasks = []
        for filename in os.listdir(path):
            with open(path + '/' + filename, 'r') as json_file:
                json_list = list(json_file)
                raw_tasks += [(json.loads(json_str), filename) for json_str in json_list]


        for task, filename in tqdm(raw_tasks, "LongBench load"):
            if task["language"] != "en":
                continue
            context = task["context"]
            context_len = len(self.tokenizer(context)["input_ids"])
            if context_len > self.max_context_len or context_len < self.min_context_len:
                continue
            task_type = "qa" if len(task["input"]) > 0 else "summary"
            anno_type = "real" if "passage" not in filename else "synth"
            if (self.type != "any" and task_type != self.type) or (self.anno_type != "any" and anno_type != self.anno_type):
                continue
            question = task["input"] if len(task["input"]) > 0 else "Your task is to summarize the following context"
            self.tasks.append(Task(task_type, anno_type, context_len, context, task["answers"][0], question, "LongBench", filename.split(".")[0]))

        self.tasks = np.random.permutation(self.tasks)
        if self.length >= 0:
            self.tasks = self.tasks[:self.length]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

    
    