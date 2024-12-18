import json
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from ..utils import Task

class LocalSetMusique(Dataset):
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
        return 'musique'

    def _load_data(self, path):
        self.tasks = []

        if self.type not in ["qa", "any"] or self.anno_type not in ["real", "any"]:
            return

        with open(path + '/musique_ans_v1.0_dev.jsonl', 'r') as json_file:
            json_list = list(json_file)
            raw_tasks = [(json.loads(json_str), "dev") for json_str in json_list]

        with open(path + '/musique_ans_v1.0_train.jsonl', 'r') as json_file:
            json_list = list(json_file)
            raw_tasks += [(json.loads(json_str), "train") for json_str in json_list]

        for task, partition in tqdm(raw_tasks, "MuSiQue load"):
            context = ""
            for text in task["paragraphs"]:
                title = text["title"]
                paragraph_text = text["paragraph_text"]
                context += f"TITLE: {title}\nTEXT: {paragraph_text}\n\n"
            context_len = len(self.tokenizer(context)["input_ids"])
            if context_len > self.max_context_len or context_len < self.min_context_len:
                continue
            self.tasks.append(Task("qa", "real", context_len, context, task["answer"], task["question"], "MuSiQue", partition))

        self.tasks = np.random.permutation(self.tasks)
        if self.length >= 0:
            self.tasks = self.tasks[:self.length]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]

    
    