import os
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from ..utils import Task

class LocalSetNovelQA(Dataset):
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
        return 'novel'

    def _load_data(self, path):
        print("WARNING! NovelQA hasn't ground truth answers")
        self.books = {}
        for filename in tqdm(os.listdir(path + "/Books/PublicDomain"), "NovelQA books load"):
            with open(path + "/Books/PublicDomain/" + filename, "r") as file:
                book = "".join(file.readlines())
                context_len = len(self.tokenizer(book)["input_ids"])
                if context_len > self.max_context_len or context_len < self.min_context_len:
                    continue
                self.books[filename.split(".")[0]] = (book, context_len)
        
        self.questions = {}
        for filename in tqdm(os.listdir(path + "/Data/PublicDomain"), "NovelQA questions load"):
            if filename.split(".")[0] not in self.books:
                continue
            with open(path + "/Data/PublicDomain/" + filename, "rb") as f:
                self.questions[filename.split(".")[0]] = json.load(f)

        self.order, self.map = [], {}
        for key in self.questions.keys():
            self.order += [key] * len(self.questions[key])
            self.map[key] = 0

        self.order = np.random.permutation(self.order)
        if self.length >= 0:
            self.order = self.order[:self.length]

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        key = self.order[idx]
        context, context_length = self.books[key]
        q_idx = self.map[key]
        self.map[key] += 1
        options = [option for option in self.questions[key][q_idx]["Options"].values()]
        return Task("qa", "real", context_length, context, "NOT PROVIDED", self.questions[key][q_idx]["Question"], "NovelQA", "united", options)
        

    
    