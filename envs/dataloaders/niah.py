import json
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import logging
import nltk

logger = logging.getLogger(__name__)

class NIAH(Dataset):
    def __init__(self,
                 path,
                 split,
                 seed = 52,
                 length=-1,
                 **kwargs
        ):
        super().__init__()
        if split not in ['validation', 'test', 'train', 'eval']:
            raise ValueError(f'unknown split for NIAH dataset: {split}')
        self.split = split
        np.random.seed(seed)
        self.length = length
        self._load_data(path)

    def name(self):
        return 'niah'

    def _load_data(self, path):
        self.tasks = []
        raw_tasks = []

        if self.split in ['train', 'validation']:
            with open(path + '/validation.jsonl', 'r') as json_file:
                for line in json_file:
                    raw_tasks.append(json.loads(line))

        if self.split in ['eval', 'test']:
            with open(path + '/test.jsonl', 'r') as json_file:
                for line in json_file:
                    raw_tasks.append(json.loads(line))

        for task in tqdm(raw_tasks, "NIAH load"):
            self.tasks.append(self._adapt_raw_sample(task))
            
        self.tasks = np.random.permutation(self.tasks)
        if self.length >= 0:
            self.tasks = self.tasks[:self.length]

    def _adapt_raw_sample(self, sample):
        context = sample['input']
        answer = ",".join(sample["outputs"])
        
        sentences = nltk.sent_tokenize(context)
        question = sentences[-1]
        sf_idx = []

        for idx, s in enumerate(sentences):
            for o in  sample["outputs"]:
                if o in s:
                    sf_idx.append(idx)

        return {
            'id': sample["index"],
            'question': question,
            'answer': answer,
            'chunks': sentences[:-1],
            'sf_idx': sf_idx,
        }


    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]