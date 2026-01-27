import json
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)



class RetrievalLongBench(Dataset):
    def __init__(self,
                 path,
                 tokenizer = None,
                 length = -1,
                 min_context_len = None,
                 max_context_len = None,
                 seed = 52,
                 **kwargs
        ):
        super().__init__()
        self.length = length
        self.min_context_len = min_context_len
        self.max_context_len = max_context_len
        self.tokenizer = tokenizer
        np.random.seed(seed)
        self._load_data(path)


    def _load_data(self, path):
        self.tasks = []
        with open(path, 'r') as jsonl_file:
            json_list = list(jsonl_file)
            self.tasks.extend( [json.loads(json_str) for json_str in json_list] )
        if self.length >= 0:
            self.tasks = self.tasks[:self.length]
        print(path, "has been loaded.")


    def name(self):
        return 'LongBench'


    def _adapt_raw_sample(self, sample):
        """Adapt sample to unified format expected by the model"""
        return sample


    def __len__(self):
        return len(self.tasks)


    def __getitem__(self, idx):
        return self.tasks[idx]
