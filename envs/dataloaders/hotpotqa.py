import json
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import logging

from envs.dataloaders.utils import Task
logger = logging.getLogger(__name__)

class RetrievalHotPotQA(Dataset):
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
        split = kwargs.pop('split')
        if split not in ['train', 'eval', 'all',  'fullwiki_eval']:
            #'all' includes 'train' and 'eval' but not fullwiki_eval
            raise ValueError(f'unknown split for hotpotqa dataset: {split}')
        self.split = split
        logger.info(f"{type(self)} received unknown kwargs: {kwargs}")
        self.length = length
        self.min_context_len = min_context_len
        self.max_context_len = max_context_len
        self.tokenizer = tokenizer

        np.random.seed(seed)
        self._load_data(path)

    def name(self):
        return 'hotpotqa'

    def _load_data(self, path):
        self.tasks = []

        raw_tasks = []
        if self.split in ['eval', 'all']:
            with open(path + '/hotpot_dev_distractor_v1.json', 'r') as json_file:
                raw_tasks.extend(map(lambda x: (x, 'eval'), json.load(json_file)))

        if self.split in ['train', 'all']:
            with open(path + '/hotpot_train_v1.1.json', 'r') as json_file:
                raw_tasks.extend(map(lambda x: (x, 'train'), json.load(json_file)))

        if self.split == 'fullwiki_eval':
            with open(path + '/hotpot_dev_fullwiki_v1.json', 'r') as json_file:
                raw_tasks.extend(map(lambda x: (x, 'fullwiki_eval'), json.load(json_file)))

        for task, partition in tqdm(raw_tasks, "HotPotQA load"):
            if self.min_context_len is None and self.max_context_len is None:
                # no sample filtering here
                self.tasks.append(self._adapt_raw_sample(task))
            else:
                # we filter some of the samples based on their length
                context = " ".join(title + " ".join(sentences) for title, sentences in task["context"])
                context = task['question'] + context
                context_len = len(self.tokenizer(context)["input_ids"])

                if self.min_context_len <= context_len <= self.max_context_len:
                    self.tasks.append(self._adapt_raw_sample(task))

        self.tasks = np.random.permutation(self.tasks)
        if self.length >= 0:
            self.tasks = self.tasks[:self.length]


    def _adapt_raw_sample(self, sample):
        """Adapt sample to unified format expected by the model"""
        return sample


    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]