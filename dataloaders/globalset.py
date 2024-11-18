import torch
import numpy as np
from torch.utils.data import Dataset

from .localsets.musique import LocalSetMusique

DATASETS = {
    "musique": LocalSetMusique,
}

PATHS = {
    "musique": "../data_sources/musique",
}

class GlobalSet(Dataset):
    def __init__(self, datasets, tokenizer, split_strategy, proportions = 1, lengths = -1, min_context_len = -1, max_context_len = 1e7, type = "qa", anno_type = "real", seed = 52):
        super().__init__()
        if isinstance(lengths, int):
            lengths = [lengths] * len(datasets)
        if not isinstance(proportions, list):
            proportions = [proportions] * len(datasets)
        self.datasets = [
            DATASETS[dataset](
                path = PATHS[dataset], tokenizer = tokenizer, length = length, 
                min_context_len = min_context_len, max_context_len = max_context_len,
                type = type, anno_type = anno_type, seed = seed
            ) for dataset, length in zip(datasets, lengths)
        ]

        self.split_strategy = split_strategy
        self.order = []
        for i, dataset in enumerate(self.datasets):
            self.order += [i] * int(len(dataset) * proportions[i])
        self.order = np.random.permutation(self.order)

        self.map = {i: 0 for i in range(len(self.datasets))}

    def __len__(self):
        return len(self.order)
    
    def __getitem__(self, idx):
        dataset_idx = self.order[idx]
        task = self.datasets[dataset_idx].__getitem__(self.map[dataset_idx])
        self.map[dataset_idx] = (self.map[dataset_idx] + 1) % len(self.datasets[dataset_idx])
        return task
