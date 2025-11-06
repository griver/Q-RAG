import random
import torch
from hydra.utils import instantiate
from envs.dataloaders import RetrievalHotPotQA
from envs.dataloaders import RetrievalMusique



class RetrievalCombined(torch.utils.data.Dataset):

    def __init__(self, dataset1: RetrievalHotPotQA, dataset2: RetrievalMusique, shuffle: bool):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self._name = "combined"
        self.rng = random.Random(42)
        self.shuffle = shuffle
        self.total_length = len(self.dataset1) + len(self.dataset2)
        self.indices = list(range(self.total_length))
        if self.shuffle:
            self.rng.shuffle(self.indices)


    def name(self):
        return self._name


    def __len__(self):
        return self.total_length


    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        if actual_idx < len(self.dataset1):
            sample = self.dataset1[actual_idx]
            return {**sample, "source": "hotpotqa"}
        else:
            sample = self.dataset2[actual_idx - len(self.dataset1)]
            return {**sample, "source": "musique"}


    def reshuffle(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
