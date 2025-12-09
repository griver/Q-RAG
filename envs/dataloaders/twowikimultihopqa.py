import json
import numpy as np
from torch.utils.data import Dataset



class Retrieval2WikiMultihopQA(Dataset):

    def __init__(self, path, split, length = -1, seed = 52):
        super().__init__()
        if split not in ['train', 'dev', 'test']:
            raise ValueError(f'Unknown split for 2WikiMultihopQA dataset: {split}')
        self.split = split
        self.length = length
        np.random.seed(seed)
        self.__load_data(path)


    def __load_data(self, path):
        self.samples = []
        with open(f"{path}/{self.split}.json", 'r') as f:
            self.samples = json.load(f)
        self.samples = np.random.permutation(self.samples)
        if self.length >= 0:
            self.samples = self.samples[:self.length]
        print(f"2WikiMultihopQA-{self.split} has been loaded. Samples: {len(self.samples)}")


    def name(self):
        return "2WikiMultihopQA"


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        return self.samples[idx]
