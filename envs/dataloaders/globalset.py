import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join as join_path

from .musique import LocalSetMusique, RetrievalMusique
from .hotpotqa import RetrievalHotPotQA
#from .infbench import LocalSetInfinity
#from .longbench import LocalSetLongbench
#from .loogle import LocalSetLoogle
#from .novelqa import LocalSetNovelQA
from .babilong import RetrievalBabiLong
#from .musique import RetrievalMusique

DATASETS = {
    "musique": RetrievalMusique,
    # "inf": LocalSetInfinity,
    # "longb": LocalSetLongbench,
    # "loogle": LocalSetLoogle,
    # "novel": LocalSetNovelQA,
    'babilong': RetrievalBabiLong,
    'hotpotqa': RetrievalHotPotQA
}

DATA_PATH = "data/"

PATHS = {
    "musique": join_path(DATA_PATH, "musique"),
    "inf": join_path(DATA_PATH, "inf_bench"),
    "longb": join_path(DATA_PATH, "longbench/data"),
    "loogle": join_path(DATA_PATH, "loogle"),
    "novel": join_path(DATA_PATH, "NovelQA"),
    'babilong': join_path(DATA_PATH, "babilong"),
    'hotpotqa': join_path(DATA_PATH, 'hotpotqa')
}

class SplittedSet(Dataset):
    def __init__(self, datasets, order, map):
        super().__init__()
        self.datasets = datasets
        self.order = order
        self.map = map

    def __len__(self):
        return len(self.order)
    
    def __getitem__(self, idx):
        dataset_idx = self.order[idx]
        task = self.datasets[dataset_idx].__getitem__(self.map[dataset_idx])
        self.map[dataset_idx] = (self.map[dataset_idx] + 1) % len(self.datasets[dataset_idx])
        return task



def create_datasets(
    dataset_names, tokenizer,
    lengths=-1,
    min_context_len=-1,
    max_context_len=1e7,
    type="qa",
    anno_type="real",
    seed=52,
):
    if isinstance(lengths, int):
        lengths = [lengths] * len(dataset_names)
    datasets  = []
    for i, name in enumerate(dataset_names):
        if name in ["musique", "inf", "longb", "loogle"]:
            ds = DATASETS[name](
                path=PATHS[name], tokenizer=tokenizer, length=lengths[i],
                min_context_len=min_context_len, max_context_len=max_context_len,
                type=type, anno_type=anno_type, seed=seed
            )
        datasets.append(ds)
    return datasets



class GlobalSet(Dataset):


    def __init__(self, datasets, split_strategy, proportions = 1):
        if not isinstance(proportions, list):
            proportions = [proportions] * len(datasets)
        self.datasets = datasets
        self.datasets_names = [d.name() for d in self.datasets]

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
    
    def get_train_test_split(self):
        if ":" in self.split_strategy:
            try:
                proportions = list(map(int, self.split_strategy.split(":")))
                train_len = int(proportions[0] * len(self.order) / (proportions[0] + proportions[1]))
                train_set = SplittedSet(self.datasets, self.order[:train_len], self.map)
                test_set = SplittedSet(self.datasets, self.order[train_len:], self.map)
                return train_set, test_set
            except:
                raise Exception("Incorrect split format")
        else:
            raise Exception("Incorrect split format")
        
    def print_statistics(self):
        total_samples, total_qa, total_summ, total_avg, total_max, total_min, total_weight =\
            0, 0, 0, 0, 0, 1e8, 0
        
        for dataset_name, dataset in zip(self.datasets_names, self.datasets):
            print("-" * 30)
            print(f"{dataset_name}:")
            print(f"Number of samples: {len(dataset)}")
            total_samples += len(dataset)

            qa, summ, context_lengths, weight = 0, 0, [], 0
            for task in dataset:
                context_lengths.append(task.context_length)
                if task.type == "qa":
                    qa += 1
                if task.type == "summary":
                    summ += 1
                weight += sys.getsizeof(task.context)

            total_qa += qa
            total_summ += summ
            total_weight += weight
            total_avg = int((total_avg * (total_samples - len(dataset)) + np.sum(context_lengths)) / total_samples)
            total_max = max(total_max, np.max(context_lengths))
            total_min = min(total_min, np.min(context_lengths))

            print(f"Average context length: {int(np.mean(context_lengths))}")
            print(f"Maximum context length: {np.max(context_lengths)}")
            print(f"Minimum context length: {np.min(context_lengths)}")

            print(f"Number of QA tasks: {qa}")
            print(f"Number of summary tasks: {summ}")

            print(f"Weight: {weight / 2 ** 20} MB")

        print("=" * 50)
        print(f"Total number of samples: {total_samples }")
        print(f"Total average context length: {total_avg}")
        print(f"Total maximum context length: {total_max}")
        print(f"Total minimum context length: {total_min}")
        print(f"Total number of QA tasks: {total_qa}")
        print(f"Total number of summary tasks: {total_summ}")
        print(f"Total weight: {total_weight / 2 ** 20} MB")
            

        

                

