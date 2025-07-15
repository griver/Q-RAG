from .babilong_utils import TaskDataset
import numpy as np
from torch.utils.data import Dataset

import datasets
from os.path import join as join_path
from typing import List, Any, Tuple


class RetrSentenceSampler:
    def __init__(self,
                 dataset,
                 shuffle=True,
                 subsample_size = 100,
                 random_seed=42):
        self.sample_ind = 0
        self.dataset = dataset
        self.shuffle = shuffle
        self.subsample_size = subsample_size
        self.gen = np.random.default_rng(seed=random_seed)

    def get_sample(self, num_sentences):
        sample = []
        if num_sentences <= 0:
            return sample

        sentences = []
        while len(sentences) < num_sentences:
            n = min(num_sentences - len(sentences), self.subsample_size)
            new_sents = self.sentences_from_book(max_sentences_to_sample=n)
            sentences.extend(new_sents)

        return sentences[:num_sentences]

    def sentences_from_book(self, max_sentences_to_sample):
        sentences = []
        for attempt in range(100):
            book = self.next_book()
            if self.shuffle:
                if len(book) == 0:
                    continue
                i = self.gen.choice(len(book))
                book = book[i:i+max_sentences_to_sample]  # start from random position in text
            sentences.extend(book)
            if len(sentences) > 0:
                break
        else:
            raise ValueError(f'Tried to sample sentences from dataset {attempt} times but did not succeed')
        return sentences

    def next_book(self):
        if self.shuffle:
            sample_ind = self.gen.choice(len(self.dataset))
            sample = self.dataset[int(sample_ind)]['sentences']
        else:
            sample = self.dataset[int(self.sample_ind)]['sentences']
            self.sample_ind += 1
            self.sample_ind = self.sample_ind % len(self.dataset)
        return sample


def shuffle(noise: List[Any], facts :List[Any]) -> Tuple[List[Any], List[int]]:
    """
    Shuffles noise chunks with fact chunks while keeping relative order of facts intact
    """
    N_facts = len(facts)
    N = len(noise) + N_facts
    facts_idx = sorted(np.random.choice(N, size=N_facts, replace=False))
    all = []
    noise_i, fact_i = 0, 0
    for i in range(N):
        if fact_i < N_facts and i == facts_idx[fact_i]:
            all.append(facts[fact_i])
            fact_i += 1
        else:
            all.append(noise[noise_i])
            noise_i += 1
    return all, facts_idx


class RetrievalBabiLong(Dataset):

    @classmethod
    def create(cls, path, task, num_chunks, seed=42, noise_data_path="pg19/",
               split='train'):

        fact_dataset = TaskDataset(path, task, split)  # max_n_facts=10)
        noise_path = join_path(path, noise_data_path, split)
        noise_dataset = datasets.load_from_disk(noise_path)
        noise_sampler = RetrSentenceSampler(noise_dataset, shuffle=True, random_seed=seed)

        dataset = cls(
            task_dataset=fact_dataset,
            noise_sentence_sampler=noise_sampler,
            num_sentences=num_chunks,
            random_seed=seed,
        )
        return dataset


    def __init__(
        self,
        task_dataset,
        noise_sentence_sampler,
        num_sentences,
        random_seed=42
    ):
        self.task_dataset = task_dataset
        self.noise_sampler = noise_sentence_sampler
        self.num_sentences = num_sentences
        if random_seed:
            self.gen = np.random.default_rng(seed=random_seed)

    def name(self):
        return 'babilong'

    def __getitem__(self, ind):
        sample = self.task_dataset[ind]
        sample_size = self.get_sample_size()
        num_facts = len(sample['facts'])
        num_noise = max(sample_size - num_facts, 0)
        noise_sentences = self.noise_sampler.get_sample(num_noise)
        sample['noise'] = noise_sentences

        chunks, facts_idx = shuffle(noise_sentences, sample['facts'])
        sample['chunks'] = chunks
        sample['facts_idx'] = facts_idx #all babi sentences

        if 'references_idx' in sample:
            new_ref_idx = [sample['facts_idx'][old_id] for old_id in sample['references_idx']]
            sample['references_idx'] = new_ref_idx # support facts from babi

        return sample

    def __len__(self):
        return len(self.task_dataset)

    def get_sample_size(self):
        if isinstance(self.num_sentences, list):
            return self.gen.choice(self.num_sentences)
        else:
            return self.num_sentences

    def get_partition_info(self):
        # Получаем из task_dataset
        task_ds = self.task_dataset
        if not hasattr(self.task_dataset,'get_partition_name'):
            task_ds = task_ds.task_dataset

        return task_ds.get_partition_name()


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Undi95/Meta-Llama-3-8B-Instruct-hf")
    dataset = RetrievalBabiLong.create(
        'data/babilong/', 'qa2', 100,
        noise_data_path='pg19-with-sentences/', seed=42)
    print("PARTITION:", dataset.get_partition_info())

    for i in np.random.randint(10000, size=5):
        sample = dataset[i]
        print(list(sample.keys()))
        print('Chunks:')
        for i, chunk in enumerate(sample['chunks']):
            chunk_without_newlines = chunk.replace('\n', ' ')
            print(f'#{i}: {chunk_without_newlines}', end='')
            print(" [FACT]" if i in sample['facts_idx'] else " [NOISE]", end='')
            if 'references_idx' in sample:
                print( " [SUPPORT]" if i in sample['references_idx'] else "")
        print('REFERENCES')
        print('\n'.join(sample['references']))
        print('REFERENCES from idx:')
        print('\n'.join(sample['chunks'][idx] for idx in sample['references_idx']))
        # print(f"context len: {sample.context_length}")
        # print(f'Q: {sample.question}')
        # print(f'A: {sample.answer}')
        #print(f"Contex:\n {sample.context[:1024]}...")
        print('====================')
