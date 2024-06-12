import numpy as np
from torch.utils.data import Dataset
import faiss


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


class RetrNoiseInjectionDataset(Dataset):
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

    def __getitem__(self, ind):
        sample = self.task_dataset[ind]
        sample_size = self.get_sample_size()
        num_facts = len(sample['facts'])
        num_noise = max(sample_size - num_facts, 0)
        noise_sentences = self.noise_sampler.get_sample(num_noise)
        sample['noise'] = noise_sentences
        return sample

    def __len__(self):
        return len(self.task_dataset)

    def get_sample_size(self):
        if isinstance(self.num_sentences, list):
            return self.gen.choice(self.num_sentences)
        else:
            return self.num_sentences
