from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import numpy as np
import torch
from collections import defaultdict


class ContrieverEmbeddingsAdapter(Embeddings):
    def __init__(self, model, tokenizer, max_batch_size=64):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size

    @property
    def device(self):
        return self.model.device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
        B = batch["input_ids"].shape[0]
        embeds = []
        for i in range(0, B, self.max_batch_size):
            subbatch = {k:v[i:i+self.max_batch_size] for k, v in batch.items()}
            embeds.append(self.model(**subbatch).to("cpu"))

        return embeds

    def embed_query(self, text: str) -> List[float]:
        batch = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512).to(
            self.device)
        embed = self.model(**batch).to("cpu")[0]
        return embed


def load_noise_sentences(*filenames, verbose=False):
    result = defaultdict(list)
    for f in filenames:
        data = torch.load(f)

        if verbose:
            print(f"Loaded {len(data['sentences'])} sentences from {f}")

        for k,v in data.items():
            result[k].extend(v)
        del data
    if verbose:
        print(f"Loaded {len(result['sentences'])} sentences total")
    return dict(result) #don't want to return defaultdict


def make_retrieval_sample(noisy_sentences, facts, sample_size, embedder, tokenizer):
    fact_size = num_tokens(facts, tokenizer)
    noise_size = sample_size - fact_size
    selected_noise, actual_noise_len = sample_sentences(noisy_sentences, noise_size, tokenizer)
    vectorstore, facts_ids = vectorstore_from_noise(selected_noise, embedder, facts)
    return dict(vectorstore=vectorstore, facts_ids=facts_ids, sample_size=fact_size+actual_noise_len)


def num_tokens(list_of_texts, tokenizer):
    return sum(len(tokenizer.encode(t)) for t in list_of_texts)


def sample_sentences(data, sample_size, tokenizer, sample_chunk_size=100):
    sents, embeds = data['sentences'], data['embeddings']
    selected = []
    N = len(sents)
    num_tokens = 0
    continue_search = True
    while continue_search:
        left = np.random.choice(N)
        right = min(left+sample_chunk_size, N)
        for i in range(left, right):
            l = len(tokenizer.encode(sents[i]))
            #print(f'num_tokens: {num_tokens}')
            if num_tokens + l > sample_size:
                continue_search = False
                break

            num_tokens += l
            selected.append((sents[i], embeds[i]))

    return selected, num_tokens


def vectorstore_from_noise(noise_embeddings, embedder, facts=None):
    vectorstore = FAISS.from_embeddings(noise_embeddings, embedder)
    if facts is not None:
        facts_ids = add_facts_to_vectorstore(vectorstore, facts, embedder)
    else:
        facts_ids = None

    return vectorstore, facts_ids


def add_facts_to_vectorstore(vectorstore, facts, embedder):
    facts_embeds = embedder.embed_documents(facts)
    facts_ids = vectorstore.add_embeddings(zip(facts, facts_embeds))
    return facts_ids