from datasets.dataset_dict import DatasetDict, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np
from torch import nn, Tensor
import torch
import os
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
from rl.text_env import TextEnv

def isSubArray_v2(A, B):
    
    A = np.array(A)
    B = np.array(B)
    
    n, m = len(A), len(B)
    
    if m > n:
        return False
    
    windows = np.lib.stride_tricks.sliding_window_view(A, m)
    return np.any(np.all(windows == B, axis=1))


class WordsCounterEnv(TextEnv):
    def __init__(self, 
                 dataset: DatasetDict, # load_dataset("AIRI-NLP/quality_counter_new_1024")
                 block_size: int,
                 max_length: int,
                 embedder: nn.Module,
                 embed_tokenizer: Union[PreTrainedTokenizer,PreTrainedTokenizerFast],
                 max_steps_count: int,
                 max_embed_length: int = 512,
                 add_question: bool = False,
                 subset: str = "train") -> None:
        
        self.dataset = dataset[subset].with_format("numpy") 
        self.features = ['context', 'word', 'claim', 'label']
        self.index = -1
        self.n = len(self.dataset)
        self.embed_tokenizer = embed_tokenizer
        self.tokenizer = embed_tokenizer
        self.block_size = block_size
        self.max_length = max_length
        self.embedder = embedder
        self.add_question = add_question
        self.max_steps_count = max_steps_count
        self.max_embed_length = max_embed_length
        self.action_embed_length = block_size

    def reset(self):
        self.index = (self.index + 1) % self.n
        claim = self.dataset[self.index]["claim"]
        context = self.dataset[self.index]["context"]
        self.label = self.dataset[self.index]["label"]
        word = self.dataset[self.index]["word"]
        self.word = word
        self.claim = claim

        if self.add_question:
            tok_seq = self.tokenizer(context, max_length=self.max_length, truncation=True)
        else:
            tok_seq = self.tokenizer(claim, context, max_length=self.max_length, truncation=True)
        # rm the first [CLS] token
        input_ids = np.asarray(tok_seq["input_ids"]).reshape(-1)
        attention_mask = np.asarray(tok_seq["attention_mask"]).reshape(-1)
        
        T = input_ids.shape[0]
        pad_size = 0 if T % self.block_size == 0 else (self.block_size - T % self.block_size) 
        self.input_ids = np.pad(input_ids, (0, pad_size), constant_values=(0, int(self.tokenizer.pad_token_id)))
        self.attention_mask = np.pad(attention_mask, (0, pad_size))
        self.T = T + pad_size

        self.blocks = self._split_into_blocks()
        
        self.decoded_blocks = [
            self.tokenizer.decode(self.blocks["input_ids"][i]) for i in range(self.T // self.block_size)  
        ]

        word_tokens = self.tokenizer(word, max_length=self.max_length)["input_ids"][1:-1]

        self.word_positions = [
            i for i, b in enumerate(self.blocks["input_ids"]) if  isSubArray_v2(b, word_tokens)
        ]

        question = claim if self.add_question else ""

        question_tokens = self.tokenizer(claim, max_length=self.max_length)["input_ids"]
        assert isSubArray_v2(question_tokens, word_tokens)

        return super().reset(word, self.decoded_blocks)

    def _split_into_blocks(self):
        return {
            "input_ids": self.input_ids.reshape(self.T // self.block_size, self.block_size),
            "attention_mask": self.attention_mask.reshape(self.T // self.block_size, self.block_size)
        }
    
    def step(self, action: int):

        reward = 0.0

        if action in self.word_positions and action not in self.memory.item_ids:
            reward = 1.0

        memory, memory_item, _ = super().step(action)

        done = len(memory.item_ids) >= min(memory.embeds.shape[0], self.max_steps_count)

        return  memory, memory_item, reward, done

