import numpy as np
from torch import nn, Tensor
import torch
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
import torch.utils
from rl.text_env import TextEnv, TextMemory, TextMemoryItem
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class BabilongEnv(TextEnv):

    def __init__(self,
                 embedder: nn.Module,
                 embed_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 dataset,
                 max_steps = 3,
                 max_embed_length = 512,
                 reward_model=None,
                 done_when_rewarded=True):
        
        super().__init__()

        self.done_when_rewarded=done_when_rewarded
        self.dataset = dataset
        self.max_steps = max_steps
        self.max_embed_length = max_embed_length

        self.embedder = embedder
        self.embed_tokenizer = embed_tokenizer
        self.rmodel = reward_model

        self.references = None
        self.question = None
        self.sentences = None
        self.facts_ids = None
       
        self.num_steps = 0

    def _init_from_sample(self, sample):

        self.references = list(sample['references'])
        self.question = sample['question']  # append as this is a single str
        self.answer = sample['answer']
        self.sentences = []
        self.sentences.extend(sample['noise'])
        self.sentences.extend(sample['facts'])
        self.facts_ids = np.arange(len(sample['noise']), len(self.sentences))
        self.sentences = np.array(self.sentences)


    def reset(self, new_sample=None) -> TextMemory:
        if new_sample is not None:
            self._init_from_sample(new_sample)

        elif self.dataset is not None:
            N = len(self.dataset)
            i = np.random.randint(N)
            new_sample = self.dataset[i]
            self._init_from_sample(new_sample)

        if self.rmodel:
            self.rmodel.reset()

        self.num_steps = 0
        
        return super().reset(self.question, self.sentences)
   

    def step(self, action: int):
        self.num_steps += 1

        done = self.num_steps >= self.max_steps
        r = self._reward()

        if self.done_when_rewarded and (r != 0.):
            done = True

        text_memory, text_item, text_done = super().step(action)
    
        return text_memory, text_item, r, done or text_done

    @property
    def device(self):
        return self.embedder.device

    def _reward(self):
        if not self.rmodel:
            return 0.
        return self.rmodel.reward(self)

    def close(self):
        del self.sent_embeds
