import numpy as np
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
import torch.utils
# from rl.jax_text_env import TextEnv, TextMemory, TextMemoryItem
from rl.text_env import TextEnv, TextMemory, TextMemoryItem
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class GroundTruthReward:
    def __init__(self, only_at_max_step=False):
        super().__init__()
        self.only_at_max_step = only_at_max_step

    def reward(self, env, action):
        if self.only_at_max_step and (env.num_steps < env.max_steps):
            return 0.

        is_retrieved = []
        for r in env.references:
            is_retrieved.append(r in env.text_state)

        all_retrieved = all(is_retrieved)
        return float(all_retrieved)


class BabilongEnv(TextEnv):

    def __init__(self,
                 embedder,
                 embed_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 dataset,
                 max_steps = 3,
                 max_embed_length = 500,
                 action_embed_length = 64,
                 reward_model = GroundTruthReward()):
        
        super().__init__()

        self.dataset = dataset
        self.max_steps = max_steps
        self.max_embed_length = max_embed_length
        self.action_embed_length = action_embed_length

        self.embedder = embedder
        self.embed_tokenizer = embed_tokenizer
        self.reward_model = reward_model

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
        # self.sentences.extend([
        #   f"Fact number {i}: "  + str(f) for i, f in enumerate(sample['facts'])
        # ])
        self.sentences.extend(sample['facts'])
        self.facts_ids = np.arange(len(sample['noise']), len(self.sentences))
        self.sentences = np.array(self.sentences)

        self.ref_ids = []
        for i, f in enumerate(sample['facts']):
            if f in self.references:
                self.ref_ids.append(i + len(sample['noise']))

        self.ref_ids = np.array(self.ref_ids)[len(self.ref_ids) - len(self.references):]

    def reset(self, new_sample=None) -> TextMemory:
        if new_sample is not None:
            self._init_from_sample(new_sample)

        elif self.dataset is not None:
            N = len(self.dataset)
            i = np.random.randint(N)
            new_sample = self.dataset[i]
            self._init_from_sample(new_sample)

        self.num_steps = 0

        self.refs_found = []
        self.text_state = []
        
        return super().reset(self.question, self.sentences)
   

    def step(self, action: int):
        self.num_steps += 1

        done = self.num_steps >= self.max_steps
        
        text_memory, text_item, text_done = super().step(action)
        self.text_state.append(self.sentences[action])

        r = self._reward(action)
        if r > 1e-5:
            done = True
    
        return text_memory, text_item, r, done or text_done

    @property
    def device(self):
        return self.embedder.device

    def _reward(self, action):

        return self.reward_model.reward(self, action)

        # if action in self.ref_ids:
        #     self.refs_found.append(action)
        
        # if len(self.refs_found) == len(self.ref_ids):
        #     return 1.0
        # else:
        #     return 0.0

    def close(self):
        del self.sent_embeds
