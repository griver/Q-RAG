import numpy as np
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
import torch.utils
from nltk.probability import gt_demo
from envs.text_env import PositionProcessor, TextEnv
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from envs.utils import TextMemory


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


class PositionalGTReward(GroundTruthReward):
    """
    This version takes into account position of the support facts.
    In babi tasks several events could have completely identical text descriptions,
    but only one of them can be considered a support fact/reference fact.

    I.E. Merry could visit the same location several times.
    But only the last event allows us to tell where she is at the end of the story.

    This reward takes into account temporal information that allows to distinguish
    true support facts, from similar events.
    """
    def reward(self, env, action):
        if self.only_at_max_step and (env.num_steps < env.max_steps):
            return 0.

        pred_sf = set(map(int, env.memory.item_ids))
        gt_sf = set(env.references_idx)
        return 1.0 if gt_sf.issubset(pred_sf) else 0.0


class BabilongEnv(TextEnv):

    def __init__(self,
                 dataset,
                 max_steps: int,
                 positions_processor: PositionProcessor,
                 action_embed_length: int,
                 reward_model = GroundTruthReward(),
                 max_embedding_batch: int = 500,
                 separator: str = " [SEP] ",             
                 sort_by_index: bool = True
        ):
        
        super().__init__()

        self.dataset = dataset
        self.max_steps = max_steps
        self.max_embedding_batch = max_embedding_batch
        self.action_embed_length = action_embed_length
        self.reward_model = reward_model
        self.positions_processor = positions_processor
        self.separator = separator
        self.sort_by_index = sort_by_index


        self.references = None
        self.question = None
        self.sentences = None
        self.facts_idx = None
       
        self.num_steps = 0

    def copy(self):
        return BabilongEnv(
            dataset = self.dataset,
            max_steps = self.max_steps,
            positions_processor = self.positions_processor,
            action_embed_length = self.action_embed_length,
            reward_model = self.reward_model,
            max_embedding_batch = self.max_embedding_batch,
            separator = self.separator,             
            sort_by_index = self.sort_by_index
        )

    def _init_from_sample(self, sample):
        self.references = list(sample['references'])
        self.question = sample['question']  # append as this is a single str
        self.answer = sample['answer']
        self.sentences = np.asarray(sample['chunks'])
        self.facts_idx = list(sample['facts_idx'])
        self.references_idx = sample.get('references_idx')
        
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
        
        return super()._reset(self.question, self.sentences)
   

    def step(self, action: int):
        self.num_steps += 1

        done = self.num_steps >= self.max_steps
        
        text_memory, text_item, text_done = super()._step(action)
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

    def get_sample_len(self, tokenizer):
        """
        Return total length of all texts in the current retrieval task
        """
        total_len = len(tokenizer(self.question)['input_ids'])
        total_len += sum(len(chunk) for chunk in tokenizer(list(self.sentences))['input_ids'])
        return total_len