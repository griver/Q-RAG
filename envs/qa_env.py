import numpy as np
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
import torch.utils
from nltk.probability import gt_demo
from envs.text_env import PositionProcessor, TextEnv
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from rl.feedback.feedback import AFeedbackModel
from envs.utils import TextMemory


class PositionalGTReward:
    """
    !THIS IS AN OLD VERSION OF A REWARD MODEL. USE rl.feedback.feedback.GroundTruthFeedback INSTEAD!

    This version takes into account position of the support facts.
    In babi tasks several events could have completely identical text descriptions,
    but only one of them can be considered a support fact/reference fact.

    I.E. Merry could visit the same location several times.
    But only the last event allows us to tell where she is at the end of the story.

    This reward takes into account temporal information that allows to distinguish
    true support facts, from similar events.
    """
    def __init__(self, penalize_extra_steps=False,  only_at_max_step=False):
        super().__init__()
        self.only_at_max_step = only_at_max_step
        self.penalize_extra_steps = penalize_extra_steps

    def reward(self, env, action):
        if self.only_at_max_step and (env.num_steps < env.max_steps):
            return 0.

        pred_sf = set(map(int, env.memory.item_ids))
        gt_sf = set(env.references_idx)
        if self.penalize_extra_steps:
            r = 0.5 + 0.5 * len(gt_sf) / (len(pred_sf) + 1e-5)
        else:
            r = 1.0
        return r if gt_sf.issubset(pred_sf) else 0.0


class QAEnv(TextEnv):
    """
    Question Answering Environment is an RL interface to interact with QA Datasets
    """
    def __init__(self,
                 dataset,
                 max_steps: int,
                 positions_processor: PositionProcessor,
                 action_embed_length: int,
                 feedback_model: AFeedbackModel,
                 #reward_model = GroundTruthReward(),
                 max_embedding_batch: int = 10000,
                 separator: str = " [SEP] ",             
                 sort_by_index: bool = True
        ):
        
        super().__init__()

        self.dataset = dataset
        self.max_steps = max_steps
        self.max_embedding_batch = max_embedding_batch
        self.action_embed_length = action_embed_length
        self.feedback_model = feedback_model
        #self.reward_model = reward_model
        self.positions_processor = positions_processor
        self.separator = separator
        self.sort_by_index = sort_by_index


        #self.references = None we don't need this one anymore use PositionalGTReward
        self.question = None
        self.sentences = None
       
        self.num_steps = 0

    def copy(self):
        return QAEnv(
            dataset = self.dataset,
            max_steps = self.max_steps,
            positions_processor = self.positions_processor,
            action_embed_length = self.action_embed_length,
            feedback_model=self.feedback_model.copy(),
            #reward_model = self.reward_model,
            max_embedding_batch = self.max_embedding_batch,
            separator = self.separator,             
            sort_by_index = self.sort_by_index
        )

    def _init_from_sample(self, sample):
        self.sample_id = sample["id"]
        self.question = sample['question']
        self.answer = sample['answer']
        self.sentences = np.asarray(sample['chunks'])
        self.references_idx = sample.get('sf_idx')
        self.references = [self.sentences[i] for i in self.references_idx]

    def _make_obs_and_info(self):
        """Right now this function is used only to prepare input for a feedback model"""
        pred_idx = [int(i) for i in self.memory.item_ids]
        pred_chunks = [self.sentences[i] for i in pred_idx]
        obs = {
            'question': self.question,
            'sample_id': self.sample_id,
            'pred_idx': pred_idx,
            'pred_chunks': pred_chunks,
        }
        info = {
            'sf_idx': self.references_idx,
            'sf_chunks': self.references,
        }
        return obs, info

    def reset(self, new_sample=None) -> TextMemory:
        if new_sample is not None:
            self._init_from_sample(new_sample)

        elif self.dataset is not None:
            N = len(self.dataset)
            i = np.random.randint(N)
            new_sample = self.dataset[i]
            self._init_from_sample(new_sample)

        self.num_steps = 0

        #self.refs_found = []
        #self.text_state = []

        obs = super()._reset(self.question, self.sentences)
        fb_obs, fb_info = self._make_obs_and_info()
        self.feedback_model.reset(fb_obs, fb_info)

        return obs

    def step(self, action: int):
        self.num_steps += 1

        truncated = self.num_steps >= self.max_steps
        
        text_memory, text_item, done = super()._step(action)

        fb_obs, fb_info = self._make_obs_and_info()

        # r = self._reward(action)
        # if r > 1e-5:
        #     done = True
        fb = self.feedback_model.get_feedback(fb_obs, fb_info, truncated)
        return text_memory, text_item, fb['reward'], fb['terminated'] or truncated

    # s_t = q + {R#1} + f^{0}_1 + {R#2} + n^{17}_2 + {R#0} + f^{35}_0
    # s_t = Emb_s(q + f^{0}_1 + f^{35}_0) * Emb_a(n^{17}_2)

    @property
    def device(self):
        return self.embedder.device

    # def _reward(self, action):
    #     return self.reward_model.reward(self, action)

    def get_sample_len(self, tokenizer):
        """
        Return total length of all texts in the current retrieval task
        """
        total_len = len(tokenizer(self.question)['input_ids'])
        total_len += sum(len(chunk) for chunk in tokenizer(list(self.sentences))['input_ids'])
        return total_len