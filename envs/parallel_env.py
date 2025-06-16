from abc import abstractmethod
import numpy as np
from torch import nn, Tensor
import torch
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import os
from torch.nn.utils.rnn import pad_sequence
from sortedcontainers import SortedList
from functools import reduce
from envs.text_env import TextEnv
from envs.utils import pad_sequence_power_2, stack_actions, stack_memory


TrainBatch = namedtuple("TrainBatch", [
   "state", "action", "reward", "next_state", "not_done", "q_values"
])


class ParallelTextEnv:

    def __init__(self, text_envs: List[TextEnv], 
                 state_tokenizer: PreTrainedTokenizer, 
                 action_tokenizer: PreTrainedTokenizer):
        
        self.text_envs = text_envs
        self.state_tokenizer = state_tokenizer
        self.action_tokenizer = action_tokenizer
        self.state_embed_length = text_envs[0].state_embed_length
        self.action_embed_length = text_envs[0].action_embed_length

        self.tmp_data = [[] for _ in range(len(self.text_envs))]
        # self.episodes = []

    def reset(self):
        memory = [e.reset() for e in self.text_envs]
        return memory, stack_memory(memory, self.state_tokenizer, max_length=self.state_embed_length)
    
    def rollout(self, n, s_seq, agent, random):

        a_embeds, a_embeds_target = self.get_extra_embeds(agent.critic.action_embed, agent.action_embed_target)
        env_index = list(range(len(self.text_envs)))
        episodes = []
        rewards = []

        s_par = stack_memory(s_seq, self.state_tokenizer, max_length=self.state_embed_length)
        new_state_seq = []

        size = 0

        while size < n:

            a_embeds = self.update_embeds(a_embeds, agent.critic.action_embed)
            a_embeds_target = self.update_embeds(a_embeds_target, agent.action_embed_target)

            a_embeds_pos = [emb["rope"] for emb in a_embeds]
            a_embeds_target_pos = [emb["rope"] for emb in a_embeds_target]
             
            embeds_pt = pad_sequence_power_2(a_embeds_pos, padding_value=0.0, batch_first=True)
            embeds_target_pt = pad_sequence_power_2(a_embeds_target_pos, padding_value=0.0, batch_first=True)
        
            action, _, q_values  = agent.select_action_batch(s_par, embeds_pt, embeds_target_pt, random=random)
            action = action.cpu().numpy().reshape(-1)
            q_values = q_values.cpu().numpy().reshape(-1)
            new_state_seq = []

            for i, si, ai, qi, env in zip(env_index, s_seq, action, q_values, self.text_envs):
                transition = env.step_and_maybe_reset(ai, self.action_tokenizer, agent.critic.action_embed, agent.action_embed_target)
                transition = transition._replace(state=si, q_values=qi)
                self.tmp_data[i].append(transition)
                new_state_seq.append(transition.new_state)
                if transition.done:
                    a_embeds[i], a_embeds_target[i] = transition.embeds
                    episodes.append(self.tmp_data[i])
                    size += len(self.tmp_data[i])
                    self.tmp_data[i] = []
        
            s_seq = new_state_seq
            s_par = stack_memory(s_seq, self.state_tokenizer, max_length=self.state_embed_length)

        s_seq, a_seq, r_seq, s_next_seq, not_dones_seq, q_seq = [], [], [], [], [], [] 
        r_sum = 0.0

        all_episodes = reduce(lambda e1, e2: e1 + e2, episodes)

        for tr in all_episodes:
            s_seq.append(tr.state)
            a_seq.append(tr.action)
            s_next_seq.append(tr.next_state)
            r_seq.append(tr.reward)
            not_dones_seq.append(1 - int(tr.done))
            q_seq.append(tr.q_values)
            
            r_sum += tr.reward
            if tr.done:
                rewards.append(r_sum)
                r_sum = 0.0

        s_stack = stack_memory(s_seq, self.state_tokenizer, max_length=self.state_embed_length)
        next_s_stack = stack_memory(s_next_seq, self.state_tokenizer, max_length=self.state_embed_length)
        a_stack = stack_actions(a_seq, self.action_tokenizer, max_length=self.action_embed_length)

        return new_state_seq, rewards, TrainBatch(
            state=s_stack,
            q_values=torch.FloatTensor(q_seq).to(torch.get_default_device()),
            action=a_stack,
            reward=torch.FloatTensor(r_seq).to(torch.get_default_device()),
            next_state=next_s_stack,
            not_done=torch.IntTensor(not_dones_seq).to(torch.get_default_device()),
        )
    
    @torch.no_grad()
    def get_extra_embeds(self, embedder: nn.Module, embedder_target: nn.Module) -> np.ndarray:

        embeds = []
        embeds_target = []

        for e in self.text_envs:
            e1, e2 = e.get_extra_embeds(self.action_tokenizer, embedder, embedder_target)
            embeds.append(e1)
            embeds_target.append(e2)

        return list(embeds), list(embeds_target)
    
    @torch.no_grad()
    def update_embeds(self, embeds, embedder: nn.Module) -> np.ndarray:

        new_embeds = []
        
        for emb, env in zip(embeds, self.text_envs):
            new_embeds.append(env.update_embeds(emb, embedder))
            
        return new_embeds
