from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch import nn
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from envs.text_env import TextEnv
from envs.utils import TextMemory, custom_pad_sequence, stack_actions, stack_memory


@dataclass
class TrainBatch:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    not_done: torch.Tensor
    q_values: torch.Tensor

@dataclass
class EnvData:
    not_dones: List[int]
    q_values: List[float]
    rewards: List[float]
    states: List[Any]
    actions: List[Any]
    next_states: List[Any]
    reward_sum: float = 0.0


class ParallelTextEnv:

    def __init__(self, 
                 text_envs: List[TextEnv], 
                 state_tokenizer: PreTrainedTokenizer, 
                 action_tokenizer: PreTrainedTokenizer):
        
        self.text_envs = text_envs
        self.state_tokenizer = state_tokenizer
        self.action_tokenizer = action_tokenizer
        self.action_embed_length = text_envs[0].action_embed_length
        self.device = torch.get_default_device()
        

    def reset(self) -> Tuple[List[TextMemory], TextMemory]:
        """
        Reset all environments.
        
        Returns:
            Tuple of (memory list, stacked memory tensor)
        """
        memory = [env.reset() for env in self.text_envs]
        stacked_memory = stack_memory(
            memory, 
            self.state_tokenizer, 
            max_length=self.action_embed_length
        )
        return memory, stacked_memory
    
    
    @torch.no_grad()
    def rollout(self, 
                n: int, 
                cur_s_seq: List[Any], 
                agent: Any, 
                random: bool = False) -> Tuple[List[Any], List[float], TrainBatch]:
        """
        Perform rollout for n steps across all environments.
        
        Args:
            n: Number of steps to rollout
            cur_s_seq: Current state sequence for each environment
            agent: Agent instance for action selection
            random: Whether to use random actions
        
        Returns:
            Tuple of (new_state_seq, rewards, TrainBatch)
        """
    
        a_embeds, a_embeds_target = self.get_extra_embeds(
            agent.critic.action_embed, 
            agent.action_embed_target
        )
        
        env_count = len(self.text_envs)
        env_data = [EnvData([], [], [], [], [], []) for _ in range(env_count)]
        rewards = []
        size = 0
        
        # Precompute stacked states
        s_par = stack_memory(
            cur_s_seq, 
            self.state_tokenizer, 
            max_length=self.action_embed_length
        )

        while size < n + env_count:
            # Update embeddings
            a_embeds = self.update_embeds(a_embeds, agent.critic.action_embed)
            a_embeds_target = self.update_embeds(a_embeds_target, agent.action_embed_target)

            # Prepare embeddings for batch processing
            embeds_pt, embeds_target_pt = self._prepare_embeddings(
                a_embeds, 
                a_embeds_target
            )
        
            # Select actions in batch
            action, _, q_values = agent.select_action_batch(
                s_par, embeds_pt, embeds_target_pt, random=random
            )
            action = action.cpu().numpy().reshape(-1)
            q_values = q_values.cpu().numpy().reshape(-1)
            
            # Process environment steps
            new_state_seq = self._process_env_steps(
                cur_s_seq, action, q_values, agent, env_data, 
                a_embeds, a_embeds_target, rewards
            )
            
            size += env_count
            
            # Update for next iteration
            cur_s_seq = new_state_seq
            s_par = stack_memory(
                cur_s_seq, 
                self.state_tokenizer, 
                max_length=self.action_embed_length
            )

        # Prepare training data
        train_batch = self._prepare_train_batch(env_data)
        
        return cur_s_seq, rewards, train_batch
    

    def _prepare_embeddings(self, 
                           a_embeds: List[Dict], 
                           a_embeds_target: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    
        a_embeds_pos = [emb["rope"] for emb in a_embeds]
        a_embeds_target_pos = [emb["rope"] for emb in a_embeds_target]
        
        embeds_pt = custom_pad_sequence(
            a_embeds_pos, 
            padding_value=0.0, 
            batch_first=True, 
            pad_to_power_2=False
        )
        embeds_target_pt = custom_pad_sequence(
            a_embeds_target_pos, 
            padding_value=0.0, 
            batch_first=True, 
            pad_to_power_2=False
        )
        
        return embeds_pt, embeds_target_pt
    

    def _process_env_steps(self,
                          cur_s_seq: List[Any],
                          actions: np.ndarray,
                          q_values: np.ndarray,
                          agent: Any,
                          env_data: List[EnvData],
                          a_embeds: List[Dict],
                          a_embeds_target: List[Dict],
                          rewards: List[float]) -> List[Any]:
        
        new_state_seq = []
        
        for i, (env, data) in enumerate(zip(self.text_envs, env_data)):
            transition = env.step_and_maybe_reset(
                actions[i], 
                self.action_tokenizer, 
                agent.critic.action_embed, 
                agent.action_embed_target
            )
            
            # Enhance transition with current data
            transition = transition._replace(
                state=cur_s_seq[i], 
                q_values=q_values[i]
            )
            
            # Store data
            data.states.append(transition.state)
            data.actions.append(transition.action)
            data.next_states.append(transition.next_state)
            data.rewards.append(transition.reward)
            data.not_dones.append(1 - int(transition.done))
            data.q_values.append(q_values[i])
            data.reward_sum += transition.reward
            
            new_state_seq.append(transition.new_state)
            
            # Handle episode completion
            if transition.done:
                a_embeds[i], a_embeds_target[i] = transition.embeds
                rewards.append(data.reward_sum)
                data.reward_sum = 0.0
        
        return new_state_seq
    

    def _flatten_env_data(self, env_data: List[EnvData]) -> Dict[str, List]:
        """Flatten environment data into single lists"""
        result = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'not_dones': [],
            'q_values': []
        }
        
        for data in env_data:
            if data.states:
                # For Q update
                result['states'].extend(data.states[:-1])
                result['actions'].extend(data.actions[:-1])
                result['next_states'].extend(data.next_states[:-1])
                # For TD targets
                result['rewards'].append(data.rewards)
                result['not_dones'].append(data.not_dones)
                result['q_values'].append(data.q_values)
        
        return result
    

    def _prepare_train_batch(self, env_data: List[EnvData]) -> TrainBatch:
        """Prepare training batch from collected environment data"""
        # Flatten data from all environments
        all_data = self._flatten_env_data(env_data)
        
        # Stack data for training
        s_stack = stack_memory(
            all_data['states'], 
            self.state_tokenizer, 
            max_length=self.action_embed_length
        )
        next_s_stack = stack_memory(
            all_data['next_states'], 
            self.state_tokenizer, 
            max_length=self.action_embed_length
        )
        a_stack = stack_actions(
            all_data['actions'], 
            self.action_tokenizer, 
            max_length=self.action_embed_length
        )
        
        return TrainBatch(
            state=s_stack,
            q_values=torch.FloatTensor(all_data['q_values']).to(self.device),
            action=a_stack,
            reward=torch.FloatTensor(all_data['rewards']).to(self.device),
            next_state=next_s_stack,
            not_done=torch.IntTensor(all_data['not_dones']).to(self.device),
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
