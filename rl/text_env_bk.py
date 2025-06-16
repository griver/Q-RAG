from abc import abstractmethod
import numpy as np
from torch import nn, Tensor
import torch
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import os
from torch.nn.utils.rnn import pad_sequence
from rl.replay_buffer import ReplayBuffer
from sortedcontainers import SortedList
from functools import reduce


os.environ['TOKENIZERS_PARALLELISM'] = 'true'
MAX_TOKEN_LENGTH = {}

TextMemory = namedtuple("TextMemory", ["item_ids", "available_ids", "available_mask", "input_ids", "attention_mask", "text"]) 
TextMemoryItem = namedtuple("TextMemoryItem", ["index", "position", "input_ids", "attention_mask", "text"]) 
Transition = namedtuple("Transition", [
   "state", "action", "reward", "next_state", "done", "new_state", "embeds", "q_values"
])
TrainBatch = namedtuple("TrainBatch", [
   "state", "action", "reward", "next_state", "not_done", "q_values"
])


@torch.no_grad()
def pad_sequence_power_2(seq_list: List[Tensor], padding_value, batch_first=True):
    max_len = max(map(len, seq_list))
    max_len_2 = 2 ** int(np.ceil(np.log2(max_len)))
    # seq_list = [t.clone() for t in seq_list]
    pad_1 = pad_sequence(seq_list, batch_first=batch_first, padding_value=padding_value)
    pad_2 = torch.nn.functional.pad(pad_1, [0, 0] * (len(pad_1.shape)-2) + [0, max_len_2 - max_len] + [0, 0], value=padding_value)
    assert pad_2.shape[1] == max_len_2
    return pad_2


def stack_memory(memory: List[TextMemory], tokenizer: PreTrainedTokenizer, max_length=None, device=None):

    if not max_length:
        max_length = MAX_TOKEN_LENGTH["state"]

    if device is None:
        device = torch.get_default_device()
    text_array = [s.text for s in memory]

    tokens = tokenizer(text_array, truncation=True, max_length=max_length)
    
    input_ids = pad_sequence_power_2(
        [torch.IntTensor(ii) for ii in tokens["input_ids"]], 
        batch_first=True, 
        padding_value=int(tokenizer.pad_token_id))
    
    assert input_ids.shape[0] == len(memory)
    
    attention_mask = pad_sequence_power_2(
        [torch.IntTensor(am) for am in tokens["attention_mask"]], 
        batch_first=True, 
        padding_value=0)
    
    available_mask = pad_sequence_power_2(
        [torch.from_numpy(si.available_mask) for si in memory], 
        batch_first=True, padding_value=False)
    
    s_memory = TextMemory(
        item_ids=[si.item_ids for si in memory],
        available_ids=[si.available_ids for si in memory],
        available_mask=available_mask.to(device),
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        text=text_array,
    )
    
    return s_memory
    

def stack_actions(actions: List[TextMemoryItem], tokenizer, max_length=None, device=None):

    if not max_length:
        max_length = MAX_TOKEN_LENGTH["action"]

    if device is None:
        device = torch.get_default_device()

    text_array = [s.text for s in actions]

    tokens = tokenizer(text_array, truncation=True, max_length=max_length)
    
    input_ids = pad_sequence_power_2(
        [torch.IntTensor(ii) for ii in tokens["input_ids"]], 
        batch_first=True, 
        padding_value=int(tokenizer.pad_token_id))
    
    attention_mask = pad_sequence_power_2(
        [torch.IntTensor(am) for am in tokens["attention_mask"]], 
        batch_first=True, 
        padding_value=0)
    
    a_block = TextMemoryItem(
        index=[si.index for si in actions],
        position=[si.position for si in actions],
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        text=[si.text for si in actions]
    )
    
    return a_block


def stack_text_list(text_array: List[str], tokenizer, max_length=None, device=None):

    if not max_length:
        max_length = MAX_TOKEN_LENGTH["action"]

    if device is None:
        device = torch.get_default_device()

    tokens = tokenizer(text_array, truncation=True, max_length=max_length)
    
    input_ids = pad_sequence_power_2(
        [torch.IntTensor(ii) for ii in tokens["input_ids"]], 
        batch_first=True, 
        padding_value=int(tokenizer.pad_token_id))
    
    attention_mask = pad_sequence_power_2(
        [torch.IntTensor(am) for am in tokens["attention_mask"]], 
        batch_first=True, 
        padding_value=0)
    
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


def get_randomized_idx(n, max_indent=1000, num_splits=5):
    "returns indices in range from 0 to max_indent+n"
    if max_indent <= 0:
        indents = [0]*num_splits
    else:
        indents = sorted(np.random.choice(max_indent, size=num_splits))
    
    idx = np.arange(n)
    split_id = sorted(np.random.choice(idx[1:n-1], size=num_splits-1, replace=False))
    splits  = np.split(idx, split_id)
    result = [e+indent for split, indent in zip(splits, indents) for e in split]
    return sorted(result) 


def relative_positions(indices, n):
    
    result = np.zeros(n, dtype=np.float32)
    sorted_indices = sorted(indices) + [n]
    current_value = 0
    start = 0
    
    for end in sorted_indices:
        if end > start:
            result[start:end] = np.sqrt(np.linspace(0, 81, end - start)) + current_value
            start = end
        current_value += 10.0
    
    return result


class TextEnv:

    separator = " [SEP] "
    # max_embed_length = MAX_TOKEN_LENGTH["state"]
    # action_embed_length = MAX_TOKEN_LENGTH["action"]
    max_embedding_batch = 500
    max_chunks_count = 1000
    index_type = "relative" # "absolute", "relative"

    @torch.no_grad()
    def get_extra_embeds(self, tokenizer, embedder: nn.Module, embedder_target: nn.Module) -> np.ndarray:
        
        batch = stack_text_list(list(self.all_texts), tokenizer)
        positions = torch.tensor(self.positions, device=torch.get_default_device())

        max_B = self.max_embedding_batch
        B, D = batch['input_ids'].shape #will brake if num dimenstions is not equal to 2. this is desirable behavior
        if B <= max_B:
            embeds = embedder(**batch, positions=positions)
            embeds_target = embedder_target(**batch, positions=positions)
        else:
            assert B == len(positions)
            embeds, embeds_target = [], []
            for i in range(0, B, max_B):
                inputs = (batch['input_ids'][i:i + max_B], batch['attention_mask'][i:i+max_B], positions[i:i+max_B])
                embeds.append( embedder(*inputs) )
                embeds_target.append( embedder_target(*inputs) )

            embeds = torch.cat(embeds, dim=0)
            embeds_target = torch.cat(embeds_target, dim=0)

        return embeds, embeds_target
    
    @torch.no_grad()
    def update_embeds(self, embeds, embedder: nn.Module):
        positions = torch.tensor(self.positions, device=torch.get_default_device())
        embeds = embedder.update_pos(embeds, positions=positions)
        return embeds
    
    @abstractmethod
    def reset(self) -> TextMemory: pass
    @abstractmethod
    def step(self, action: int) -> Tuple[TextMemory, TextMemoryItem, float, bool]: pass

    def step_and_maybe_reset(self, action: int, tokenizer, embedder: nn.Module, embedder_target: nn.Module):
        s_next, a, r, done = self.step(action)
        new_state = s_next
        embeds = None
        if done:
            new_state = self.reset()
            embeds = self.get_extra_embeds(tokenizer, embedder, embedder_target)
        
        return Transition(
            state=None,
            action=a,
            reward=r,
            done=done,
            next_state=s_next,
            new_state=new_state,
            embeds=embeds,
            q_values=None
        ) 

    def _reset(self, question: str, text_array: List[str]) -> TextMemory:
        self.all_texts = text_array
        self.question = question
        self.items_dict = SortedList(key=lambda memory_item: memory_item.index)

        if self.index_type == "random":
            assert self.max_chunks_count - len(text_array) >= 0, f'num_chunks(len(text_array)) > envs.max_chunks_count. Increase envs.max_chunks_count!' 
            self.positions = get_randomized_idx(len(text_array), max_indent=self.max_chunks_count - len(text_array))
        elif self.index_type == "absolute":
            self.positions = list(range(len(text_array)))
        elif self.index_type == "relative":
            # self.positions = np.ones(len(text_array)).tolist()
            self.positions = np.linspace(0, 9, len(text_array))

        # tokens = self.tokenize(question, self.action_embed_length)
        # self.action_tokens = self.tokenize_list(text_array)

        self.memory = TextMemory(
            item_ids=[], 
            available_ids=set(range(len(self.all_texts))), 
            available_mask=np.ones(len(self.all_texts), dtype=bool),
            text=question,
            input_ids=None,
            attention_mask=None,
            # embeds=self.get_embeds(self.all_texts)
        )

        return self.memory 
    
    def _step(self, action: int) -> Tuple[TextMemory, TextMemoryItem, bool]:

        assert action < len(self.all_texts)

        action_text = self.all_texts[action]
        # action_tokens = self.tokenize(action_text, self.action_embed_length)
        # action_tokens = self.action_tokens[action]

        memory_item = TextMemoryItem(
            index=action, 
            position=self.positions[action],
            input_ids=None,
            attention_mask=None,
            text=action_text
        )

        self.items_dict.add(memory_item)
        new_text = self.separator.join([self.question] + [it.text for it in self.items_dict])

        available_mask = self.memory.available_mask.copy()
        available_mask[action] = False

        self.memory = TextMemory(
            item_ids=self.memory.item_ids + [action],
            available_ids=self.memory.available_ids - {action},
            available_mask=available_mask,
            text=new_text,
            input_ids=None,
            attention_mask=None
        )

        done = len(self.memory.item_ids) >= len(self.all_texts)

        if self.index_type == "relative":
            self.positions = relative_positions([it.index for it in self.items_dict], len(self.positions))
            # print([it.index for it in self.items_dict], self.positions)

        return self.memory, memory_item, done
    

class ParallelTextEnv:

    def __init__(self, text_envs: List[TextEnv], 
                 state_tokenizer: PreTrainedTokenizer, 
                 action_tokenizer: PreTrainedTokenizer):
        
        self.text_envs = text_envs
        self.state_tokenizer = state_tokenizer
        self.action_tokenizer = action_tokenizer

        self.tmp_data = [[] for _ in range(len(self.text_envs))]
        # self.episodes = []

    def reset(self):
        memory = [e.reset() for e in self.text_envs]
        return memory, stack_memory(memory, self.state_tokenizer)
    
    def step(self, actions: np.ndarray) -> Tuple[TextMemory, TextMemoryItem, np.ndarray]:
        
        s_seq, a_seq, r_seq, done_seq = [], [], [], []
    
        for a, e in zip(actions, self.text_envs):
            s_next, a_data, r, done = e.step(a)
            s_seq.append(s_next)
            a_seq.append(a_data)
            r_seq.append(r)
            done_seq.append(done)

        s_batch = stack_memory(s_seq, self.state_tokenizer)
        
        return s_seq, a_seq, r_seq, done_seq, s_batch

    def step_and_maybe_reset(self, actions: np.ndarray, embedder: nn.Module, embedder_target: nn.Module) -> Tuple[TextMemory, TextMemoryItem, np.ndarray]:
        
        transitions = []
    
        for a, e in zip(actions, self.text_envs):
            transitions.append(e.step_and_maybe_reset(a, embedder, embedder_target))

        return Transition(
            state=None,
            q_values=None,
            action=[t.action for t in transitions],
            reward=[t.reward for t in transitions],
            next_state=[t.next_state for t in transitions],
            new_state=[t.new_state for t in transitions],
            done=[t.done for t in transitions],
            embeds=[t.embeds for t in transitions]
        )
    
    def rollout(self, n, s_seq, agent, random):

        a_embeds, a_embeds_target = self.get_extra_embeds(agent.critic.action_embed, agent.action_embed_target)
        env_index = list(range(len(self.text_envs)))
        episodes = []
        rewards = []

        s_par = stack_memory(s_seq, self.state_tokenizer)
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
                    # if len(self.episodes) > 20:
                    #     self.episodes = self.episodes[1:]
        
            s_seq = new_state_seq
            s_par = stack_memory(s_seq, self.state_tokenizer)

        s_seq, a_seq, r_seq, s_next_seq, not_dones_seq, q_seq = [], [], [], [], [], [] 
        r_sum = 0.0

        all_episodes = reduce(lambda e1, e2: e1 + e2, episodes)
        # offset = np.random.randint(0, max(len(all_episodes) - n, 1))

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

        s_stack = stack_memory(s_seq, self.state_tokenizer)
        next_s_stack = stack_memory(s_next_seq, self.state_tokenizer)
        a_stack = stack_actions(a_seq, self.action_tokenizer)

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

        # all_texts = reduce(lambda e1, e2: e1 + e2, [list(env.all_texts) for env in self.text_envs])
        # lens = [len(env.all_texts) for env in self.text_envs]

        # batch = stack_text_list(all_texts, self.action_tokenizer)
        # all_embeds, all_embeds_target = embedder(**batch), embedder_target(**batch)
        # embeds, embeds_target = torch.split(all_embeds, lens), torch.split(all_embeds_target, lens)

        embeds = []
        embeds_target = []

        for e in self.text_envs:
            e1, e2 = e.get_extra_embeds(self.action_tokenizer, embedder, embedder_target)
            embeds.append(e1)
            embeds_target.append(e2)

        # embeds = pad_sequence_power_2(embeds, padding_value=0.0, batch_first=True)
        # embeds_target = pad_sequence_power_2(embeds_target, padding_value=0.0, batch_first=True)

        return list(embeds), list(embeds_target)
    
    @torch.no_grad()
    def update_embeds(self, embeds, embedder: nn.Module) -> np.ndarray:

        new_embeds = []
        
        for emb, env in zip(embeds, self.text_envs):
            new_embeds.append(env.update_embeds(emb, embedder))
            
        return new_embeds


class TextReplayBuffer(ReplayBuffer):

    def __init__(self, max_size, tokenizer: PreTrainedTokenizer):
        super().__init__(max_size)
        self.tokenizer = tokenizer

    
    @torch.no_grad()
    def ordered_sample(self, batch_size):
        s, a, r, next_s, next_a, not_done, q_values = super().ordered_sample(batch_size)

        s_stack = stack_memory(s, self.tokenizer)
        next_s_stack = stack_memory(next_s, self.tokenizer)
        a_stack = stack_actions(a, self.tokenizer)
        next_a_stack = stack_actions(next_a, self.tokenizer)
        
        return s_stack, a_stack, next_s_stack, next_a_stack, \
            torch.FloatTensor(r).to(torch.get_default_device()), \
            torch.FloatTensor(not_done).to(torch.get_default_device()), \
            torch.FloatTensor(q_values).to(torch.get_default_device())    


    @torch.no_grad()
    def sample(self, batch_size):
        s, a, r, next_s, next_a, not_done = super().sample(batch_size)

        s_stack = stack_memory(s, self.tokenizer)
        next_s_stack = stack_memory(next_s, self.tokenizer)
        a_stack = stack_actions(a, self.tokenizer)
        next_a_stack = stack_actions(next_a, self.tokenizer)
        
        return s_stack, a_stack, next_s_stack, next_a_stack, torch.FloatTensor(r).to(torch.get_default_device()), torch.FloatTensor(not_done).to(torch.get_default_device())  
