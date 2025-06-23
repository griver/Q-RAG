from abc import ABC, abstractmethod
import numpy as np
from torch import nn, Tensor
import torch
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import os
from torch.nn.utils.rnn import pad_sequence
from sortedcontainers import SortedList
from envs.utils import Transition, stack_text_list, TextMemory, TextMemoryItem, torch_cat_dict


class PositionProcessor(ABC):
    
    @abstractmethod
    def initialize_positions(self, num_texts: int) -> List[Union[int, float]]:
        pass
    
    @abstractmethod
    def update_positions(self, selected_indices: List[int], positions: List[int]) -> List[Union[int, float]]:
        pass


class AbsolutePositionProcessor(PositionProcessor):
    """Processor for absolute positioning (0, 1, 2, ...)."""
    
    def initialize_positions(self, num_texts: int) -> List[int]:
        return list(range(num_texts))
    
    def update_positions(self, selected_indices: List[int], positions: List[int]) -> List[int]:
        return positions


class RelativePositionProcessor(PositionProcessor):
    """Processor for relative positioning."""

    def __init__(self, step_size: int = 10):
        self.step_size = step_size
        self.max_value = int(0.9 * step_size)
    
    def initialize_positions(self, num_texts: int) -> List[float]:
        return np.linspace(0, self.max_value, num_texts).tolist()
        
    def update_positions(self, selected_indices: List[int], positions: List[int]) -> List[float]:

        n = len(positions)
        result = np.zeros(n, dtype=np.float32)
        sorted_indices = sorted(selected_indices) + [n]
        current_value = 0
        start = 0
        
        for end in sorted_indices:
            if end > start:
                # result[start:end] = np.sqrt(np.linspace(0, self.max_value ** 2, end - start)) + current_value
                result[start:end] = np.linspace(0, self.max_value, end - start) + current_value
                start = end
            current_value += self.step_size
        
        return result
    

class RandomPositionProcessor(PositionProcessor):
    """Processor for random positioning with maximum indent."""
    
    def __init__(self, max_chunks_count: int = 1000):
        self.max_chunks_count = max_chunks_count

    def get_randomized_idx(self, n, max_indent=1000, num_splits=5):
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
    
    def initialize_positions(self, num_texts: int) -> List[int]:
        if self.max_chunks_count - num_texts < 0:
            raise ValueError(
                f'num_chunks ({num_texts}) > max_chunks_count ({self.max_chunks_count}). '
                'Increase max_chunks_count!'
            )
        return self.get_randomized_idx(num_texts, max_indent=self.max_chunks_count - num_texts)
    
    def update_positions(self, selected_indices: List[int], positions: List[int]) -> List[int]:
        return positions
    

class TextEnv:

    separator: str
    action_embed_length: int
    max_embedding_batch: int
    positions_processor: PositionProcessor
    sort_by_index: bool = True

    @torch.no_grad()
    def get_extra_embeds(self, tokenizer, embedder: nn.Module, embedder_target: nn.Module) -> np.ndarray:
        
        batch = stack_text_list(list(self.all_texts), tokenizer, self.action_embed_length)
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

            embeds = torch_cat_dict(embeds, dim=0)
            embeds_target = torch_cat_dict(embeds_target, dim=0)

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
        # self.items_dict = SortedList(key=lambda memory_item: memory_item.index)
        self.items_list = []

        self.positions = self.positions_processor.initialize_positions(len(text_array))

        self.memory = TextMemory(
            item_ids=[], 
            available_ids=set(range(len(self.all_texts))), 
            available_mask=np.ones(len(self.all_texts), dtype=bool),
            text=question,
            input_ids=None,
            attention_mask=None
        )

        return self.memory 
    
    def _step(self, action: int) -> Tuple[TextMemory, TextMemoryItem, bool]:

        assert action < len(self.all_texts)

        action_text = self.all_texts[action]
        
        memory_item = TextMemoryItem(
            index=action, 
            position=self.positions[action],
            input_ids=None,
            attention_mask=None,
            text=action_text
        )

        self.items_list.append(memory_item)
        maybe_sorted = sorted(self.items_list, key=lambda memory_item: memory_item.index) if self.sort_by_index else self.items_list
        new_text = self.separator.join([self.question] + [it.text for it in maybe_sorted])

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

        self.positions = self.positions_processor.update_positions([it.index for it in self.items_list], self.positions)
        
        return self.memory, memory_item, done

