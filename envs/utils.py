from abc import abstractmethod
import numpy as np
from torch import nn, Tensor
import torch
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence


TextMemory = namedtuple("TextMemory", ["item_ids", "available_ids", "available_mask", "input_ids", "attention_mask", "text"]) 
TextMemoryItem = namedtuple("TextMemoryItem", ["index", "position", "input_ids", "attention_mask", "text"]) 
Transition = namedtuple("Transition", [
   "state", "action", "reward", "next_state", "done", "new_state", "embeds", "q_values"
])


@torch.no_grad()
def pad_sequence_power_2(seq_list: List[Tensor], padding_value, batch_first=True):
    max_len = max(map(len, seq_list))
    max_len_2 = 2 ** int(np.ceil(np.log2(max_len)))
    pad_1 = pad_sequence(seq_list, batch_first=batch_first, padding_value=padding_value)
    pad_2 = torch.nn.functional.pad(pad_1, [0, 0] * (len(pad_1.shape)-2) + [0, max_len_2 - max_len] + [0, 0], value=padding_value)
    assert pad_2.shape[1] == max_len_2
    return pad_2


def stack_text_list(text_array: List[str], tokenizer: PreTrainedTokenizer, max_length: int, device=None):

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


def stack_memory(memory: List[TextMemory], tokenizer: PreTrainedTokenizer, max_length: int, device=None):

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
    

def stack_actions(actions: List[TextMemoryItem], tokenizer, max_length: int, device=None):

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


def torch_cat_dict(
    dicts: List[Dict[str, torch.Tensor]], 
    dim: int = 0
) -> Dict[str, torch.Tensor]:
    
    if not dicts:
        return {}
    
    # Check that all dictionaries have the same keys
    keys = dicts[0].keys()
    for d in dicts[1:]:
        if d.keys() != keys:
            raise ValueError("All dictionaries must have the same keys")
    
    return {
        key: torch.cat([d[key] for d in dicts], dim=dim)
        for key in keys
    }