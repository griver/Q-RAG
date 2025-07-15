from abc import abstractmethod

import numpy as np
from torch import nn, Tensor
import torch
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
from functools import reduce


TextMemory = namedtuple("TextMemory", ["item_ids", "available_ids", "available_mask", "input_ids", "attention_mask", "text"]) 
TextMemoryItem = namedtuple("TextMemoryItem", ["index", "position", "input_ids", "attention_mask", "text"]) 
Transition = namedtuple("Transition", [
   "state", "action", "reward", "next_state", "done", "new_state", "embeds", "q_values"
])


@torch.no_grad()
def custom_pad_sequence(seq_list: List[Tensor], padding_value, batch_first=True, padding_side='right', pad_to_power_2=True):
    max_len = max(map(len, seq_list))
    max_len_2 = 2 ** int(np.ceil(np.log2(max_len)))
    pad_1 = pad_sequence(seq_list, batch_first=batch_first, padding_side=padding_side, padding_value=padding_value)

    if not pad_to_power_2:
        return pad_1

    if padding_side == 'right':
        seq_padding = [0, max_len_2 - max_len]
    elif padding_side == 'left':
        seq_padding = [max_len_2 - max_len, 0]
    else:
        raise ValueError('padding_side must be either "left" or "right"')

    pad_2 = torch.nn.functional.pad(pad_1, [0, 0] * (len(pad_1.shape)-2) + seq_padding + [0, 0], value=padding_value)
    assert pad_2.shape[1] == max_len_2
    return pad_2


def stack_text_list(text_array: List[str], tokenizer: PreTrainedTokenizer, max_length: int, device=None):

    if device is None:
        device = torch.get_default_device()

    tokens = tokenizer(text_array, truncation=True, max_length=max_length)
    
    input_ids = custom_pad_sequence(
        [torch.IntTensor(ii) for ii in tokens["input_ids"]], 
        batch_first=True, 
        padding_value=int(tokenizer.pad_token_id),
        padding_side=tokenizer.padding_side
    )
    
    attention_mask = custom_pad_sequence(
        [torch.IntTensor(am) for am in tokens["attention_mask"]], 
        batch_first=True, 
        padding_value=0,
        padding_side=tokenizer.padding_side
    )
    
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


def stack_memory(memory: List[TextMemory], tokenizer: PreTrainedTokenizer, max_length: int, device=None):
    
    if device is None:
        device = torch.get_default_device()

    # text_array = [s.text for s in memory]
    # tokens = tokenizer(text_array, truncation=True, max_length=max_length)
    text_array = []
    lens = []
 
    for m in memory:
        mt = m.text.split("[SEP]") 
        text_array.extend(mt)
        lens.append(len(mt))

    tokens_split = tokenizer(text_array, truncation=True, max_length=max_length)
    tokens = {
       "input_ids": [],
       "attention_mask": [] 
    }

    sep_token = tokenizer.sep_token_id
    if sep_token is None:
        sep_token = tokenizer.eos_token_id

    i1 = 0
    for l in lens:

        if l > 1:
            tokens["input_ids"].append(reduce(
                lambda s1, s2: s1[:-1] + [sep_token] + s2[1:], 
                tokens_split["input_ids"][i1:i1+l]
            ))
            tokens["attention_mask"].append(reduce(
                lambda s1, s2: s1[:-1] + [1] + s2[1:], 
                tokens_split["attention_mask"][i1:i1+l]
            ))
        else:
            tokens["input_ids"].append(tokens_split["input_ids"][i1])
            tokens["attention_mask"].append(tokens_split["attention_mask"][i1])

        i1 = i1 + l

    input_ids = custom_pad_sequence(
        [torch.IntTensor(ii) for ii in tokens["input_ids"]],
        batch_first=True,
        padding_value=int(tokenizer.pad_token_id),
        padding_side=tokenizer.padding_side
    )

    assert input_ids.shape[0] == len(memory)
    
    attention_mask = custom_pad_sequence(
        [torch.IntTensor(am) for am in tokens["attention_mask"]], 
        batch_first=True, 
        padding_value=0,
        padding_side=tokenizer.padding_side
    )
    
    available_mask = custom_pad_sequence(
        [torch.from_numpy(si.available_mask) for si in memory], 
        batch_first=True, padding_value=False,
        # this padding is different as available_mask has nothing to do with actual texts (it marks chunks available for retrieval)
        padding_side='right', pad_to_power_2=False
    )
    
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
    
    input_ids = custom_pad_sequence(
        [torch.IntTensor(ii) for ii in tokens["input_ids"]], 
        batch_first=True, 
        padding_value=int(tokenizer.pad_token_id),
        padding_side=tokenizer.padding_side
    )
    
    attention_mask = custom_pad_sequence(
        [torch.IntTensor(am) for am in tokens["attention_mask"]], 
        batch_first=True, 
        padding_value=0,
        padding_side=tokenizer.padding_side
    )
    
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