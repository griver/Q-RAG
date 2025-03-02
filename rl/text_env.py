import numpy as np
from torch import nn, Tensor
import torch
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import os
from torch.nn.utils.rnn import pad_sequence
from rl.replay_buffer import ReplayBuffer


os.environ['TOKENIZERS_PARALLELISM'] = 'true'

TextMemory = namedtuple("TextMemory", ["item_ids", "available_ids", "available_mask", "input_ids", "attention_mask", "text", "embeds"]) 
TextMemoryItem = namedtuple("TextMemoryItem", ["index", "input_ids", "attention_mask", "text"]) 


def pad_sequence_power_2(seq_list: List[Tensor], padding_value, batch_first=True):
    max_len = max(map(len, seq_list))
    max_len_2 = 2 ** int(np.ceil(np.log2(max_len)))
    pad_1 = pad_sequence(seq_list, batch_first=batch_first, padding_value=padding_value)
    pad_2 = torch.nn.functional.pad(pad_1, [0, 0] * (len(pad_1.shape)-2) + [0, max_len_2 - max_len] + [0, 0], value=padding_value)
    assert pad_2.shape[1] == max_len_2
    return pad_2


class TextEnv:

    separator = " [SEP] "
    embedder: nn.Module
    embed_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    max_embed_length = 500
    action_embed_length = 64
    max_batch_size = 128

    def tokenize(self, text: str, max_len=None) -> np.ndarray:
        if max_len is None:
            max_len = self.max_embed_length
        tokens = self.embed_tokenizer(text, truncation=True, max_length=max_len)
        return {k: np.asarray(v) for k, v in tokens.items()}
    
    def tokenize_list(self, text_array: List[str]) -> np.ndarray:
        tokens = self.embed_tokenizer(text_array, truncation=True, max_length=self.max_embed_length)
        return [{k: np.asarray(v[i]) for k, v in tokens.items()} for i in range(len(text_array))]

    @torch.no_grad()
    def get_embeds(self, sentences: List[str]) -> np.ndarray:
        batch = self.embed_tokenizer(
            list(sentences), 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_embed_length
        ).to(torch.get_default_device())

        B = batch["input_ids"].shape[0]
        assert batch["input_ids"].shape[1] <= 500
        embeds = []
        for i in range(0, B, self.max_batch_size):
            subbatch = {k:v[i:i+self.max_batch_size] for k, v in batch.items()}
            embeds.append(self.embedder(**subbatch).to("cpu"))

        return torch.cat(embeds, dim=0).numpy()
    
    @torch.no_grad()
    def get_extra_embeds(self, embedder: nn.Module) -> np.ndarray:
        batch = self.embed_tokenizer(
            list(self.all_texts), 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_embed_length
        ).to(torch.get_default_device())

        return embedder(**batch)


    def reset(self, question: str, text_array: List[str]) -> TextMemory:
        self.all_texts = text_array

        tokens = self.tokenize(question, self.action_embed_length)
        # self.action_tokens = self.tokenize_list(text_array)

        self.memory = TextMemory(
            item_ids=[], 
            available_ids=set(range(len(self.all_texts))), 
            available_mask=np.ones(len(self.all_texts), dtype=bool),
            text=question,
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            embeds=self.get_embeds(self.all_texts)
        )

        return self.memory 
    
    def step(self, action: int) -> Tuple[TextMemory, TextMemoryItem, float]:

        assert action < len(self.all_texts)

        action_text = self.all_texts[action]
        action_tokens = self.tokenize(action_text, self.action_embed_length)
        # action_tokens = self.action_tokens[action]
        is_empty = len(self.memory.text) == 0

        new_text = action_text if is_empty else self.memory.text + self.separator + action_text
        
        if is_empty: 
            input_ids = action_tokens["input_ids"].copy()
            attention_mask = action_tokens["attention_mask"].copy()
        else:
            input_ids = np.concatenate(
                [self.memory.input_ids, np.asarray([self.embed_tokenizer.sep_token_id]), action_tokens["input_ids"]]
            )
            attention_mask = np.concatenate(
                [self.memory.attention_mask, np.ones(1, dtype=np.int32), action_tokens["attention_mask"]]
            )

        available_mask = self.memory.available_mask.copy()
        available_mask[action] = False

        self.memory = TextMemory(
            item_ids=self.memory.item_ids + [action],
            available_ids=self.memory.available_ids - {action},
            available_mask=available_mask,
            text=new_text,
            input_ids=input_ids[:self.max_embed_length],
            attention_mask=attention_mask[:self.max_embed_length],
            embeds=self.memory.embeds
        )

        memory_item = TextMemoryItem(
            index=action, 
            input_ids=action_tokens["input_ids"],
            attention_mask=action_tokens["attention_mask"],
            text=action_text
        )

        done = len(self.memory.item_ids) >= len(self.all_texts)

        return self.memory, memory_item, done
    

class TextReplayBuffer(ReplayBuffer):

    def __init__(self, max_size, tokenizer: PreTrainedTokenizer):
        super().__init__(max_size)
        self.tokenizer = tokenizer

    def stack_memory(self, memory: List[TextMemory]):
        input_ids = pad_sequence_power_2(
            [torch.from_numpy(si.input_ids) for si in memory], 
            batch_first=True, 
            padding_value=int(self.tokenizer.pad_token_id))
        
        assert input_ids.shape[0] == len(memory)
        
        attention_mask = pad_sequence_power_2(
            [torch.from_numpy(si.attention_mask) for si in memory], 
            batch_first=True, 
            padding_value=0)
        
        embeds = pad_sequence_power_2(
            [torch.from_numpy(si.embeds) for si in memory], 
            padding_value=0.0,
            batch_first=True)
        
        available_mask = pad_sequence_power_2(
            [torch.from_numpy(si.available_mask) for si in memory], 
            batch_first=True, padding_value=False)
        
        s_memory = TextMemory(
            item_ids=[si.item_ids for si in memory],
            available_ids=[si.available_ids for si in memory],
            available_mask=available_mask.to(torch.get_default_device()),
            input_ids=input_ids.to(torch.get_default_device()),
            attention_mask=attention_mask.to(torch.get_default_device()),
            text=[si.text for si in memory],
            embeds=embeds.to(torch.get_default_device()).type(torch.float32)
        )
        
        return s_memory
    

    def stack_actions(self, actions: List[TextMemoryItem]):
        input_ids = pad_sequence_power_2(
            [torch.from_numpy(si.input_ids) for si in actions], 
            batch_first=True, 
            padding_value=int(self.tokenizer.pad_token_id))
        
        attention_mask = pad_sequence_power_2(
            [torch.from_numpy(si.attention_mask) for si in actions], 
            batch_first=True, 
            padding_value=0)
        
        a_blocck = TextMemoryItem(
            index=[si.index for si in actions],
            input_ids=input_ids.to(torch.get_default_device()),
            attention_mask=attention_mask.to(torch.get_default_device()),
            text=[si.text for si in actions]
        )
        
        return a_blocck

    @torch.no_grad()
    def sample(self, batch_size):
        s, a, r, next_s, not_done, entropy = super().sample(batch_size)

        s_stack = self.stack_memory(s)
        next_s_stack = self.stack_memory(next_s)
        a_stack = self.stack_actions(a)
        
        return s_stack, a_stack, next_s_stack, torch.FloatTensor(r).to(torch.get_default_device()), torch.FloatTensor(not_done).to(torch.get_default_device()), torch.FloatTensor(entropy).to(torch.get_default_device())    
