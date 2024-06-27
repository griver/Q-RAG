from datasets.dataset_dict import DatasetDict, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np
from rl.replay_buffer import TextReplayBuffer
from torch import nn, Tensor
import torch
import os
from collections import namedtuple
from typing import Tuple, Dict, List, Any
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from torch.nn.utils.rnn import pad_sequence


os.environ['TOKENIZERS_PARALLELISM'] = 'true'


Memory = namedtuple("Memory", ["block_ids", "input_ids", "attention_mask", "text"]) 
MemoryBlock = namedtuple("MemoryBlock", ["index", "input_ids", "attention_mask", "text"]) 


class WordsCounterEnv:
    def __init__(self, 
                 dataset: DatasetDict, # load_dataset("AIRI-NLP/quality_counter_new_1024")
                 block_size: int,
                 max_length: int,
                 block_embedding: nn.Module,
                 tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                 subset: str = "train") -> None:
        self.dataset = dataset[subset].with_format("numpy") 
        self.features = ['context', 'word', 'claim', 'label']
        self.index = -1
        self.n = len(self.dataset)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_length = max_length
        self.block_embedding = block_embedding

    def _get_state(self):
        return {
            "memory": self.memory,
            "embeds": self.embeds,
            "blocks_count": self.embeds.shape[0],
            "target": self.label
        }

    def reset(self):
        self.index = (self.index + 1) % self.n
        claim = self.dataset[self.index]["claim"]
        context = self.dataset[self.index]["context"]
        self.label = self.dataset[self.index]["label"]
        word = self.dataset[self.index]["word"]
        tok_seq = self.tokenizer(claim, context, max_length=self.max_length+1, truncation=True)
        # rm the first [CLS] token
        input_ids = np.asarray(tok_seq["input_ids"]).reshape(-1)[1:]
        attention_mask = np.asarray(tok_seq["attention_mask"]).reshape(-1)[1:]
        
        T = input_ids.shape[0]
        pad_size = 0 if T % self.block_size == 0 else (self.block_size - T % self.block_size) 
        self.input_ids = np.pad(input_ids, (0, pad_size), constant_values=(0, int(self.tokenizer.pad_token_id)))
        self.attention_mask = np.pad(attention_mask, (0, pad_size))
        self.T = T + pad_size

        self.blocks = self._split_into_blocks()
        self.embeds = self._embed(self.blocks)

        empty_tok = self.tokenizer("")
        self.memory = Memory(block_ids=[], 
                             input_ids=np.asarray(empty_tok["input_ids"])[1:], 
                             attention_mask=np.asarray(empty_tok["attention_mask"])[1:], 
                             text="")
 
        self.decoded_blocks = [
            self.tokenizer.decode(self.blocks["input_ids"][i]) for i in range(self.embeds.shape[0])  
        ]
        self.word_positions = [i for i, b in enumerate(self.decoded_blocks) if word in b.replace(f'\'{word}\'', '')]

        return self._get_state()

    def _split_into_blocks(self):
        return {
            "input_ids": self.input_ids.reshape(self.T // self.block_size, self.block_size),
            "attention_mask": self.attention_mask.reshape(self.T // self.block_size, self.block_size)
        }
    
    @torch.no_grad()
    def _embed(self, blocks):
        blocks = {
            "input_ids": torch.from_numpy(blocks["input_ids"]).cuda(),
            "attention_mask": torch.from_numpy(blocks["attention_mask"]).cuda()
        }
        embeds = self.block_embedding(blocks).cpu().numpy()
        assert embeds.shape[0] == self.T // self.block_size
        assert len(embeds.shape) == 2
        return embeds
    
    def step(self, action: int) -> Tuple[Dict[str, Any], MemoryBlock, float, bool]:

        assert action < self.embeds.shape[0]
        assert action not in self.memory.block_ids

        memory_block = MemoryBlock(
            index=action, 
            input_ids=self.blocks["input_ids"][action],
            attention_mask=self.blocks["attention_mask"][action],
            text=self.decoded_blocks[action]
        )

        reward = 0.0

        if action in self.word_positions and action not in self.memory.block_ids:
            reward = 1.0

        if len(self.memory.block_ids) == 0:
            prev_input_ids = np.asarray([], dtype=memory_block.input_ids.dtype)
            prev_attention_mask = np.asarray([], dtype=memory_block.attention_mask.dtype)
            prev_text = ""
        else:
            sep_id = np.asarray([self.tokenizer.sep_token_id], dtype=memory_block.input_ids.dtype)
            sep_att = np.asarray([1], dtype=memory_block.attention_mask.dtype)
            
            prev_input_ids = np.concatenate([self.memory.input_ids, sep_id])
            prev_attention_mask = np.concatenate([self.memory.attention_mask, sep_att])
            prev_text = self.memory.text + " " + self.tokenizer.sep_token + " "

        self.memory = Memory(
            block_ids=self.memory.block_ids + [action],
            input_ids=np.concatenate([prev_input_ids, memory_block.input_ids]),
            attention_mask=np.concatenate([prev_attention_mask, memory_block.attention_mask]),
            text=prev_text + memory_block.text
        )

        done = len(self.memory.block_ids) >= self.embeds.shape[0]

        return self._get_state(), memory_block, reward, done


class ReplayAdapter(TextReplayBuffer):

    def __init__(self, max_size, tokenizer: PreTrainedTokenizer):
        super().__init__(max_size)
        self.tokenizer = tokenizer

    def stack_memory(self, memory: List[Memory]):
        input_ids = pad_sequence(
            [torch.from_numpy(si.input_ids) for si in memory], 
            batch_first=True, 
            padding_value=int(self.tokenizer.pad_token_id))
        
        attention_mask = pad_sequence(
            [torch.from_numpy(si.attention_mask) for si in memory], 
            batch_first=True, 
            padding_value=0)
        
        s_memory = Memory(
            block_ids=[si.block_ids for si in memory],
            input_ids=input_ids,
            attention_mask=attention_mask,
            text=[si.text for si in memory]
        )
        
        return s_memory
    

    def stack_actions(self, actions: List[MemoryBlock]):
        input_ids = pad_sequence(
            [torch.from_numpy(si.input_ids) for si in actions], 
            batch_first=True, 
            padding_value=int(self.tokenizer.pad_token_id))
        
        attention_mask = pad_sequence(
            [torch.from_numpy(si.attention_mask) for si in actions], 
            batch_first=True, 
            padding_value=0)
        
        a_blocck = MemoryBlock(
            index=[si.index for si in actions],
            input_ids=input_ids,
            attention_mask=attention_mask,
            text=[si.text for si in actions]
        )
        
        return a_blocck

    def sample(self, batch_size):
        s, a, r, next_s, not_done = super().sample(batch_size)

        s_memory = self.stack_memory([si["memory"] for si in s])
        next_s_memory = self.stack_memory([si["memory"] for si in next_s])
        embeds = torch.from_numpy(np.stack([si["embeds"] for si in s]))
        a_blocks = self.stack_actions(a)
        
        return s_memory, a_blocks, next_s_memory, embeds, torch.from_numpy(r), torch.from_numpy(not_done)