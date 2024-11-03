import numpy as np
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import os
from rl.replay_buffer import ReplayBuffer
from flax import nnx
from jax import numpy as jnp
import jax


os.environ['TOKENIZERS_PARALLELISM'] = 'true'

TextMemory = namedtuple("TextMemory", ["item_ids", "available_ids", "available_mask", "input_ids", "attention_mask", "text", "embeds"]) 
TextMemoryItem = namedtuple("TextMemoryItem", ["index", "input_ids", "attention_mask", "text"]) 


def pad_sequence_power_2(seq_list: List[np.ndarray], padding_value):
    seq_list = list(map(np.asarray, seq_list))
    max_len = max(map(len, seq_list))
    max_len_2 = 2 ** int(np.ceil(np.log2(max_len)))
    return np.stack([
        np.pad(seq, [(0, max_len_2 - len(seq))] + [(0, 0)] * (len(seq.shape)-1), constant_values=padding_value) for seq in seq_list
    ])


@nnx.jit
def fast_embed(model, input_ids, attention_mask):
    return model(input_ids, attention_mask)


def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)


class TextEnv:

    separator = " [SEP] "
    embedder: nnx.Module
    embed_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    max_embed_length = 256
    max_batch_size = 256

    def tokenize(self, text: str, max_len = None) -> np.ndarray:
        max_len = self.max_embed_length if max_len is None else max_len
        tokens = self.embed_tokenizer(text, truncation=True, max_length=max_len)
        len_2 = 2 ** int(np.ceil(np.log2(len(tokens["input_ids"]))))
        tokens = {
            "input_ids": np.pad(tokens["input_ids"], (0, len_2 - len(tokens["input_ids"])), constant_values=int(self.embed_tokenizer.pad_token_id)),
            "attention_mask": np.pad(tokens["attention_mask"], (0, len_2 - len(tokens["input_ids"])), constant_values=0)
        }
        return {k: np.asarray(v) for k, v in tokens.items()}
    
    def tokenize_list(self, text_array: List[str]) -> np.ndarray:
        tokens = self.embed_tokenizer(text_array, truncation=True, max_length=self.max_embed_length)
        return [{k: np.asarray(v[i]) for k, v in tokens.items()} for i in range(len(text_array))]

    def get_embeds(self, sentences: List[str]) -> np.ndarray:

        batch = self.embed_tokenizer(list(sentences), truncation=True, max_length=self.max_embed_length)
        batch = {
            "input_ids": pad_sequence_power_2(batch["input_ids"], int(self.embed_tokenizer.pad_token_id)),
            "attention_mask": pad_sequence_power_2(batch["attention_mask"], 0)
        }
        
        B = batch["input_ids"].shape[0]
        assert batch["input_ids"].shape[1] <= self.max_embed_length
        embeds = []
        for i in range(0, B, self.max_batch_size):
            subbatch = {k:v[i:i+self.max_batch_size] for k, v in batch.items()}
            embeds.append(fast_embed(self.embedder, subbatch["input_ids"], subbatch["attention_mask"]))

        return jax.device_get(jnp.concatenate(embeds, axis=0))

    
    def get_extra_embeds(self, embedder: nnx.Module) -> np.ndarray:
        batch = self.embed_tokenizer(list(self.all_texts), truncation=True, max_length=self.max_embed_length)
        batch = {
            "input_ids": pad_sequence_power_2(batch["input_ids"], int(self.embed_tokenizer.pad_token_id)),
            "attention_mask": pad_sequence_power_2(batch["attention_mask"], 0)
        }

        return fast_embed(embedder, batch["input_ids"], batch["attention_mask"])


    def reset(self, question: str, text_array: List[str]) -> TextMemory:
        self.all_texts = text_array

        tokens = self.tokenize(question, max_len=512)
        # self.action_tokens = self.tokenize_list(text_array)
        assert is_power_of_two(len(tokens["input_ids"]))

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
        action_tokens = self.tokenize(action_text)
        # action_tokens = self.action_tokens[action]
        is_empty = len(self.memory.text) == 0

        new_text = action_text if is_empty else self.memory.text + self.separator + action_text
        tokens = self.tokenize(new_text, max_len=512)

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        len_2 = 2 ** int(np.ceil(np.log2(len(input_ids))))
        
        input_ids = np.pad(input_ids, (0, len_2 - len(input_ids)), constant_values=int(self.embed_tokenizer.pad_token_id))
        attention_mask = np.pad(attention_mask, (0, len_2 - len(input_ids)), constant_values=0)

        available_mask = self.memory.available_mask.copy()
        available_mask[action] = False

        assert is_power_of_two(len(input_ids))
        assert is_power_of_two(len(action_tokens["input_ids"]))

        self.memory = TextMemory(
            item_ids=self.memory.item_ids + [action],
            available_ids=self.memory.available_ids - {action},
            available_mask=available_mask,
            text=new_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
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
            [si.input_ids for si in memory], 
            padding_value=int(self.tokenizer.pad_token_id))
        
        attention_mask = pad_sequence_power_2(
            [si.attention_mask for si in memory], 
            padding_value=0)
        
        embeds = pad_sequence_power_2(
            [si.embeds for si in memory], 
            padding_value=0)
        
        available_mask = pad_sequence_power_2(
            [si.available_mask for si in memory], 
            padding_value=False)
        
        s_memory = TextMemory(
            item_ids=[si.item_ids for si in memory],
            available_ids=[si.available_ids for si in memory],
            available_mask=available_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            text=[si.text for si in memory],
            embeds=embeds.astype(np.float32)
        )
        
        return s_memory
    

    def stack_actions(self, actions: List[TextMemoryItem]):
        input_ids = pad_sequence_power_2(
            [si.input_ids for si in actions], 
            padding_value=int(self.tokenizer.pad_token_id))
        
        attention_mask = pad_sequence_power_2(
            [si.attention_mask for si in actions], 
            padding_value=0)
        
        a_blocck = TextMemoryItem(
            index=[si.index for si in actions],
            input_ids=input_ids,
            attention_mask=attention_mask,
            text=[si.text for si in actions]
        )
        
        return a_blocck

    def sample(self, batch_size):
        s, a, r, next_s, not_done, entropy = super().sample(batch_size)

        s_stack = self.stack_memory(s)
        next_s_stack = self.stack_memory(next_s)
        a_stack = self.stack_actions(a)
        
        return s_stack, a_stack, next_s_stack, r.astype(np.float32), not_done.astype(np.float32), entropy.astype(np.float32)   
