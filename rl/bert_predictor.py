from abc import ABC, abstractmethod

import fsspec.utils
from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np
from torch import einsum, nn, Tensor
import torch
from copy import deepcopy
from typing import Dict
import rotary_embedding_torch
from einops import rearrange, repeat
from rotary_embedding_torch import apply_rotary_emb
import torch.nn.functional as F

class BertPredictor(nn.Module):

    def __init__(self, bert: RobertaModel, num_hidden_layers, tokenizer, model_dim, output_size, n_output) -> None:
        super().__init__()
        
        self.model_dim = model_dim
        self.head = nn.Linear(model_dim, output_size)
        self.n_output = n_output
        self.tokenizer = tokenizer
        self.pad_token_id: int = tokenizer.pad_token_id
        self.cls_token_id: int = tokenizer.cls_token_id
        self.sep_token_id: int = tokenizer.sep_token_id
        self.register_buffer('cls_token', torch.tensor([tokenizer.cls_token_id]))
        self.register_buffer('sep_token', torch.tensor([tokenizer.sep_token_id]))

        config = deepcopy(bert.config)
        config.num_hidden_layers = num_hidden_layers

        self.model = AutoModel.from_config(config)

        self.model.embeddings.load_state_dict({k: v.clone() for k, v in bert.embeddings.state_dict().items()})
        for i in range(config.num_hidden_layers):
            self.model.encoder.layer[i].load_state_dict({k: v.clone() for k, v in bert.encoder.layer[i].state_dict().items()})

        self.model.train()

        vocab_size: int = self.model.embeddings.word_embeddings.weight.shape[0]
        if self.n_output > 1:
            extended_vocab_size = vocab_size + self.n_output
            self.register_buffer('output_token_ids', torch.arange(vocab_size, extended_vocab_size))
            self.model.resize_token_embeddings(extended_vocab_size)

    def forward(self, input_ids, attention_mask, *args, **kw):
        
        assert attention_mask.shape[1] == input_ids.shape[1]
 
        out = self.model.forward(
            input_ids, attention_mask, return_dict=False
        )[0]

        if self.n_output > 1:
            prediction  = out[:, 1: self.n_output + 1]
        else:
            mask = attention_mask.reshape(out.shape[0], out.shape[1], 1)
            prediction  = (out * mask).sum(1) / mask.sum(1)

        #print(f'Embedder Output [shape={prediction.shape}, dtype={prediction.dtype}, device={prediction.device}]')
        return prediction / 10

@torch.no_grad()
def get_embedder_dim(tokenizer, embedder):
    embedder_dim = embedder.config.hidden_size
    # out = embedder(tokenizer.encode('hello!', return_tensors='pt').to(embedder.device))
    # embedder_dim = out.last_hidden_state.shape[-1]
    return embedder_dim


class SimpleEmbedder(nn.Module):

    def __init__(self, model_name, normalize_embeds=True, tokenizer_kwargs=None, model_kwargs=None) -> None:
        super().__init__()
        self.normalize_embeds = normalize_embeds
        tokenizer_kwargs = {} if tokenizer_kwargs is None else tokenizer_kwargs
        model_kwargs = {} if model_kwargs is None else model_kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model_dim = get_embedder_dim(self.tokenizer, self.model)
        self.model.train()
        print('SimpleEmbedder.dtype:', self.model.dtype)
        print('SimpleEmbedder.model_dim:', self.model_dim)

    def forward(self, input_ids, attention_mask, *args, **kw):
        assert attention_mask.shape[1] == input_ids.shape[1]

        out = self.model.forward(input_ids, attention_mask)
        #this two lines adapted from QWEN 3
        embeds = self.last_token_pool(out.last_hidden_state, attention_mask)
        if self.normalize_embeds:
            embeds = F.normalize(embeds, p=2, dim=1)
        #print(f'Embedder Output [shape={embeds.shape}, dtype={embeds.dtype}, device={embeds.device}]')
        return embeds

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# class SimpleEmbedderOld(nn.Module):
#     """Load a HuggingFace model and produce a single embedding per sequence."""
#
#     def __init__(self,
#                  model_name: str,
#                  revision: str = "main",
#                  pooling: str = "cls",
#                  use_fast_tokenizer: bool = True):
#         super().__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name,
#                                                        revision=revision,
#                                                        use_fast=use_fast_tokenizer)
#         self.model = AutoModel.from_pretrained(model_name, revision=revision)
#         self.pooling = pooling
#
#     def forward(self, input_ids, attention_mask, *args, **kwargs):
#         assert attention_mask.shape[1] == input_ids.shape[1]
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#
#         last_hidden = outputs.last_hidden_state
#         if self.pooling == "cls":
#             emb = last_hidden[:, 0]
#         elif self.pooling == "mean":
#             mask = attention_mask.unsqueeze(-1)
#             emb = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
#         else:
#             raise ValueError(f"Unknown pooling type: {self.pooling}")
#         return self.out_proj(emb)


class PositionalRotaryEmbedding(rotary_embedding_torch.RotaryEmbedding):

    def __init__(self, 
                 dim, 
                 custom_freqs = None, 
                 freqs_for = 'lang', 
                 theta=10000, 
                 max_freq=10, 
                 num_freqs=1, 
                 learned_freq=False, 
                 use_xpos=False, 
                 xpos_scale_base=512, 
                 interpolate_factor=1, 
                 theta_rescale_factor=1, 
                 seq_before_head_dim=False, 
                 cache_if_possible=True, 
                 cache_max_seq_len=8192):
        super().__init__(dim, custom_freqs, freqs_for, theta, max_freq, num_freqs, 
                         learned_freq, use_xpos, xpos_scale_base, 1, #set interpolate_factor to 1. to ignore assert requiring interpolate factor >= 1 
                         theta_rescale_factor, seq_before_head_dim, cache_if_possible, cache_max_seq_len)
        
        assert interpolate_factor > 0.
        self.interpolate_factor = interpolate_factor
        print('interpolate factor:', self.interpolate_factor)
        
        freqs = self.freqs
        positions = torch.arange(cache_max_seq_len, device = torch.get_default_device())

        freqs = einsum('..., f -> ... f', positions.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        self.cached_freqs = freqs.detach()
        
    def forward(
        self,
        t: Tensor
    ):
        return self.cached_freqs[t.type(torch.int32)].detach()

    def get_seq_pos(self, positions, offset = 0):
        return (positions + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, positions, seq_dim = None, offset = 0, scale = 1.0):
        seq_dim = self.default_seq_dim if seq_dim is None else seq_dim

        seq = self.get_seq_pos(positions, offset=offset)
        freqs = self.forward(seq)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        # return apply_rotary_emb(freqs, t, scale = scale, seq_dim = seq_dim)
        D = t.shape[-1] // 2

        t1 = apply_rotary_emb(freqs, t[..., :D], scale = scale, seq_dim = seq_dim)
        t2 = apply_rotary_emb(freqs, t[..., D:], scale = scale, seq_dim = seq_dim)

        return torch.cat([t1, t2], dim=-1)
    

class EmbedderBase(nn.Module, ABC):

    def __init__(self, model: BertPredictor):
        super().__init__()
        self.model = model
        self.tokenizer = model.tokenizer

    @abstractmethod
    def update_pos(self, embeds: Dict[str, Tensor], positions: Tensor, *args, **kw): pass

    @abstractmethod
    def forward(self, input_ids, attention_mask, positions, *args, **kw): pass

    
class EmbedderNone(EmbedderBase):

    def update_pos(self, embeds, positions, *args, **kw):
        return embeds

    def forward(self, input_ids, attention_mask, positions, *args, **kw):
        embeds = self.model.forward(input_ids, attention_mask, *args, **kw)
        return {"rope": embeds}


class EmbedderWithAbsoluteEncoding(EmbedderBase):
    def __init__(self,
                 model: BertPredictor,
                 max_seq_len=1000,
                 interpolate_factor=1
                 ) -> None:
        super().__init__(model)
        self.rotary_emb = PositionalRotaryEmbedding(
            dim=model.model_dim // 2,
            cache_max_seq_len=max_seq_len,
            interpolate_factor=interpolate_factor
        )

    def update_pos(self, embeds, positions, *args, **kw):
        return embeds

    def forward(self, input_ids, attention_mask, positions, *args, **kw):
        embeds = self.model.forward(input_ids, attention_mask, *args, **kw)
        seq_dim = 0 if len(embeds.shape) == 2 else 1
        embeds = self.rotary_emb.rotate_queries_or_keys(embeds, positions, seq_dim=seq_dim, offset=0)
       
        return {"rope": embeds}
    

class EmbedderWithRelativeEncoding(EmbedderBase):
    def __init__(self, 
                 model: BertPredictor,
                 max_seq_len=1000) -> None:
        super().__init__(model)
        self.rotary_emb = PositionalRotaryEmbedding(dim=model.model_dim // 2, cache_max_seq_len=max_seq_len)
        
    def update_pos(self, embeds, positions, *args, **kw):
        seq_dim = 0 if len(embeds["none"].shape) == 2 else 1        
        rope_embeds = self.rotary_emb.rotate_queries_or_keys(embeds["none"], positions, seq_dim=seq_dim, offset=0) 

        return {"rope": rope_embeds, "none": embeds["none"]}

    def forward(self, input_ids, attention_mask, positions, *args, **kw):
        embeds = self.model.forward(input_ids, attention_mask, *args, **kw)
        seq_dim = 0 if len(embeds.shape) == 2 else 1        
        rope_embeds = self.rotary_emb.rotate_queries_or_keys(embeds, positions, seq_dim=seq_dim, offset=0)
        return {"rope": rope_embeds, "none": embeds}
