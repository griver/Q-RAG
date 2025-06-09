from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np
from torch import einsum, nn, Tensor
import torch
from copy import deepcopy
from typing import Dict
import rotary_embedding_torch
from einops import rearrange, repeat
from rotary_embedding_torch import apply_rotary_emb

class BertPredictor(nn.Module):

    def __init__(self, bert: RobertaModel, num_hidden_layers, tokenizer, model_dim, output_size, n_output) -> None:
        super().__init__()
        
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

    def _inject_class_token(self, input_ids: Tensor, attention_mask: Tensor):
        input_ids = input_ids.clone()
        input_ids[(input_ids == self.sep_token_id) | (input_ids == self.cls_token_id) | (attention_mask < 1e-5)] = self.pad_token_id
        
        prefix_1 = self.cls_token[None, :].repeat(input_ids.shape[0], 1)

        if self.n_output > 1:
            prefix_2 = self.output_token_ids[None, :].repeat(input_ids.shape[0], 1)
            prefix_3 = self.sep_token[None, :].repeat(input_ids.shape[0], 1)
            input_ids = torch.cat([prefix_1, prefix_2, prefix_3, input_ids], dim=1)
        else:
            input_ids = torch.cat([prefix_1, input_ids], dim=1)

        attention_mask = self.get_attention_mask(input_ids)
        token_type_ids = self.get_token_type_ids(input_ids)

        return input_ids, attention_mask, token_type_ids

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)
    
    def forward(self, input_ids, attention_mask, *args, **kw):
        
        # input_ids, attention_mask, token_type_ids = self._inject_class_token(input_ids, attention_mask)

        assert input_ids.shape[1] <= 512
        assert attention_mask.shape[1] == input_ids.shape[1]
 
        out = self.model.forward(
            input_ids, attention_mask, return_dict=False
        )[0]

        if self.n_output > 1:
            prediction  = out[:, 1: self.n_output + 1]
        else:
            mask = attention_mask.reshape(out.shape[0], out.shape[1], 1)
            prediction  = (out * mask).sum(1) / mask.sum(1)

        return prediction / 10



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

        return apply_rotary_emb(freqs, t, scale = scale, seq_dim = seq_dim)


class EmbedderWithPosEncoding(BertPredictor):
    def __init__(self,
                 bert: RobertaModel,
                 num_hidden_layers,
                 tokenizer,
                 model_dim,
                 output_size,
                 n_output,
                 encoding_type='rope',
                 max_seq_len=1000,
                 interpolate_factor=1
                 ) -> None:
        super().__init__(bert, num_hidden_layers, tokenizer, model_dim, output_size, n_output)
        assert encoding_type in ['rope', 'none'], "encoding_type must be 'rope' or 'none'"
        self.encoding_type = encoding_type
        if encoding_type != 'none': #
            self.rotary_emb = PositionalRotaryEmbedding(
                dim=model_dim // 2,
                cache_max_seq_len=max_seq_len,
                interpolate_factor=interpolate_factor
            )

    def forward(self, input_ids, attention_mask, positions, *args, **kw):
        embeds = super().forward(input_ids, attention_mask, *args, **kw)
        if self.encoding_type == 'none':
            return embeds

        # positions = kw.get('positions', None)
        seq_dim = 0 if len(embeds.shape) == 2 else 1

        # if positions is None:
        #     positions = torch.arange(embeds.shape[seq_dim], device = embeds.device, dtype = embeds.dtype)
        
        embeds = self.rotary_emb.rotate_queries_or_keys(embeds, positions, seq_dim=seq_dim, offset=0)
       
        return embeds
