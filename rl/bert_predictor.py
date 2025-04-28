from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np
from torch import nn, Tensor
import torch
from copy import deepcopy
from typing import Dict

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


import rotary_embedding_torch


class EmbedderWithPosEncoding(BertPredictor):
    def __init__(self, bert: RobertaModel, num_hidden_layers, tokenizer, model_dim, output_size, n_output, positional_coding=None) -> None:
        super().__init__(bert, num_hidden_layers, tokenizer, model_dim, output_size, n_output)
        self.rotary_emb = None

        if positional_coding == 'rope':
            #add positional shift to first part of the embeds
            self.rotary_emb = rotary_embedding_torch.RotaryEmbedding(dim=model_dim // 2)


    def forward(self, input_ids, attention_mask, *args, **kw):
        embeds = super().forward(input_ids, attention_mask, *args, **kw)
        if self.rotary_emb is not None:
            #Add positional embeddings to chunk embeds
            #This version only works  for a contiguous set of ids, i.e. [n, n+1, n+2, ... n+T-1]
            offset = kw.get('positional_offset', None)

            if offset is not None:
                T, D = embeds.shape
                assert T == 1, 'if you need to embed batch of chunks with custom positional indices, then embed them one by one'
                # TODO: implement version for custom set of ids, if needed

            offset = 0 if offset is None else offset
            embeds = self.rotary_emb.rotate_queries_or_keys(embeds, seq_dim=0, offset=offset)

        return embeds
