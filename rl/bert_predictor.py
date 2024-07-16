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
        self.pad_token_id: int = tokenizer.pad_token_id
        self.cls_token_id: int = tokenizer.cls_token_id
        self.sep_token_id: int = tokenizer.sep_token_id
        self.register_buffer('cls_token', torch.tensor([tokenizer.cls_token_id]))
        self.register_buffer('sep_token', torch.tensor([tokenizer.sep_token_id]))

        config = deepcopy(bert.config)
        config.num_hidden_layers = num_hidden_layers

        self.model = RobertaModel(config)

        self.model.embeddings.load_state_dict({k: v.clone() for k, v in bert.embeddings.state_dict().items()})
        for i in range(config.num_hidden_layers):
            self.model.encoder.layer[i].load_state_dict({k: v.clone() for k, v in bert.encoder.layer[i].state_dict().items()})

        self.model.train()

        vocab_size: int = self.model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + self.n_output
        self.register_buffer('output_token_ids', torch.arange(vocab_size, extended_vocab_size))
        self.model.resize_token_embeddings(extended_vocab_size)

    def _inject_class_token(self, input_ids: Tensor, attention_mask: Tensor):
        input_ids = input_ids.clone()
        input_ids[(input_ids == self.sep_token_id) | (input_ids == self.cls_token_id) | (attention_mask < 1e-5)] = self.pad_token_id
        
        prefix_1 = self.cls_token[None, :].repeat(input_ids.shape[0], 1)
        prefix_2 = self.output_token_ids[None, :].repeat(input_ids.shape[0], 1)
        prefix_3 = self.sep_token[None, :].repeat(input_ids.shape[0], 1)

        if self.n_output > 1:
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
        
        input_ids, attention_mask, token_type_ids = self._inject_class_token(input_ids, attention_mask)

        out = self.model.forward(
            input_ids, attention_mask, token_type_ids, return_dict=False
        )[0]

        if self.n_output > 1:
            prediction  = out[:, 1: self.n_output + 1]
        else:
            prediction  = out[:, 0]

        return self.head(prediction)