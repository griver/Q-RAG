import numpy as np
from torch import nn, Tensor
import torch
from .text_env import TextMemory, TextMemoryItem
import copy


def logsumexp(inputs: Tensor, attention_mask: Tensor, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    inputs_copy = inputs.clone()
    inputs_copy[attention_mask == False] = torch.min(inputs_copy) - 1

    s, _ = torch.max(inputs_copy, dim=dim, keepdim=True)
    s_o = inputs - s
    outputs = s + (s_o.exp() * attention_mask.type(torch.int32)).sum(dim=dim, keepdim=True).log()

    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class TextQNet(nn.Module):

    def __init__(self, state_embed, action_embed) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.action_embed = action_embed

    def forward(self, s: TextMemory, a: TextMemoryItem): 
        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        a_embed = self.action_embed(input_ids=a.input_ids, attention_mask=a.attention_mask)
        logits = (s_embed * a_embed).sum(-1) 
        return logits


class ActionEmbedTarget(nn.Module):
    action_embed: nn.Module

    def __init__(self, action_embed: nn.Module, q_net: TextQNet) -> None:
        super().__init__()
        self.action_embed = action_embed
        self.action_embed.load_state_dict({
            k: v.clone() for k, v in q_net.action_embed.state_dict().items()
        })

    @torch.no_grad()
    def update(self, q_net: TextQNet, decay: float = 0.01):
        soft_update(self.action_embed, q_net.action_embed, decay)

    @torch.no_grad()
    def forward(self, *args, **kw):
        return self.action_embed.forward(*args, **kw)


class TextQNetPolicy(nn.Module):
    state_embed: nn.Module

    def __init__(self, state_embed: nn.Module, q_net: TextQNet) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.state_embed.load_state_dict({
            k: v.clone() for k, v in q_net.state_embed.state_dict().items()
        })

    @torch.no_grad()
    def update(self, q_net: TextQNet):
        hard_update(self.state_embed, q_net.state_embed)

    @torch.no_grad()
    def forward(self, s: TextMemory, a_embeds: Tensor, alpha: float):
        assert alpha > -1e-8

        input_ids = torch.from_numpy(s.input_ids).cuda()[None,]
        attention_mask = torch.from_numpy(s.attention_mask).cuda()[None,]
        mask = torch.from_numpy(s.available_mask).cuda()[None,]
        a_embeds = a_embeds[None,]

        s_embed = self.state_embed(input_ids=input_ids, attention_mask=attention_mask)
        s_embed = s_embed[:, None, :]
        
        logits = (s_embed * a_embeds).sum(-1) 
        logits[mask == False] = logits.min() - 1

        if alpha < 1e-8:
            return torch.argmax(logits, -1)

        probs = (logits / alpha).softmax(-1)
        probs[mask == False] = 0
        dist = torch.distributions.Categorical(probs = probs)
        action = dist.sample()

        return action, dist.log_prob(action), dist.entropy()
    

class TextRandomPolicy(nn.Module):


    @torch.no_grad()
    def forward(self, s: TextMemory):

        mask = torch.from_numpy(s.available_mask).cuda()
        
        probs = (torch.ones(mask.shape[0], device=mask.device)).softmax(-1)
        probs[mask == False] = 0
        dist = torch.distributions.Categorical(probs = probs)
        action = dist.sample()

        return action, dist.log_prob(action), dist.entropy()


class TextVNet(nn.Module):

    state_embed: nn.Module

    def __init__(self, state_embed: nn.Module, q_net: TextQNet) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.state_embed.load_state_dict({
            k: v.clone() for k, v in q_net.state_embed.state_dict().items()
        })


    @torch.no_grad()
    def update(self, q_net: TextQNet, decay: float = 0.01):
        soft_update(self.state_embed, q_net.state_embed, decay)

    @torch.no_grad()
    def forward(self, s: TextMemory, alpha: float):
        assert alpha > 1e-8

        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        s_embed = s_embed[:, None, :]
        a_embeds: Tensor = s.embeds
        
        logits = (s_embed * a_embeds).sum(-1) 

        return alpha * logsumexp(logits / alpha, attention_mask=s.available_mask, dim=-1)
