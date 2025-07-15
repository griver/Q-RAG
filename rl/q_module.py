import numpy as np
from torch import nn, Tensor
import torch
from envs.utils import TextMemory, TextMemoryItem
import copy


def logsumexp(inputs: Tensor, attention_mask: Tensor, dim=1, keepdim=False):
    
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    
    s_o = inputs - s
    exp_x = torch.exp(s_o) * attention_mask.to(inputs.dtype)

    outputs = s + torch.log(exp_x.sum(dim=dim, keepdim=True).clamp(min=1e-10))

    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    target.load_state_dict({
            k: v.clone() for k, v in source.state_dict().items()
    })
    # for target_param, param in zip(target.parameters(), source.parameters()):
    #     target_param.data.copy_(param.data)


class TextQNet(nn.Module):

    def __init__(self, state_embed, action_embed) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.action_embed = action_embed
        # self.action_embed.eval()
        # self.weight = nn.Parameter(torch.ones(1))
        # self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, s: TextMemory, a: TextMemoryItem): 
        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        # self.action_embed.eval()
        a_embed = self.action_embed(input_ids=a.input_ids, attention_mask=a.attention_mask, positions=a.position)["rope"]
        # a_embed = self.action_embed.update_pos(a_embed, positions=a.position)
        
        D = s_embed.shape[-1] // 2
        logits_1 = (s_embed[:, :D] * a_embed[:, :D]).sum(-1) 
        logits_2 = (s_embed[:, D:] * a_embed[:, D:]).sum(-1) 

        # logits_1 = logits_1 * self.weight + self.bias
        # logits_2 = logits_2 * self.weight + self.bias

        return logits_1, logits_2

        # return (s_embed * a_embed).sum(-1) 


class TextQNetTarget(TextQNet):
    @torch.no_grad()
    def update(self, q_net: TextQNet, decay: float = 0.01):
        soft_update(self, q_net, decay)


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
    
    @torch.no_grad()
    def update_pos(self, *args, **kw):
        return self.action_embed.update_pos(*args, **kw)



class TextQNetPolicy(nn.Module):
    state_embed: nn.Module

    def __init__(self, state_embed: nn.Module, q_net: TextQNet, top_k_actions=5) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.state_embed.load_state_dict({
            k: v.clone() for k, v in q_net.state_embed.state_dict().items()
        })
        self.top_k_actions = top_k_actions

    @torch.no_grad()
    def update(self, q_net: TextQNet):
        hard_update(self.state_embed, q_net.state_embed)

    @torch.no_grad()
    def forward(self, s: TextMemory, a_embeds: Tensor, alpha: float, return_arg_max=False):
        
        # a_embeds = a_embeds.unsqueeze(1)
        # print("a_embeds", a_embeds.shape)

        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        s_embed = s_embed.unsqueeze(1)
        
        logits = (s_embed * a_embeds).sum(-1).squeeze(-1) 
        # print("logits", logits.shape)
        logits[s.available_mask == False] = logits.min() - 1

        #print('\033[96m'+f'logits: {logits.shape}  topk: {self.top_k_actions}'+"\033[0m")
        top_k_actions = min(logits.size(1), self.top_k_actions)
        top_ids = torch.topk(logits, top_k_actions, dim=1).indices
        top_mask = torch.zeros_like(logits > 0).scatter_(1, top_ids, True)
        # print("top_mask", top_mask.shape)

        if return_arg_max:
            return torch.argmax(logits, -1), logits

        probs = ((logits - logits.max(-1, keepdim=True).values) / alpha).softmax(-1)
        probs[(s.available_mask & top_mask) == False] = 0
        probs = probs / probs.sum(-1, keepdim=True)
        dist = torch.distributions.Categorical(probs = probs)
        action = dist.sample()

        # print("action", action.shape)

        return action, logits
    

class TextRandomPolicy(nn.Module):


    @torch.no_grad()
    def forward(self, s: TextMemory):

        mask = s.available_mask
        
        probs = (torch.ones(mask.shape[0], mask.shape[1], device=mask.device)).softmax(-1)
        probs[mask == False] = 0
        dist = torch.distributions.Categorical(probs = probs)
        action = dist.sample()

        return action


class TextVNet(nn.Module):

    state_embed: nn.Module

    def __init__(self, state_embed: nn.Module, q_net: TextQNet, top_k_actions=5) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.state_embed.load_state_dict({
            k: v.clone() for k, v in q_net.state_embed.state_dict().items()
        })
        self.top_k_actions = top_k_actions


    @torch.no_grad()
    def update(self, q_net: TextQNet, decay: float = 0.01):
        soft_update(self.state_embed, q_net.state_embed, decay)

    @torch.no_grad()
    def forward(self, s: TextMemory, a_embeds_target: Tensor, alpha: float):
        # assert alpha > 1e-8

        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        s_embed = s_embed.unsqueeze(1)
        a_embeds: Tensor = a_embeds_target
        
        # logits = (s_embed * a_embeds).sum(-1) 
        D = s_embed.shape[-1] // 2
        logits_1 = (s_embed[:, :, :D] * a_embeds[:, :, :D]).sum(-1) 
        logits_2 = (s_embed[:, :, D:] * a_embeds[:, :, D:]).sum(-1) 

        top_k_actions = min(logits_1.size(1), self.top_k_actions)
        top_ids_1 = torch.topk(logits_1, top_k_actions, dim=1).indices
        top_mask_1 = torch.zeros_like(logits_1 > 0).scatter_(1, top_ids_1, True)

        top_ids_2 = torch.topk(logits_2, top_k_actions, dim=1).indices
        top_mask_2 = torch.zeros_like(logits_2 > 0).scatter_(1, top_ids_2, True)

        v1 = alpha * logsumexp(logits_1 / alpha, attention_mask=s.available_mask & top_mask_1, dim=-1)
        v2 = alpha * logsumexp(logits_2 / alpha, attention_mask=s.available_mask & top_mask_2, dim=-1)
    
        return v1, v2


class TextMaxQNet(nn.Module):

    state_embed: nn.Module

    def __init__(self, state_embed: nn.Module, q_net: TextQNet) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.state_embed.load_state_dict({
            k: v.clone() for k, v in q_net.state_embed.state_dict().items()
        })

        self.weight = nn.Parameter(torch.ones(1)).cuda()
        self.bias = nn.Parameter(torch.zeros(1)).cuda()


    @torch.no_grad()
    def update(self, q_net: TextQNet, decay: float = 0.01):
        soft_update(self.state_embed, q_net.state_embed, decay)
        self.weight.data = self.weight.data * (1 - decay) + q_net.weight.data * decay
        self.bias.data = self.bias.data * (1 - decay) + q_net.bias.data * decay

    @torch.no_grad()
    def forward(self, s: TextMemory):

        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        s_embed = s_embed[:, None, :]
        a_embeds: Tensor = s.embeds
        
        D = s_embed.shape[-1] // 2
        logits_1 = (s_embed[:, :, :D] * a_embeds[:, :, :D]).sum(-1) 
        logits_2 = (s_embed[:, :, D:] * a_embeds[:, :, D:]).sum(-1) 

        logits_1[s.available_mask == False] = torch.min(logits_1)
        logits_2[s.available_mask == False] = torch.min(logits_2)

        return (logits_1.max(-1).values, 
                logits_2.max(-1).values)

