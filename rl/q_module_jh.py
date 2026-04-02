import numpy as np
from torch import nn, Tensor
import torch
from envs.utils import TextMemory, TextMemoryItem
import copy
from contextlib import nullcontext


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

class DecomInnerProd(nn.Module):
    def __init__(self, n_decom_q=8, embed_dim=1024, num_heads=8, keep_orig=False):
        super().__init__()
        self.prelin = nn.Linear(embed_dim, n_decom_q * embed_dim)
        self.decom = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        # self.decom
        self.decom_q = nn.Parameter(torch.randn(1, n_decom_q, embed_dim) * 0.02)
        self.keep_orig = keep_orig

    def forward(self, s_embed, a_embed, together=False):
        # print("In", s_embed.shape, a_embed.shape)
        logits_shape = a_embed.shape[:-1]
        if s_embed.dim() == 3: s_embed = s_embed.view(-1, s_embed.shape[-1])
        if a_embed.dim() == 3: a_embed = a_embed.view(-1, a_embed.shape[-1])
        # print("reshape", s_embed.shape, a_embed.shape, "logit_shape", logits_shape)

        s_embed = s_embed.unsqueeze(1)
        # a_embed = a_embed.unsqueeze(1)

        decom_q = self.decom_q.expand(s_embed.shape[0], -1, -1)
        _s_embed = self.prelin(s_embed).reshape(decom_q.shape)
        decom_s_embed = self.decom(decom_q, _s_embed, _s_embed)[0] # B D H
        if self.keep_orig: decom_s_embed = torch.cat([s_embed, decom_s_embed], dim=1)

        logits_1s, logits_2s = [], []

        if together:
            for s_embed in decom_s_embed.unbind(dim=1):
                logits_1s.append((s_embed * a_embed).sum(-1))
        
            # print("together in list", logits_1s[0].shape)
            logits_1 = torch.logsumexp(torch.stack(logits_1s, dim=0), dim=0)
            # print("together logits", logits_1.shape)
            logits_1 = logits_1.reshape(logits_shape)
            
            return logits_1 

        else:
            D = s_embed.shape[-1] // 2
            for s_embed in decom_s_embed.unbind(dim=1):
                logits_1s.append((s_embed[:, :D] * a_embed[:, :D]).sum(-1))
                logits_2s.append((s_embed[:, D:] * a_embed[:, D:]).sum(-1))
        
            # print("in list", logits_1s[0].shape, logits_2s[0].shape)
            logits_1 = torch.logsumexp(torch.stack(logits_1s, dim=0), dim=0)
            logits_2 = torch.logsumexp(torch.stack(logits_2s, dim=0), dim=0)
            # print("logits", logits_1.shape, logits_2.shape)
            logits_1 = logits_1.reshape(logits_shape)
            logits_2 = logits_2.reshape(logits_shape)

            return logits_1, logits_2 

class PMMMat(nn.Module):
    def __init__(self, d_embed=1024, d_mat=5, k=None):
        super().__init__()
        _d_mat = d_mat * (d_mat + 1) // 2
        self.p = nn.Parameter(torch.randn(2, _d_mat))
        self.s = nn.Parameter(torch.randn(2 * d_embed, _d_mat))

        self.triu = torch.triu_indices(d_mat, d_mat)
        self.d_mat = d_mat
        if k is None: k = (d_mat + 1) // 2
        self.k = k

    def _mat(self, param):
        M = torch.zeros(1, param.shape[0], self.d_mat, self.d_mat)
        i, j = self.triu
        M[..., i, j] = param
        return M

    def forward(self, s_embed, a_embed, together=False):
        # print("In", s_embed.shape, a_embed.shape) 
        # In torch.Size([1, 1, 1024]) torch.Size([1, 10, 1024])

        # HACK?
        logits_shape = a_embed.shape[:-1]
        if s_embed.dim() == 3: s_embed = s_embed.view(-1, s_embed.shape[-1])#2 [1 512 1 1]
        if a_embed.dim() == 3: a_embed = a_embed.view(-1, a_embed.shape[-1])
        s_embed = s_embed.unsqueeze(-1).unsqueeze(-1)
        a_embed = a_embed.unsqueeze(-1).unsqueeze(-1)
        
        P = self._mat(self.p).expand(s_embed.shape[0], -1, -1, -1)
        S = self._mat(self.s).expand(a_embed.shape[0], -1, -1, -1)
        # print(P.shape, S.shape, s_embed.shape, a_embed.shape)
        
        P = P.chunk(2, dim=1) #2 [1 1 5 5]
        S = S.chunk(4, dim=1) #4 [10 512 5 5]
        s_embed = s_embed.chunk(2, dim=1)
        a_embed = a_embed.chunk(2, dim=1)

        eig1 = torch.linalg.eigvalsh(P[0].squeeze(dim=1) + (s_embed[0] * S[0]).sum(dim=1) + (a_embed[0] * S[1]).sum(dim=1))[..., self.k].reshape(logits_shape)
        eig2 = torch.linalg.eigvalsh(P[1].squeeze(dim=1) + (s_embed[1] * S[2]).sum(dim=1) + (a_embed[1] * S[3]).sum(dim=1))[..., self.k].reshape(logits_shape)

        if together: return eig1 + eig2
        return eig1, eig2

        # s = self.lin(s_embed).unsqueeze(-1).unsqueeze(-1)
        # a = self.lin(a_embed).unsqueeze(-1).unsqueeze(-1) # 10 2 1 1

        # A = self._mat(self.A).unsqueeze(0).unsqueeze(0).expand(a_embed.shape[0], -1, -1, -1).to(s_embed.device)
        # B = self._mat(self.B).unsqueeze(0).unsqueeze(0).expand(s_embed.shape[0], -1, -1, -1).to(s_embed.device)
        # C = self._mat(self.C).unsqueeze(0).unsqueeze(0).expand(a_embed.shape[0], -1, -1, -1).to(s_embed.device)
        # # print(s.shape,a.shape,A.shape,B.shape,C.shape)
        # eig = torch.linalg.eigvalsh(A + s * B + a * C)[:, :, self.k]
        
        # if together: return eig.sum(dim=1).reshape(logits_shape)
        # logit1, logit2 = eig.unbind(dim=1)
        # return logit1.reshape(logits_shape), logit2.reshape(logits_shape)

class GenIn(nn.Module):
    def __init__(self, d_embed=1024):
        super().__init__()
        self.s = nn.Linear(d_embed, d_embed, bias=False)
    
    def forward(self, s_embed, a_embed, together=False):
        # print("In", s_embed.shape, a_embed.shape) 
        # In torch.Size([1, 1, 1024]) torch.Size([1, 10, 1024])

        # HACK?
        logits_shape = a_embed.shape[:-1]
        if s_embed.dim() == 3: s_embed = s_embed.view(-1, s_embed.shape[-1])
        if a_embed.dim() == 3: a_embed = a_embed.view(-1, a_embed.shape[-1])

        s_embed = self.s(s_embed)
        a_embed = self.s(a_embed)

        if together:
            return (s_embed * a_embed).sum(-1).reshape(logits_shape)

        else:
            D = s_embed.shape[-1] // 2
            logits_1 = (s_embed[:, :D] * a_embed[:, :D]).sum(-1)
            logits_2 = (s_embed[:, D:] * a_embed[:, D:]).sum(-1)

            return logits_1.reshape(logits_shape), logits_2.reshape(logits_shape) 



class TextQNet(nn.Module): # Critic

    def __init__(self, state_embed, action_embed, decom_inner_prod=None) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.action_embed = action_embed
        # self.action_embed.eval()
        # self.weight = nn.Parameter(torch.ones(1))
        # self.bias = nn.Parameter(torch.zeros(1))
        # self.simple_state_query_decom = nn.Linear(...)
        self.decom_inner_prod = decom_inner_prod

    def forward(self, s: TextMemory, a: TextMemoryItem): 
        context = torch.no_grad() if isinstance(self.decom_inner_prod, (PMMMat, GenIn)) else nullcontext()

        with context:
            # print(f'Embedder Input [input_ids shape={s.input_ids.shape}]')
            s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
            # print(f'Embedder Output [shape={s_embed.shape}, dtype={s_embed.dtype}, device={s_embed.device}]')
            # self.action_embed.eval()
            # print(f'Action Embedder Input [input_ids shape={a.input_ids.shape}]')
            a_embed = self.action_embed(input_ids=a.input_ids, attention_mask=a.attention_mask, positions=a.position)["rope"]
            # print(f'Action Embedder Output [shape={a_embed.shape}, dtype={a_embed.dtype}, device={a_embed.device}]')
            # a_embed = self.action_embed.update_pos(a_embed, positions=a.position)

        if self.decom_inner_prod is not None:
            logits_1, logits_2 = self.decom_inner_prod(s_embed, a_embed)
        else:

            D = s_embed.shape[-1] // 2
            logits_1 = (s_embed[:, :D] * a_embed[:, :D]).sum(-1) 
            logits_2 = (s_embed[:, D:] * a_embed[:, D:]).sum(-1) 

            # logits_1 = logits_1 * self.weight + self.bias
            # logits_2 = logits_2 * self.weight + self.bias

        return logits_1, logits_2

        # return (s_embed * a_embed).sum(-1) 


class TextQNetTarget(TextQNet): # Critic target, EMA, unused
    @torch.no_grad()
    def update(self, q_net: TextQNet, decay: float = 0.01):
        soft_update(self, q_net, decay)


class ActionEmbedTarget(nn.Module): # Action embedder eval
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



class TextQNetPolicy(nn.Module): # Critic eval, state-half
    state_embed: nn.Module

    def __init__(self, state_embed: nn.Module, q_net: TextQNet, top_k_actions=5, decom_inner_prod=None) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.state_embed.load_state_dict({
            k: v.clone() for k, v in q_net.state_embed.state_dict().items()
        })
        self.top_k_actions = top_k_actions
        self.decom_inner_prod = decom_inner_prod

    @torch.no_grad()
    def update(self, q_net: TextQNet, decom_inner_prod=None):
        hard_update(self.state_embed, q_net.state_embed)
        if decom_inner_prod is not None:
            hard_update(self.decom_inner_prod, decom_inner_prod)

    @torch.no_grad()
    def forward(self, s: TextMemory, a_embeds: Tensor, alpha: float, return_arg_max=False):
        
        # a_embeds = a_embeds.unsqueeze(1)
        # print("a_embeds", a_embeds.shape)

        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        s_embed = s_embed.unsqueeze(1)
        
        if self.decom_inner_prod is not None:
            logits = self.decom_inner_prod(s_embed, a_embeds, together=True).squeeze(-1)
        else:
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
        # print(f'probs.sum(): {probs.sum(-1).item()}')
        # print(f'availables: {s.available_mask[0].tolist()}')
        # print(f'top_mask: {top_mask[0].tolist()}')
        # print(f'top_mask & avail: {(top_mask & s.available_mask)[0].tolist()}')
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


class TextVNet(nn.Module): # ???

    state_embed: nn.Module

    def __init__(self, state_embed: nn.Module, q_net: TextQNet, top_k_actions=5, decom_inner_prod=None) -> None:
        super().__init__()
        self.state_embed = state_embed
        self.state_embed.load_state_dict({
            k: v.clone() for k, v in q_net.state_embed.state_dict().items()
        })
        self.top_k_actions = top_k_actions
        self.decom_inner_prod = decom_inner_prod

    @torch.no_grad()
    def update(self, q_net: TextQNet, decay: float = 0.01, decom_inner_prod=None):
        soft_update(self.state_embed, q_net.state_embed, decay)
        if decom_inner_prod is not None:
            soft_update(self.decom_inner_prod, decom_inner_prod, decay)

    @torch.no_grad()
    def forward(self, s: TextMemory, a_embeds_target: Tensor, alpha: float):
        # assert alpha > 1e-8

        s_embed = self.state_embed(input_ids=s.input_ids, attention_mask=s.attention_mask)
        s_embed = s_embed.unsqueeze(1)
        a_embeds: Tensor = a_embeds_target
        
        # logits = (s_embed * a_embeds).sum(-1) 
        if self.decom_inner_prod is not None:
            logits_1, logits_2 = self.decom_inner_prod(s_embed, a_embeds)
        else:
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


class TextMaxQNet(nn.Module): # ??? unused

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

