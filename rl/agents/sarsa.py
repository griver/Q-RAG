from functools import partial
import os
import torch
from torch import nn, Tensor
from torch import optim
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from collections import namedtuple
from ..q_module import TextQNet, TextQNetPolicy, TextQNetTarget, TextRandomPolicy, ActionEmbedTarget, TextMaxQNet, TextVNet
from envs.utils import TextMemory, TextMemoryItem
import copy
from functools import partial
import os
import torch
from torch import nn, Tensor
from torch import optim
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from collections import namedtuple
from ..q_module import TextQNet, TextQNetPolicy, TextRandomPolicy, ActionEmbedTarget, TextMaxQNet, TextVNet
import copy
import numpy as np
import math


SARSAArgs = namedtuple("SARSAArgs", ["lr", "max_steps", "warmup_steps", "exploration_steps",
            "epsilon_warmup", "epsilon_final"])

@partial(torch.compile, options={"shape_padding": True}, dynamic=False)
def train_step(
            critic,
            critic_target,
            critic_optim,
            state_batch: TextMemory, 
            action_batch: TextMemoryItem, 
            next_state_batch: TextMemory, 
            next_action_batch: TextMemoryItem, 
            reward_batch: Tensor, 
            mask_batch: Tensor,
            gamma: Tensor):
    
    reward_batch = reward_batch.squeeze()
    mask_batch = mask_batch.squeeze()

    qf_1, qf_2 = critic(state_batch, action_batch) 
    qf = qf_1 + qf_2

    with torch.no_grad():
        v_1, v_2 = critic_target(next_state_batch, next_action_batch)
        next_q_value = reward_batch + mask_batch * gamma * (v_1 + v_2)
        
    qf_loss = F.mse_loss(qf, next_q_value)
    
    critic_optim.zero_grad()
    qf_loss.backward()
    critic_optim.step()

    return qf_loss


@partial(torch.compile, options={"shape_padding": True})
def policy_apply(policy, state, a_embeds, alpha, return_argmax: bool):
    return policy(state, a_embeds, alpha, return_argmax)




class SARSA(object):

    DEFAULT_OPT = dict(
        optim = 'adamw',
        lr = torch.tensor(5e-5),
        eps = 1e-06,
        weight_decay = 0.01,
        beta1 = 0.9,
        beta2 = 0.98,
        dropout = 0.1,
        scheduler = 'linear',
        total_steps = 200000,
        lr_min_ratio = 0.0,
        warmup_steps = 1000,
    )

    def __init__(self, 
                 state_embed: nn.Module,
                 action_embed: nn.Module,
                 state_embed_target: nn.Module,
                 action_embed_target: nn.Module,
                 args: SARSAArgs):

        self.gamma = 0.99
        self.tau = 0.01

        self.epsilon = LinearAnnealingVal(
            args.warmup_steps, args.exploration_steps,
            args.epsilon_warmup, args.epsilon_final
        )

        self.critic = TextQNet(state_embed, action_embed).to(torch.get_default_device())
        self.critic_optim, self.scheduler = set_optim(self.critic, **SARSA.DEFAULT_OPT)
        # self.critic_optim = AdamW(self.critic.parameters(), lr=torch.tensor(args.lr), betas=(0.9, 0.98), weight_decay = 0.01, eps=1e-6)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_optim, args.max_steps, args.lr * 1e-2)

        self.critic_target = TextQNetTarget(state_embed_target, action_embed_target).to(torch.get_default_device())
        self.critic_target.load_state_dict({
            k: v.clone() for k, v in self.critic.state_dict().items()
        })

        self.policy = TextQNetPolicy(copy.deepcopy(state_embed), self.critic).to(torch.get_default_device())
        self.random_policy = TextRandomPolicy().to(torch.get_default_device())

    @torch.no_grad()
    def select_action(self, state: TextMemory, a_embeds: Tensor, evaluate=False):

        curr_epsilon = self.epsilon.step()

        self.policy.eval()

        if np.random.random() < curr_epsilon and not evaluate:
            action, logp, entropy = self.random_policy.forward(state)
        else:
            input_ids = torch.from_numpy(state.input_ids).to(torch.get_default_device()).unsqueeze(0)
            attention_mask = torch.from_numpy(state.attention_mask).to(torch.get_default_device()).unsqueeze(0)
            mask = torch.from_numpy(state.available_mask).to(torch.get_default_device()).unsqueeze(0)
            
            torch_state = TextMemory(
                item_ids=None,
                available_ids=None,
                available_mask=mask,
                text=None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                embeds=None
            )
            action, logp, entropy = policy_apply(self.policy, torch_state, a_embeds, torch.tensor(0), True)
        
        return action.squeeze().item()


    def update(self, 
                state_batch: TextMemory, 
                action_batch: TextMemoryItem, 
                next_state_batch: TextMemory, 
                next_action_batch: TextMemoryItem, 
                reward_batch: Tensor, 
                mask_batch: Tensor):
        
        state_batch = TextMemory(
                item_ids=None,
                available_ids=None,
                available_mask=state_batch.available_mask,
                text=None,
                input_ids=state_batch.input_ids,
                attention_mask=state_batch.attention_mask,
                embeds=state_batch.embeds
            )
        
        next_state_batch = TextMemory(
                item_ids=None,
                available_ids=None,
                available_mask=next_state_batch.available_mask,
                text=None,
                input_ids=next_state_batch.input_ids,
                attention_mask=next_state_batch.attention_mask,
                embeds=next_state_batch.embeds
            )
        
        action_batch = TextMemoryItem(
            index=None, 
            input_ids=action_batch.input_ids,
            attention_mask=action_batch.attention_mask,
            text=None
        )

        next_action_batch = TextMemoryItem(
            index=None, 
            input_ids=next_action_batch.input_ids,
            attention_mask=next_action_batch.attention_mask,
            text=None
        )
        
        qf_loss = train_step(
            self.critic, self.critic_target, self.critic_optim,
            state_batch, action_batch, next_state_batch, next_action_batch, reward_batch, mask_batch, torch.tensor(self.gamma))        
        
        self.scheduler.step()
        self.critic_target.update(self.critic, self.tau)

        return qf_loss.item()


