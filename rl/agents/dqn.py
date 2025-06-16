from functools import partial
import os
import torch
from torch import nn, Tensor
from torch import optim
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from collections import namedtuple

from rl.agents.sarsa import LinearAnnealingVal, set_optim
from ..q_module import TextQNet, TextQNetPolicy, TextRandomPolicy, ActionEmbedTarget, TextMaxQNet, TextVNet
from .text_env import TextMemory, TextMemoryItem
import copy
import numpy as np

DQNArgs = namedtuple("DQNArgs", ["gamma", "tau", "lr", "max_steps"])

# @partial(torch.compile)
def train_step(
            critic,
            v_net_target,
            state_batch: TextMemory, 
            action_batch: TextMemoryItem, 
            next_state_batch: TextMemory, 
            reward_batch: Tensor, 
            mask_batch: Tensor,
            alpha: Tensor,
            gamma: Tensor):
    
    reward_batch = reward_batch.squeeze()
    mask_batch = mask_batch.squeeze()

    with torch.no_grad():
        v_next_target_1, v_next_target_2 = v_net_target(next_state_batch, alpha=alpha)
        v_next_target = torch.minimum(v_next_target_1, v_next_target_2)
        next_q_value = reward_batch + mask_batch * gamma * v_next_target
        
    qf_1, qf_2 = critic(state_batch, action_batch)  
    qf_loss = 0.5 * F.mse_loss(qf_1, next_q_value) + 0.5 * F.mse_loss(qf_2, next_q_value)    
    
    qf_loss.backward()

    return qf_loss


@partial(torch.compile)
def policy_apply(policy, state, a_embeds, alpha, return_argmax: bool):
    return policy(state, a_embeds, alpha, return_argmax)


class DQN(object):

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
                 args: DQNArgs):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = 0.005
        self.start_lr = args.lr

        self.epsilon = LinearAnnealingVal(
            1000, 10000,
            0.3, 0.01
        )

        self.critic = TextQNet(state_embed, action_embed).to(torch.get_default_device())
        self.critic_optim, self.sheduler = set_optim(self.critic, **DQN.DEFAULT_OPT)
        # self.critic_optim = AdamW(self.critic.parameters(), lr=torch.tensor(args.lr), betas=(0.9, 0.99), weight_decay = 0.01, eps=1e-6)
        # self.sheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_optim, args.max_steps, args.lr * 1e-2)

        self.v_net_target = TextVNet(state_embed_target, self.critic).to(torch.get_default_device())
        self.policy = TextQNetPolicy(copy.deepcopy(state_embed), self.critic).to(torch.get_default_device())
        self.random_policy = TextRandomPolicy().to(torch.get_default_device())
        self.action_embed_target = ActionEmbedTarget(action_embed_target, self.critic).to(torch.get_default_device())

    @torch.no_grad()
    def select_action(self, state: TextMemory, a_embeds: Tensor, evaluate=False, random=False):

        epsilon = self.epsilon.step()
        random = np.random.random() < epsilon

        if random and not evaluate:
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
            action, _ = policy_apply(self.policy, torch_state, a_embeds, torch.tensor(self.alpha), True)
        
        return action.squeeze().item()


    def update(self, 
                state_batch: TextMemory, 
                action_batch: TextMemoryItem, 
                next_state_batch: TextMemory, 
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
        
        self.critic_optim.zero_grad()

        qf_loss = train_step(
            self.critic, self.v_net_target, 
            state_batch, action_batch, next_state_batch, reward_batch, mask_batch, 
            torch.tensor(self.alpha), torch.tensor(self.gamma))      

        self.critic_optim.step()  
        
        self.sheduler.step()
        self.alpha = 0.005 * self.sheduler.get_lr()[0].item() / self.start_lr 
            
        self.v_net_target.update(self.critic, self.tau)
        self.action_embed_target.update(self.critic, self.tau)

        return qf_loss.item()