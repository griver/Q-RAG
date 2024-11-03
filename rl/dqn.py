import os
import torch
from torch import nn, Tensor
from torch import optim
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from collections import namedtuple
from .q_module import TextQNet, TextQNetPolicy, TextRandomPolicy, ActionEmbedTarget, TextMaxQNet
from .text_env import TextMemory, TextMemoryItem
import copy


DQNArgs = namedtuple("DQNArgs", ["gamma", "tau", "lr", "max_steps"])


class DQN(object):
    def __init__(self, 
                 state_embed: nn.Module,
                 action_embed: nn.Module,
                 state_embed_target: nn.Module,
                 action_embed_target: nn.Module,
                 args: DQNArgs):

        self.gamma = args.gamma
        self.tau = args.tau

        self.critic = TextQNet(state_embed, action_embed).cuda()
        self.critic_optim = AdamW(self.critic.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay = 0.01, eps=1e-6)
        self.sheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_optim, args.max_steps, args.lr * 1e-2)

        self.v_net_target = TextMaxQNet(state_embed_target, self.critic)
        self.policy = TextQNetPolicy(copy.deepcopy(state_embed), self.critic)
        self.random_policy = TextRandomPolicy()
        self.action_embed_target = ActionEmbedTarget(action_embed_target, self.critic)


    def select_action(self, state: TextMemory, a_embeds: Tensor, evaluate=False, random=False):
        if random:
            action, logp, entropy = self.random_policy.forward(state)
        else:
            alpha = 0.0 if evaluate else 0.01
            action, logp, entropy = self.policy(state, a_embeds, alpha)
        
        return action.squeeze().item()

    def update(self, 
                state_batch: TextMemory, 
                action_batch: TextMemoryItem, 
                next_state_batch: TextMemory, 
                reward_batch: Tensor, 
                mask_batch: Tensor, 
                updates_count: int):
        
        reward_batch = reward_batch.squeeze()
        mask_batch = mask_batch.squeeze()

        with torch.no_grad():
            v_next_target_1, v_next_target_2 = self.v_net_target(next_state_batch)
            v_next_target = torch.minimum(v_next_target_1, v_next_target_2)
            next_q_value = reward_batch + mask_batch * self.gamma * v_next_target
            
        qf_1, qf_2 = self.critic(state_batch, action_batch)  
        qf_loss = 0.5 * F.mse_loss(qf_1, next_q_value) + 0.5 * F.mse_loss(qf_2, next_q_value)    
      
        self.critic_optim.zero_grad()
        qf_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.critic.parameters(), 1.0)
        self.critic_optim.step()
        self.sheduler.step()
            
        self.v_net_target.update(self.critic, self.tau)
        self.action_embed_target.update(self.critic, self.tau)

        return qf_loss.item()