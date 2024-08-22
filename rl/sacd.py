import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Adam
from collections import namedtuple
from .q_module import TextQNet, TextQNetPolicy, TextRandomPolicy, TextVNet, ActionEmbedTarget
from .text_env import TextMemory, TextMemoryItem
import copy


SACArgs = namedtuple("SACArgs", ["gamma", "tau", "alpha", "target_update_interval", "automatic_entropy_tuning", "lr"])


class SAC(object):
    def __init__(self, 
                 state_embed: nn.Module,
                 action_embed: nn.Module,
                 state_embed_target: nn.Module,
                 action_embed_target: nn.Module,
                 actions_count: int,
                 args: SACArgs):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.critic = TextQNet(state_embed, action_embed).cuda()
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.v_net_target = TextVNet(state_embed_target, self.critic)
        self.policy = TextQNetPolicy(copy.deepcopy(state_embed), self.critic)
        self.random_policy = TextRandomPolicy()
        self.action_embed_target = ActionEmbedTarget(action_embed_target, self.critic)

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -actions_count
            self.log_alpha = torch.tensor(-1.0, requires_grad=True, device=torch.device("cuda"))
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr * 3)


    def select_action(self, state: TextMemory, a_embeds: Tensor, evaluate=False, random=False):
        alpha = self.alpha if evaluate is False else 0.0

        if random:
            action, logp, entropy = self.random_policy.forward(state)
        else:
            action, logp, entropy = self.policy(state, a_embeds, alpha)
        
        return action.squeeze().item(), logp.squeeze().item(), entropy.squeeze().item()

    def update(self, 
                state_batch: TextMemory, 
                action_batch: TextMemoryItem, 
                next_state_batch: TextMemory, 
                reward_batch: Tensor, 
                mask_batch: Tensor, 
                entropy_batch: Tensor,
                updates_count: int):
        
        reward_batch = reward_batch.squeeze()
        mask_batch = mask_batch.squeeze()

        with torch.no_grad():
            v_next_target = self.v_net_target(next_state_batch, self.alpha)
            next_q_value = reward_batch + mask_batch * self.gamma * v_next_target
            
        qf = self.critic(state_batch, action_batch)  
        qf_loss = F.mse_loss(qf, next_q_value)  
      
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()


        if self.automatic_entropy_tuning:
            alpha_loss = (self.log_alpha.exp() * (entropy_batch + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.)
            
        if updates_count % self.target_update_interval == 0:
            self.v_net_target.update(self.critic, self.tau)
            self.action_embed_target.update(self.critic, self.tau)

        return qf_loss.item(), alpha_loss.item()