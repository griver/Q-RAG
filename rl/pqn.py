from functools import partial
import os
import torch
from torch import nn, Tensor
from torch import optim
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from collections import namedtuple
from rl.text_env import pad_sequence_power_2
from rl.sarsa import set_optim
from .q_module import TextQNet, TextQNetPolicy, TextRandomPolicy, ActionEmbedTarget, TextMaxQNet, TextVNet
from .text_env import TextMemory, TextMemoryItem
import copy

PQNArgs = namedtuple("PQNArgs", ["gamma", "tau", "lr", "max_steps"])

@partial(torch.compile, mode="reduce-overhead")
def train_step(
            critic,
            state_batch: TextMemory, 
            action_batch: TextMemoryItem, 
            reward_batch: Tensor):
    
    reward_batch = reward_batch.squeeze()
    
    qf_1, qf_2 = critic(state_batch, action_batch)
    qf = qf_1 + qf_2  
    qf_loss = F.mse_loss(qf, reward_batch)   
    
    qf_loss.backward()

    return qf_loss


@partial(torch.compile)
def policy_apply(policy, state, a_embeds, alpha, return_argmax: bool):
    return policy(state, a_embeds, alpha, return_argmax)


class PQN(object):

    DEFAULT_OPT = dict(
        optim = 'adamw',
        lr = 5e-5,
        eps = 1e-06,
        weight_decay = 0.01,
        beta1 = 0.9,
        beta2 = 0.98,
        dropout = 0.1,
        scheduler = 'linear',
        total_steps = 40000,
        lr_min_ratio = 0.0,
        warmup_steps = 1000,
    )


    def __init__(self, 
                 state_embed: nn.Module,
                 action_embed: nn.Module,
                 state_embed_target: nn.Module,
                 action_embed_target: nn.Module):

        self.gamma = 0.99
        self.alpha = 0.005
        self.Lambda = 0.6
        self.tau = 0.01
        self.start_lr = PQN.DEFAULT_OPT["lr"]

        self.critic = TextQNet(state_embed, action_embed).to(torch.get_default_device())
        self.critic_optim, self.sheduler = set_optim(self.critic, **PQN.DEFAULT_OPT)
       
        self.policy = TextQNetPolicy(copy.deepcopy(state_embed), self.critic).to(torch.get_default_device())
        self.random_policy = TextRandomPolicy().to(torch.get_default_device())

        self.v_net_target = TextVNet(state_embed_target, self.critic).to(torch.get_default_device())
        self.action_embed_target = ActionEmbedTarget(action_embed_target, self.critic).to(torch.get_default_device())


    @torch.no_grad()
    def select_action(self, state: TextMemory, a_embeds: Tensor, evaluate=False, random=False):
        
        input_ids = torch.from_numpy(state.input_ids).to(torch.get_default_device())
        attention_mask = torch.from_numpy(state.attention_mask).to(torch.get_default_device())
        mask = torch.from_numpy(state.available_mask).to(torch.get_default_device()).unsqueeze(0)

        input_ids = pad_sequence_power_2(
            [input_ids], 
            batch_first=True, 
            padding_value=0)
        
        attention_mask = pad_sequence_power_2(
            [attention_mask], 
            batch_first=True, 
            padding_value=0)
        
        torch_state = TextMemory(
            item_ids=None,
            available_ids=None,
            available_mask=mask,
            text=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            embeds=torch.from_numpy(state.embeds).to(torch.get_default_device()).unsqueeze(0)
        )
        action, q_values = policy_apply(self.policy, torch_state, a_embeds, torch.tensor(self.alpha), evaluate)

        v1, v2 = self.v_net_target(torch_state, alpha=self.alpha)
        q_values_target = v1 + v2

        if random:
            action, logp, entropy = self.random_policy.forward(state)
            
        return action.squeeze().item(), q_values.squeeze(), q_values_target.squeeze()
        
    

    @torch.no_grad()
    def _get_target(self, lambda_returns, next_q, q_values, rewards, dones_mask):
        target_bootstrap = (
            rewards + self.gamma * dones_mask * next_q
        )
        delta = lambda_returns - next_q
        lambda_returns = (
            target_bootstrap + self.gamma * self.Lambda * delta
        )
        lambda_returns = dones_mask * lambda_returns + (1.0 - dones_mask) * rewards
        next_q = q_values

        return lambda_returns, next_q


    def update(self, 
                state_batch: TextMemory, 
                action_batch: TextMemoryItem, 
                next_state_batch: TextMemory, 
                q_values_batch: Tensor,
                reward_batch: Tensor, 
                mask_batch: Tensor):

        # with torch.no_grad():
        #     v1, v2 = self.v_net_target(state_batch, alpha=self.alpha)
        #     q_values_batch = v1 + v2
        
        last_q = mask_batch[-1] * q_values_batch[-1]
        lambda_returns = reward_batch[-1] + self.gamma * last_q

        targets = []

        for t in range(q_values_batch.shape[0] - 2, -1, -1):
            lambda_returns, last_q = self._get_target(lambda_returns, last_q, q_values_batch[t], reward_batch[t], mask_batch[t])
            targets.append(lambda_returns)

        targets.reverse()
        targets = torch.stack(targets) 
        
        state_batch = TextMemory(
                item_ids=None,
                available_ids=None,
                available_mask=state_batch.available_mask[:-1],
                text=None,
                input_ids=state_batch.input_ids[:-1],
                attention_mask=state_batch.attention_mask[:-1],
                embeds=None
            )
        
        action_batch = TextMemoryItem(
            index=None, 
            input_ids=action_batch.input_ids[:-1],
            attention_mask=action_batch.attention_mask[:-1],
            text=None
        )
        
        self.critic_optim.zero_grad()

        qf_loss = train_step(
            self.critic,
            state_batch, action_batch, targets)      

        self.critic_optim.step()  
        
        self.sheduler.step()
        self.alpha = 0.005 * self.sheduler.get_lr()[0] / self.start_lr 

        # self.v_net_target.update(self.critic, self.tau)
        # self.action_embed_target.update(self.critic, self.tau)

        return qf_loss.item()