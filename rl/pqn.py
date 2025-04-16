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
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


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
def policy_apply(policy, v_net, state, a_embeds, alpha, return_argmax: bool):
    action, q_values = policy(state, a_embeds, alpha, return_argmax)
    # v1, v2 = v_net(state, alpha=alpha)
    # q_values_target = v1 + v2
    q_values_target = torch.tensor(0.0)
    return action, q_values, q_values_target


class PQN(object):

    def __init__(self, config: DictConfig):

        self.gamma = config.pqn.hyperparams.gamma
        self.alpha = config.pqn.hyperparams.alpha
        self.alpha_start = self.alpha 
        self.Lambda = config.pqn.hyperparams.Lambda
        self.tau = config.pqn.hyperparams.tau
        self.start_lr = config.pqn.optimizer.lr

        state_embed: nn.Module = instantiate(config.pqn.state_embed)
        action_embed: nn.Module = instantiate(config.pqn.action_embed)
        state_embed_target: nn.Module = instantiate(config.pqn.state_embed_target)
        action_embed_target: nn.Module = instantiate(config.pqn.action_embed_target)

        self.critic = TextQNet(state_embed, action_embed).to(torch.get_default_device())
        self.critic_optim = instantiate(config.pqn.optimizer, params=self.critic.parameters())
        self.scheduler = instantiate(config.pqn.scheduler, optimizer=self.critic_optim)
       
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
        action, q_values, q_values_target = policy_apply(self.policy, self.v_net_target, torch_state, a_embeds, torch.tensor(self.alpha), evaluate)

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

        with torch.no_grad():
            v1, v2 = self.v_net_target(state_batch, alpha=self.alpha / 10)
            q_values_batch = v1 + v2
        
        last_q = mask_batch[-2] * q_values_batch[-1]
        lambda_returns = reward_batch[-2] + self.gamma * last_q

        targets = [lambda_returns]

        for t in range(q_values_batch.shape[0] - 3, -1, -1):
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
        self.scheduler.step()
        
        self.alpha = self.alpha_start * self.scheduler.get_lr()[0] / self.start_lr 

        return qf_loss.item()