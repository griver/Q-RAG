from functools import partial
import os
from typing import Tuple
import torch
from torch import nn, Tensor
from torch import optim
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from collections import namedtuple
from rl.bert_predictor import EmbedderWithAbsoluteEncoding
from envs.utils import custom_pad_sequence, stack_memory, stack_text_list
from ..q_module import TextQNet, TextQNetPolicy, TextRandomPolicy, ActionEmbedTarget, TextMaxQNet, TextVNet
from envs.utils import TextMemory, TextMemoryItem
import copy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


@partial(torch.compile)
def policy_apply(policy, v_net, state, a_embeds,  a_embeds_target, alpha, return_argmax: bool):
    action, q_values = policy(state, a_embeds, alpha, return_argmax)
    v1, v2 = v_net(state, a_embeds_target, alpha=alpha)
    q_values_target = v1 + v2
    return action, q_values, q_values_target


class PQN(object):

    def __init__(self, config: DictConfig):

        self.config = copy.deepcopy(config)
        self.gamma = config.pqn.hyperparams.gamma
        self.alpha = config.pqn.hyperparams.alpha
        self.alpha_start = self.alpha 
        self.Lambda = config.pqn.hyperparams.Lambda
        self.tau = config.pqn.hyperparams.tau
        self.start_lr = config.pqn.optimizer.lr
        # ===new===
        self.max_grad_norm = config.pqn.hyperparams.max_grad_norm
        self.accumulate_grads = config.pqn.hyperparams.accumulate_grads
        if self.accumulate_grads < 1:
            raise ValueError("cfg.accumulate_gradients must be a positive integer")
        self._update_step = 0  # number of updates from the start of the training
        self.action_embed_length = config.pqn.hyperparams.action_embed_length

        state_embed: nn.Module = instantiate(config.pqn.state_embed)
        action_embed: nn.Module = instantiate(config.pqn.action_embed)
        state_embed_target: nn.Module = instantiate(config.pqn.state_embed_target)
        action_embed_target: nn.Module = instantiate(config.pqn.action_embed_target)
        state_embed_copy = copy.deepcopy(state_embed)
        
        self.critic = TextQNet(state_embed, action_embed).to(torch.get_default_device())
        self.critic_optim = instantiate(config.pqn.optimizer, params=self.critic.parameters())
        self.scheduler = instantiate(config.pqn.scheduler, optimizer=self.critic_optim)
       
        self.policy = TextQNetPolicy(state_embed_copy, self.critic).to(torch.get_default_device())
        self.random_policy = TextRandomPolicy().to(torch.get_default_device())

        self.v_net_target = TextVNet(state_embed_target, self.critic).to(torch.get_default_device())
        self.action_embed_target = ActionEmbedTarget(action_embed_target, self.critic).to(torch.get_default_device())

        self.state_tokenizer = state_embed.tokenizer
        self.action_tokenizer = action_embed.tokenizer

        self.train_step = self.make_train_step()


    def make_train_step(self):

        # @partial(torch.compile)
        def train_step(
                    critic,
                    state_batch: TextMemory, 
                    action_batch: TextMemoryItem, 
                    reward_batch: Tensor
        ):
            
            reward_batch = reward_batch.squeeze()
            
            qf_1, qf_2 = critic(state_batch, action_batch)
            qf_1, qf_2 = qf_1.squeeze(), qf_2.squeeze()
            # qf_loss = F.mse_loss(qf_1 + qf_2, reward_batch)   
            qf_loss = 0.5 * F.mse_loss(2 * qf_1, reward_batch) + 0.5 * F.mse_loss(2 * qf_2, reward_batch)   
            
            (qf_loss / self.accumulate_grads).backward()

            return qf_loss
        
        return train_step


    @torch.no_grad()
    def select_action_batch(self, state: TextMemory, a_embeds: Tensor,  a_embeds_target: Tensor, evaluate=False, random=False):
        
        torch_state = TextMemory(
            item_ids=None,
            available_ids=None,
            available_mask=state.available_mask,
            text=None,
            input_ids=state.input_ids,
            attention_mask=state.attention_mask,
        )
        action, q_values, q_values_target = policy_apply(self.policy, self.v_net_target, torch_state, a_embeds,  a_embeds_target, torch.tensor(self.alpha), evaluate)

        if random:
            action = self.random_policy.forward(state)
            
        return action.squeeze(), q_values.squeeze(), q_values_target.squeeze()


    @torch.no_grad()
    def select_action(self, state: TextMemory, a_embeds: Tensor, a_embeds_target: Tensor, evaluate=False, random=False):

        state = stack_memory([state], self.critic.action_embed.tokenizer, max_length=self.action_embed_length)
        a_embeds = custom_pad_sequence([a_embeds], padding_value=0.0, batch_first=True, pad_to_power_2=False)
        a_embeds_target = custom_pad_sequence([a_embeds_target], padding_value=0.0, batch_first=True, pad_to_power_2=False)
                
        action, q_values, q_values_target =  self.select_action_batch(state, a_embeds, a_embeds_target, evaluate, random)

        return action.item(), q_values, q_values_target
        
    
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
        
        
        last_q = mask_batch[:, -2] * q_values_batch[:, -1]
        lambda_returns = reward_batch[:, -2] + self.gamma * last_q

        targets = [lambda_returns]

        for t in range(q_values_batch.shape[1] - 3, -1, -1):
            lambda_returns, last_q = self._get_target(lambda_returns, last_q, q_values_batch[:, t], reward_batch[:, t], mask_batch[:, t])
            targets.append(lambda_returns)

        targets.reverse()
        targets = torch.stack(targets, dim=1)
        assert targets.shape[0] == q_values_batch.shape[0]
        assert targets.shape[1] == q_values_batch.shape[1] - 1
        targets = targets.reshape(-1)

        state_batch = TextMemory(
                item_ids=None,
                available_ids=None,
                available_mask=state_batch.available_mask,
                text=None,
                input_ids=state_batch.input_ids,
                attention_mask=state_batch.attention_mask
            )
        
        action_batch = TextMemoryItem(
            index=None,
            position=torch.tensor(action_batch.position, device=action_batch.input_ids.device, dtype=torch.float32), 
            input_ids=action_batch.input_ids,
            attention_mask=action_batch.attention_mask,
            text=None
        )
        
        qf_loss = self.train_step(self.critic, state_batch, action_batch, targets) #computes backward inside
        
        self._update_step += 1
        if self._update_step % self.accumulate_grads == 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optim.step()
            self.scheduler.step()
            self.critic_optim.zero_grad()

            self.alpha = self.alpha_start * float(self.scheduler.get_lr()[0]) / self.start_lr
            self.v_net_target.update(self.critic, self.tau)
            self.action_embed_target.update(self.critic, self.tau)
            self.policy.update(self.critic)

        return qf_loss.item()
    
    def train(self):
        self.policy.train()
        self.critic.train()

    def eval(self):
        self.policy.eval()
        self.critic.action_embed.eval()
        # self.v_net_target.train()
        # self.action_embed_target.train()

    def save(self, checkpoint_path: str, verbose=False) -> None:
        """
        Save state of all networks, optimizer and scheduler into a single file

        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            "critic": self.critic.state_dict(),
            "policy": self.policy.state_dict(),
            "random_policy": self.random_policy.state_dict(),
            "v_net_target": self.v_net_target.state_dict(),
            "action_embed_target": self.action_embed_target.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "alpha": self.alpha, #changes in training phase
        }
        torch.save(checkpoint, checkpoint_path)
        if verbose:
            print(f"[INFO] PQN checkpoint saved → {checkpoint_path}")

    def load(self, checkpoint_path: str, strict: bool = True, verbose=False) -> None:
        """
        load network state_dict from checkpoint
        `strict` goes to `load_state_dict`.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=torch.get_default_device(), weights_only=False)

        self.critic.load_state_dict(checkpoint["critic"], strict=strict)
        self.policy.load_state_dict(checkpoint["policy"], strict=strict)
        # у random_policy обычно нет параметров, но на всякий случай
        if "random_policy" in checkpoint:
            self.random_policy.load_state_dict(checkpoint["random_policy"], strict=False)
        self.v_net_target.load_state_dict(checkpoint["v_net_target"], strict=strict)
        self.action_embed_target.load_state_dict(checkpoint["action_embed_target"], strict=strict)

        # при наличии — восстанавливаем оптимизатор и scheduler
        if "critic_optim" in checkpoint:
            self.critic_optim.load_state_dict(checkpoint["critic_optim"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        # восстанавливаем значение α, если оно было сохранено
        self.alpha = checkpoint.get("alpha", self.alpha)

        print(f"[INFO] PQN checkpoint loaded  ← {checkpoint_path}")


class PQNActor:

    def __init__(self, agent: PQN):
        self.agent = agent
        self.embeds = []
        self.embeds_target = []

    @torch.no_grad()
    def get_embeds(self, all_texts, positions) -> Tuple[Tensor, Tensor]:
        tokenizer = self.agent.action_tokenizer
        embedder = self.agent.critic.action_embed
        embedder_target = self.agent.action_embed_target

        batch = stack_text_list(list(all_texts), tokenizer)
        positions = torch.tensor(positions, device=torch.get_default_device())
        embeds, embeds_target = embedder(**batch, positions=positions), embedder_target(**batch, positions=positions)

        return embeds, embeds_target
    
    @torch.no_grad()
    def update_embeds(self, k, positions):
        positions = torch.tensor(positions, device=torch.get_default_device())
        embedder = self.agent.critic.action_embed
        embedder_target = self.agent.action_embed_target
        self.embeds[k] = embedder.update_pos(self.embeds[k], positions=positions)
        self.embeds_target[k] = embedder_target.update_pos(self.embeds_target[k], positions=positions)
        
    def step(self, s_seq, chunks, positions, is_random):
        for k, ch, pos in zip(range(len(chunks)), chunks, positions):
            if ch is not None:
                self.embeds[k], self.embeds_target[k] = self.get_embeds(ch, pos)
            if pos is not None:
                self.update_embeds(k, pos)

        s_par = stack_memory(s_seq, self.state_tokenizer)
        
        a_embeds_pos = [emb["rope"] for emb in self.embeds]
        a_embeds_target_pos = [emb["rope"] for emb in self.embeds_target]
             
        embeds_pt = custom_pad_sequence(a_embeds_pos, padding_value=0.0, batch_first=True, pad_to_power_2=False)
        embeds_target_pt = custom_pad_sequence(a_embeds_target_pos, padding_value=0.0, batch_first=True, pad_to_power_2=False)
        
        action, _, q_values  = self.agent.select_action_batch(s_par, embeds_pt, embeds_target_pt, random=is_random)
        
        return action, q_values
            
        




