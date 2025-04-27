import sys
import os

repo_dir = os.path.dirname(os.path.abspath("./"))
if repo_dir not in sys.path:
    print(f'add repository dir: {repo_dir}')
    sys.path.append(repo_dir)


from babilong_fix import QA2FixWrapper
from rl.retrieval_babilong import RetrNoiseInjectionDataset, RetrSentenceSampler
from babilong_utils import TaskDataset
from torch.utils.tensorboard import SummaryWriter
import datasets
from datasets import Dataset, load_dataset, load_from_disk
import torch
import sys
from rl.pqn import PQN
import time
import numpy as np
from collections import deque
from rl.babilong_env import BabilongEnv
from rl.text_env import ParallelTextEnv
from rl.sacd import SAC, SACArgs
from rl.text_env import TextReplayBuffer
from transformers import AutoModel, AutoTokenizer
from rl.bert_predictor import BertPredictor
from rl.text_env import TextEnv, TextReplayBuffer
from tqdm import tqdm
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra import initialize, compose
import random


@torch.no_grad()
def evaluate(env_test, agent):
    s_t = env_test.reset()
    done_t = False
    a_embeds_t, a_embeds_target_t = env_test.get_extra_embeds(agent.critic.action_embed, agent.action_embed_target)
    r_sum_t = 0
    while not done_t:
        action_t, _, _ = agent.select_action(s_t, a_embeds_t, a_embeds_target_t, random=False, evaluate=True)
        s_t, _, reward_t, done_t = env_test.step(action_t)
        r_sum_t += reward_t
    
    return r_sum_t


def load_config(name, overrides=None):
    with initialize(version_base="1.3", config_path="./configs"):
        cfg = compose(
            config_name=name,
            overrides=overrides if overrides else []
        )
        OmegaConf.resolve(cfg)
        return cfg


def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

    
cfg = load_config(name="training")
agent_config = cfg.algo
env_config = cfg.envs

writer = instantiate(cfg.logger.tensorboard)

torch.set_default_device(cfg.device)
torch.set_float32_matmul_precision('high')
set_all_seeds(cfg.seed)

agent = PQN(agent_config)

tokenizer = agent.critic.state_embed.tokenizer
env: BabilongEnv = instantiate(env_config.env, embed_tokenizer=tokenizer)
parallel_env = ParallelTextEnv([env, env.copy()], tokenizer, tokenizer)
env_test: BabilongEnv = instantiate(env_config.test_env, embed_tokenizer=tokenizer)

R = []
ep_number = 0
env_steps = 0
qf_loss = 0.0
progress_bar = tqdm(range(cfg.steps_count), desc="Training")
print("parallel envs count", len(parallel_env.text_envs))


# buffer = TextReplayBuffer(cfg.buffer_size, tokenizer=tokenizer)
s_seq, _ = parallel_env.reset()
step = 0

for it in progress_bar:
    
    # if step == 0 or done:

    agent.v_net_target.update(agent.critic, agent.tau)
    agent.action_embed_target.update(agent.critic, agent.tau)
    agent.policy.update(agent.critic)
    agent.policy.train()
    agent.critic.train()
    # agent.v_net_target.train()
    # agent.action_embed_target.train()

    s_seq, rewards, train_batch = parallel_env.rollout(cfg.batch_size, s_seq, agent, random=(step < 2 * cfg.learning_start))
    step += train_batch.reward.shape[0]
    R.extend(rewards)

    #     # s = env.reset()
    #     s_seq, s_par = parallel_env.reset()
    #     prev_dones = np.asarray([False] * len(parallel_env.text_envs))
    #     episodes = [[] for _ in range(len(parallel_env.text_envs))]
        
    #     a_embeds, a_embeds_target = parallel_env.get_extra_embeds(agent.critic.action_embed, agent.action_embed_target)

    # action, _, q_values  = agent.select_action_batch(s_par, a_embeds, a_embeds_target, random=(step < (cfg.learning_start * 2) // len(parallel_env.text_envs)))
    # action = action.cpu().numpy().reshape(-1)
    # q_values = q_values.cpu().numpy().reshape(-1)
    # # s_next, a_data, reward, done = env.step(action)
    # s_seq_next, a_seq, r_seq, done_seq, s_par_next  = parallel_env.step(action)
    # for episode, si, ai_data, si_next, ri, di, qi, di_prev in zip(episodes, s_seq, a_seq, s_seq_next, r_seq, done_seq, q_values, prev_dones):
    #     if not di_prev:
    #         episode.append((si, ai_data, si_next, ai_data, ri, di, qi))
    
    # s = s_next
    # r_sum += reward
    # s_seq = s_seq_next
    # s_par = s_par_next
    # prev_dones = prev_dones | done_seq
    # done = np.all(np.asarray(prev_dones))

    # if done:
    #     r_sum = 0.0
    #     for e in episodes:
    #         for tranz in e:
    #             buffer.add(*tranz)
    #             r_sum += tranz[4]
    #             env_steps += 1
        
    #     R.append(r_sum / len(parallel_env.text_envs))
    #     ep_number += len(parallel_env.text_envs)

    # if step > cfg.learning_start // len(parallel_env.text_envs) and env_steps >= 32:
    #     env_steps = 0
    #     for _ in range(1):
            # s_batch, a_batch, next_s_batch, _, r_batch, not_done_batch, q_batch = buffer.ordered_sample(cfg.batch_size)
    qf_loss = agent.update(
        train_batch.state, 
        train_batch.action, 
        train_batch.next_state, 
        train_batch.q_values, 
        train_batch.reward, 
        train_batch.not_done)
    
    if it % cfg.eval_interval == 0:
        
        writer.add_scalar("R", np.mean(R), step)
        writer.add_scalar("qf_loss", qf_loss, step)

        agent.policy.eval()
        agent.critic.action_embed.eval()

        r_eval = [evaluate(env_test, agent) for _ in range(cfg.eval_episodes)]
        writer.add_scalar("eval r_sum", np.mean(r_eval), step)

        progress_bar.set_postfix({
            'reward': np.mean(R),
            "eval_reward": np.mean(r_eval),
            'qf_loss': qf_loss,
            "alpha": agent.alpha,
            'step': step,
        })

        R = []

