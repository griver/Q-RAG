import sys
import os

repo_dir = os.path.dirname(os.path.abspath("./"))
if repo_dir not in sys.path:
    print(f'add repository dir: {repo_dir}')
    sys.path.append(repo_dir)

from torch.utils.tensorboard import SummaryWriter
import torch
import sys
from rl.pqn import PQN
import numpy as np
from rl.babilong_env import BabilongEnv
from rl.text_env import ParallelTextEnv, MAX_TOKEN_LENGTH
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
    a_embeds_t, a_embeds_target_t = env_test.get_extra_embeds(agent.action_tokenizer, agent.critic.action_embed, agent.action_embed_target)
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

    
cfg: DictConfig = load_config(name="training")
agent_config: DictConfig = cfg.algo
env_config: DictConfig = cfg.envs

writer: SummaryWriter = instantiate(cfg.logger.tensorboard)

torch.set_default_device(cfg.device)
torch.set_float32_matmul_precision('high')
set_all_seeds(cfg.seed)

MAX_TOKEN_LENGTH["state"] = cfg.max_state_length
MAX_TOKEN_LENGTH["action"] = cfg.max_action_length

agent = PQN(agent_config)

env: BabilongEnv = instantiate(env_config.env)
env_test: BabilongEnv = instantiate(env_config.test_env)
parallel_env = ParallelTextEnv(
    [env] + [env.copy() for _ in range(cfg.envs_parallel - 1)], 
    state_tokenizer=agent.state_tokenizer,
    action_tokenizer=agent.action_tokenizer)

progress_bar = tqdm(range(cfg.steps_count), desc="Training")

states_list, _ = parallel_env.reset()
step = 0
train_rewards = []

for it in progress_bar:
    
    agent.train()
    states_list, rewards, train_batch = parallel_env.rollout(cfg.batch_size, states_list, agent, random=(step < 2 * cfg.learning_start))
    step += train_batch.reward.shape[0]
    train_rewards.extend(rewards)

    qf_loss = agent.update(
        train_batch.state, 
        train_batch.action, 
        train_batch.next_state, 
        train_batch.q_values, 
        train_batch.reward, 
        train_batch.not_done)
    
    if it % cfg.eval_interval == 0:

        agent.eval()
        
        writer.add_scalar("train r_sum", np.mean(train_rewards), step)
        writer.add_scalar("qf_loss", qf_loss, step)

        r_eval = [evaluate(env_test, agent) for _ in range(cfg.eval_episodes)]
        writer.add_scalar("eval r_sum", np.mean(r_eval), step)

        progress_bar.set_postfix({
            'reward': np.mean(train_rewards),
            "eval_reward": np.mean(r_eval),
            'qf_loss': qf_loss,
            'step': step,
        })

        train_rewards = []

