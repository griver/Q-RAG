import sys
import os

repo_dir = os.path.dirname(os.path.abspath("../"))
if repo_dir not in sys.path:
    print(f'add repository dir: {repo_dir}')
    sys.path.append(repo_dir)

from rl.pqn import PQN
import numpy as np
from rl.babilong_env import BabilongEnv
from rl.text_env import TextReplayBuffer
from tqdm import tqdm
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import initialize, compose
import random
from envs.dataloaders.babilong.retrieval_babilong import RetrievalBabiLong

print(RetrievalBabiLong)

@torch.no_grad()
def evaluate(env_test, agent):
    s = env_test.reset()
    done = False
    a_embeds = env_test.get_extra_embeds(agent.critic.action_embed)
    r_sum = 0
    while not done:
        action, _, _ = agent.select_action(s, a_embeds, random=False, evaluate=True)
        s, _, reward, done = env_test.step(action)
        r_sum += reward
    
    return r_sum


def load_config(name, overrides=None):
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name=name,
            overrides=overrides if overrides else []
        )
        OmegaConf.resolve(cfg)
        cfg = prepare_config(cfg)
        return cfg


def prepare_config(cfg):
    """
    modifies config for parameters that should depend on each other
    """
    enumerate_facts = (cfg.positional_coding == 'enum') #TODO: add version that enumerate all chunks
    cfg.envs.env.dataset.task_dataset.add_sentence_idx = enumerate_facts
    cfg.envs.test_env.dataset.task_dataset.add_sentence_idx = enumerate_facts
    return cfg

def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

    
cfg = load_config(name="training_pos")
#TODO: add shuffling between babi facts and noise!

agent_config = cfg.algo
env_config = cfg.envs

writer = instantiate(cfg.logger.tensorboard)

torch.set_default_device(cfg.device)
torch.set_float32_matmul_precision('high')
set_all_seeds(cfg.seed)

agent = PQN(agent_config)

tokenizer = agent.critic.state_embed.tokenizer
env: BabilongEnv = instantiate(env_config.env, embedder=agent.action_embed_target, embed_tokenizer=tokenizer)
env_test: BabilongEnv = instantiate(env_config.test_env, embedder=agent.action_embed_target, embed_tokenizer=tokenizer)

buffer = TextReplayBuffer(cfg.buffer_size, tokenizer=tokenizer)
R = []
ep_number = 0
progress_bar = tqdm(range(cfg.steps_count), desc="Training")

for step in progress_bar:
    
    if step == 0 or done:

        agent.v_net_target.update(agent.critic, agent.tau)
        agent.action_embed_target.update(agent.critic, agent.tau)
        agent.policy.update(agent.critic)
        agent.policy.train()
        agent.critic.train()

        s = env.reset()
        r_sum = 0

        a_embeds = env.get_extra_embeds(agent.critic.action_embed)


    action, _, q_values  = agent.select_action(s, a_embeds, random=(step < cfg.learning_start * 2))
    s_next, a_data, reward, done = env.step(action)
    buffer.add(s, a_data, s_next, a_data, reward, done, 0, q_values.max().cpu().item())
    
    s = s_next
    r_sum += reward
    
    if step > cfg.learning_start and step % cfg.update_every == 0:
        for _ in range(1):
            s_batch, a_batch, next_s_batch, _, r_batch, not_done_batch, entropy_batch, q_batch = buffer.ordered_sample(cfg.batch_size)
            qf_loss = agent.update(s_batch, a_batch, next_s_batch, q_batch, r_batch, not_done_batch)
    
    if done:
        R.append(r_sum)
        ep_number += 1
    
    if step % cfg.eval_interval == 0 and ep_number > 0:
        
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

