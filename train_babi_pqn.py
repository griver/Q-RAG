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
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from hydra import initialize, compose
import random
from datetime import datetime


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
        cli_cfg = OmegaConf.from_cli()
        cfg = OmegaConf.merge(cfg, cli_cfg)
        #OmegaConf.resolve(cfg)
        cfg = prepare_config(cfg)
        return cfg

def prepare_config(cfg):
    """
    modifies config for parameters that should depend on each other
    """
    if cfg.logger.log_dir is not None:
        dir_name = datetime.now().strftime("%b%d_%H-%M-%S") + cfg.logger.tensorboard.comment
        cfg.logger.log_dir = os.path.join(cfg.logger.log_dir, dir_name)
        cfg.logger.tensorboard.log_dir = os.path.join(cfg.logger.log_dir, 'tb_logs/')

    # enumerate_facts = (cfg.positional_coding == 'enum') #TODO: add version that enumerate all chunks
    # cfg.envs.env.dataset.task_dataset.add_sentence_idx = enumerate_facts
    # cfg.envs.test_env.dataset.task_dataset.add_sentence_idx = enumerate_facts
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
os.makedirs(cfg.logger.log_dir, exist_ok=True)
config_save_path = os.path.join(cfg.logger.log_dir, "config.yaml")
OmegaConf.save(config=cfg, f=config_save_path, resolve=False)
print(f"[INFO] Training config saved to {config_save_path}")

# path to checkpoints and metric to determine the best model
ckpt_last_path = os.path.join(cfg.logger.log_dir, "model_last.pt")
ckpt_best_path = os.path.join(cfg.logger.log_dir, "model_best.pt")
best_eval_reward = -float("inf")


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

total_steps = cfg.steps_count * cfg.accumulate_grads
eval_interval = cfg.eval_interval * cfg.accumulate_grads
#assuming we don't need to scale cfg.learning_start with grad_accumulation
progress_bar = tqdm(range(total_steps), desc="Training")

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
    
    if it % eval_interval == 0:

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
        agent.save(ckpt_last_path)
        #torch.save(agent.state_dict(), ckpt_last_path)

        mean_eval_reward = np.mean(r_eval)
        if mean_eval_reward > best_eval_reward:
            best_eval_reward = mean_eval_reward
            agent.save(ckpt_best_path)
            #torch.save(agent.state_dict(), ckpt_best_path)
            #print(f"[INFO] New best model saved with reward {best_eval_reward:.3f}")

        train_rewards = []

