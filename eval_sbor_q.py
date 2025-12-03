import os
import sys
import argparse
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torchvision.ops.misc import interpolate
from tqdm import tqdm
import json

# ---- add repository root to PYTHONPATH (so that rl.* modules resolve) ---- #
repo_dir = os.path.dirname(os.path.abspath("./"))
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from rl.agents.pqn import PQN  # noqa: E402
from envs.qa_env import QAEnv  # noqa: E402



class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器处理NumPy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Seed everything (Python, NumPy, PyTorch, CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def prepare_eval_config(eval_cfg, train_cfg):
#     """Compute additional complex value changes"""
#     # Override only test‑environment‑specific fields so that
#     # the training configuration remains untouched.
#     max_chunks_count = eval_cfg.envs.get("max_chunks_count", None)
#     max_seq_len = eval_cfg.algo.model.predictor.get("max_seq_len", None)
#     interpolate_factor = eval_cfg.algo.model.predictor.get("interpolate_factor", None)
#
#     assert max_chunks_count == max_seq_len == interpolate_factor == None, \
#         'If you specified at least one of the [envs.max_chunks_count, algo.model.predictor.max_seq_len, algo.model.predictor.interpolate_factor] then you should also specify others'
#
#     eval_max_chunks = eval_cfg.envs.num_sentences
#     if train_cfg.index_type == 'random':
#             train_max_chunks = train_cfg.envs.max_chunks_count
#     elif train_cfg.index_type == 'absolute':
#             train_max_chunks = train_cfg.envs.num_sentences
#
#     if eval_max_chunks > train_max_chunks:
#         eval_cfg.envs.max_chunks_count = eval_max_chunks + 1
#         eval_cfg.algo.model.predictor.max_seq_len = max(eval_max_chunks + 1, train_cfg.algo.model.predictor.max_seq_len)
#         eval_cfg.algo.model.predictor.interpolate_factor = eval_max_chunks / train_max_chunks
#         print(f'Current indexing type is {train_cfg.index_type}')
#         print(
#             "The following parameters are updated:",
#             f"...eval_cfg.envs.max_chunks_count={eval_cfg.envs.max_chunks_count}",
#             f"...eval_cfg.algo.model.predictor.max_seq_len={eval_cfg.algo.model.predictor.max_seq_len}",
#             f"...eval_cfg.algo.model.predictor.interpolate_factor={eval_cfg.algo.model.predictor.interpolate_factor}",
#             sep='\n')
#
#     return eval_cfg

def calc_fact_f1_em(predicted_support_idxs, gt_support_idxs, total_elements):
    # Taken from hotpot_eval
    pred_sf = set(map(int, predicted_support_idxs))
    gt_sf = set(map(int, gt_support_idxs))
    tp, fp, fn, tn = 0, 0, 0, 0
    for e in pred_sf:
        if e in gt_sf:
            tp += 1
        else:
            fp += 1
    for e in gt_sf:
        if e not in pred_sf:
            fn += 1

    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if gt_sf.issubset(pred_sf) else 0.0

    tn = total_elements - len(gt_sf) - fp
    tpr = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    fpr = 1.0 * fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # In case everything is empty, set both f1, em to be 1.0.
    # Without this change, em gets 1 and f1 gets 0
    if not pred_sf and not gt_sf:
        f1, em = 1.0, 1.0
    return f1, em, tpr, fpr


#Q_BORDER = 0.0

@torch.no_grad()
def evaluate_episode(env: QAEnv, agent: PQN, q_border: float, sample=None, total_elements=10) -> float:
    """Run a single episode and return the cumulative reward."""
    if sample is None:
        state = env.reset()
    else:
        state = env.reset(sample)
    text_len = env.get_sample_len(agent.action_tokenizer)
    done = False

    # Pre‑compute static embeddings that do not change during an episode
    embeds, embeds_target = env.get_extra_embeds(
        agent.action_tokenizer,
        agent.critic.action_embed,
        agent.action_embed_target,
    )
    episode_return = 0.0
    episode_len = 0
    actions = []
    q_values = []

    while not done:

        embeds = env.update_embeds(embeds, agent.critic.action_embed)
        embeds_target = env.update_embeds(embeds_target, agent.action_embed_target)

        action, qval, _ = agent.select_action(
            state,
            embeds["rope"], embeds_target["rope"],
            random=False,
            evaluate=True,
        )
        state, _, reward, done = env.step(action)
        episode_return += reward
        qval_max = qval.max()
        q_values.append(qval_max)

        done = done or (qval_max < q_border)

        if qval_max > q_border:
            episode_len += 1
            actions.append(action)
    # print('\n')
    # print(q_values)

    pred_sf = [int(i) for i in actions]
    gt_sf = list(env.references_idx)
    f1, em, tpr, fpr =  calc_fact_f1_em(pred_sf, gt_sf, total_elements=total_elements)

    return {
        'return':episode_return,
        'text_len':text_len,
        "episode_len": episode_len,
        'f1': f1,
        'em': em,
        'tpr': tpr, 
        'fpr': fpr
    }

@torch.no_grad()
def collect_episode_stats(env: QAEnv, agent: PQN, sample=None) -> dict:
    """Прогоняет эпизод до конца (max steps) и возвращает сырые данные."""
    if sample is None:
        state = env.reset()
    else:
        state = env.reset(sample)
    
    # Pre-compute static embeddings
    embeds, embeds_target = env.get_extra_embeds(
        agent.action_tokenizer, agent.critic.action_embed, agent.action_embed_target,
    )
    
    actions = []
    q_values = []
    rewards = []
    
    done = False
    while not done:
        embeds = env.update_embeds(embeds, agent.critic.action_embed)
        embeds_target = env.update_embeds(embeds_target, agent.action_embed_target)

        # Force evaluate=True, random=False
        action, qval, _ = agent.select_action(
            state, embeds["rope"], embeds_target["rope"], random=False, evaluate=True
        )
        
        state, _, r, done = env.step(action)
        
        # Сохраняем сырые данные (конвертируем в float/int для JSON)
        q_values.append(float(qval.max().cpu().numpy()))
        actions.append(int(action))
        rewards.append(float(r))

    # Возвращаем структуру для сохранения
    return {
        "rewards": rewards,
        "q_values": q_values,          # Список Q-значений для каждого шага
        "pred_idx": actions,            # Список выбранных индексов чанков
        "sf_idx": list(env.references_idx), # Список правильных чанков (Ground Truth)
        #"total_elements": 20           # Общее кол-во чанков
    }


def load_eval_config(name):
    cli_cfg = OmegaConf.from_cli()
    eval_cfg = OmegaConf.load(name)
    # eval_cfg = OmegaConf.merge(eval_cfg, cli_cfg)

    train_cfg_path = os.path.join(eval_cfg.pretrained_path, 'config.yaml')
    if not os.path.exists(train_cfg_path):
        raise FileNotFoundError(f"Could not find config.yaml at {train_cfg_path}")
    train_cfg = OmegaConf.load(train_cfg_path)
    #prepare_eval_config(eval_cfg, train_cfg)
    cfg = OmegaConf.merge(train_cfg, eval_cfg, cli_cfg)
    OmegaConf.resolve(cfg)
    return cfg

def main(argv: List[str] | None = None) -> None:
    cfg = load_eval_config("configs/testing.yaml")
    # Set global MAX_TOKEN_LENGTH constants before tokenisers are built
    # MAX_TOKEN_LENGTH["state"] = cfg.max_state_length
    # MAX_TOKEN_LENGTH["action"] = cfg.max_action_length

    set_all_seeds(cfg.seed)

    # Respect the device stored in the training config; fall back to CPU if absent
    print("device", getattr(cfg, "device", "cpu"))
    torch.set_default_device(getattr(cfg, "device", "cpu"))
    torch.set_float32_matmul_precision("high")

    # -----------------------------------------------------------------------
    # Build agent & load checkpoint
    # -----------------------------------------------------------------------
    agent = PQN(cfg.algo)

    ckpt_filename = "model_last.pt" if cfg.use_last else "model_best.pt"
    ckpt_path = os.path.join(cfg.pretrained_path, ckpt_filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    agent.load(ckpt_path, strict=True)
    agent.eval()

    # print('Проверка на OOD (обучались на hotpotqa, эвал на musique)')
    # test_env = cfg.envs.test_env
    # test_env['dataset']['dataset']['_target_'] = 'envs.dataloaders.musique.RetrievalMusique'
    # test_env['dataset']['dataset']['path'] = '/trinity/home/a.anokhin/rmt_other_datasets/data/dataloaders/data_sources/musique'
    # env_test: QAEnv = instantiate(test_env)

    env_test: QAEnv = instantiate(cfg.envs.test_env)

    #Сбор данных (Inference)
    # ==========================================
    jsonl_file = cfg.pretrained_path + "/episode_logs_q.jsonl"
    print(f"Collecting stats into {jsonl_file}...")

    with open(jsonl_file, "w") as f_out:
        for i in tqdm(range(cfg.num_samples), desc="Inference", ncols=80):
            sample = env_test.dataset[i]
            # Запускаем БЕЗ q_border, собираем полный эпизод
            res = collect_episode_stats(env_test, agent, sample=sample)
            # stats["id"] = i
            # # stats["pred_texts"] = sample["chunks"]
            # stats["question"] = sample["question"]
            # stats["answer"] = sample["answer"]

            entry = {
                "id": sample["id"],
                "question": sample["question"],
                "answer": sample["answer"],
                "sf_idx": [int(idx) for idx in sample["sf_idx"]],
                "pred_idx": res["pred_idx"],
                "sf_texts": [sample["chunks"][idx] for idx in sample["sf_idx"]],
                "pred_texts": [sample["chunks"][idx] for idx in res["pred_idx"]],
                "q_values": res["q_values"],
                "return": 0,
                "text_len": 10,
                "f1": 0,
                "em": 0,
            }
            f_out.write(json.dumps(entry, cls=NumpyEncoder) + "\n")


if __name__ == "__main__":
    main()