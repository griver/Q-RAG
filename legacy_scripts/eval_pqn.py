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
from hydra import compose, initialize
from torchvision.ops.misc import interpolate
from tqdm import tqdm

# ---- add repository root to PYTHONPATH (so that rl.* modules resolve) ---- #
repo_dir = os.path.dirname(os.path.abspath("../"))
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from rl.agents.pqn import PQN  # noqa: E402
from envs.qa_env import QAEnv  # noqa: E402


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

def calc_fact_f1_em(predicted_support_idxs, gt_support_idxs):
    # Taken from hotpot_eval
    pred_sf = set(map(int, predicted_support_idxs))
    gt_sf = set(map(int, gt_support_idxs))
    tp, fp, fn = 0, 0, 0
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

    # In case everything is empty, set both f1, em to be 1.0.
    # Without this change, em gets 1 and f1 gets 0
    if not pred_sf and not gt_sf:
        f1, em = 1.0, 1.0
    return f1, em


@torch.no_grad()
def evaluate_episode(env: QAEnv, agent: PQN, sample_id=None) -> float:
    """Run a single episode and return the cumulative reward."""
    if sample_id is None:
        state = env.reset()
    else:
        state = env.reset(env.dataset[sample_id])

    text_len = env.get_sample_len(agent.action_tokenizer)
    done = False

    # Pre‑compute static embeddings that do not change during an episode
    embeds, embeds_target = env.get_extra_embeds(
        agent.action_tokenizer,
        agent.critic.action_embed,
        agent.action_embed_target,
    )
    episode_return = 0.0

    while not done:

        embeds = env.update_embeds(embeds, agent.critic.action_embed)
        embeds_target = env.update_embeds(embeds_target, agent.action_embed_target)

        action, _, _ = agent.select_action(
            state,
            embeds["rope"], embeds_target["rope"],
            random=False,
            evaluate=True,
        )
        state, _, reward, done = env.step(action)
        episode_return += reward

    pred_sf = [int(i) for i in state.item_ids]
    gt_sf = list(env.references_idx)
    f1, em =  calc_fact_f1_em(pred_sf, gt_sf)

    return {
        'return':episode_return,
        'text_len':text_len,
        'f1': f1,
        'em': em,
    }


def load_eval_config(name):
    with initialize(version_base="1.3", config_path="../configs"):
        eval_cfg = compose(
            config_name=name,
            overrides=sys.argv[1:]
        )
        # cli_cfg = OmegaConf.from_cli()
        # eval_cfg = OmegaConf.load(name)
        # eval_cfg = OmegaConf.merge(eval_cfg, cli_cfg)

    train_cfg_path = os.path.join(eval_cfg.pretrained_path, 'config.yaml')
    if not os.path.exists(train_cfg_path):
        raise FileNotFoundError(f"Could not find config.yaml at {train_cfg_path}")
    train_cfg = OmegaConf.load(train_cfg_path)
    #prepare_eval_config(eval_cfg, train_cfg)
    cfg = OmegaConf.merge(train_cfg, eval_cfg)
    OmegaConf.resolve(cfg)
    return cfg

def main(argv: List[str] | None = None) -> None:
    cfg = load_eval_config("testing.yaml")
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

    env_test: QAEnv = instantiate(cfg.envs.test_env)

    max_samples = len(env_test.dataset)
    if not (0 < cfg.num_samples <= max_samples):
        print(f'set num samples from {cfg.num_samples} to {max_samples}')
        cfg.num_samples = max_samples
    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    returns = []
    text_lens = []
    all_em = []
    all_f1 = []

    for i in tqdm(range(cfg.num_samples), desc="Evaluating", ncols=80):
        sample_id = None if cfg.random_sampling else i
        res = evaluate_episode(env_test, agent, sample_id=sample_id)
        returns.append(res['return'])
        text_lens.append(res['text_len'])
        all_f1.append(res['f1'])
        all_em.append(res['em'])

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    fact_em = sum(all_em) / len(all_em)
    fact_f1 = sum(all_f1) / len(all_f1)

    print(
        f"Evaluated on {cfg.num_samples} episodes, max_retrieves={cfg.envs.max_steps} | "
        f"Mean return: {mean_return:.3f} ± {std_return:.3f} (std) | "
        f"Mean text len: {np.mean(text_lens):.2f} | "
        f"EM: {fact_em:.3f} | F1: {fact_f1:.3f}"
    )


if __name__ == "__main__":
    main()
