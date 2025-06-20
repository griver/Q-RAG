import os
import sys
import argparse
import random
from datetime import datetime
from typing import List
import json

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

# ---- add repository root to PYTHONPATH (so that rl.* modules resolve) ---- #
repo_dir = os.path.dirname(os.path.abspath("./"))
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from rl.pqn import PQN  # noqa: E402
from rl.babilong_env import BabilongEnv  # noqa: E402
from rl.text_env import MAX_TOKEN_LENGTH  # noqa: E402
from rl.qa_env import QARetrievalEnv


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


def prepare_config(cfg, max_steps: int):
    """Apply CLI overrides to the loaded config in‑place and return it."""
    # Override only test‑environment‑specific fields so that the training
    # configuration remains untouched.
    cfg.envs.test_env.max_steps = max_steps
    #cfg.envs.test_env.dataset.num_sentences = num_sentences
    return cfg

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
def evaluate_episode(env: BabilongEnv, agent: PQN) -> float:
    """Run a single episode and return the cumulative reward."""
    state = env.reset()
    text_len = env.get_sample_len(agent.action_tokenizer)
    done = False

    # Pre‑compute static embeddings that do not change during an episode
    extra_embeds = env.get_extra_embeds(
        agent.action_tokenizer,
        agent.critic.action_embed,
        agent.action_embed_target,
    )
    episode_return = 0.0

    while not done:
        action, _, _ = agent.select_action(
            state,
            *extra_embeds,
            random=False,
            evaluate=True,
        )
        state, _, reward, done = env.step(action)
        episode_return += reward

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---

    # 1. Получаем индексы предсказанных и правильных чанков
    pred_sf_idxs = [int(i) for i in state.item_ids]
    # gt_sf_idxs берем из env, как и раньше, это правильно
    gt_sf_idxs = list(env.references_idx)

    # 2. Считаем метрики F1 и EM
    f1, em =  calc_fact_f1_em(pred_sf_idxs, gt_sf_idxs)

    # 3. Получаем текст вопроса и тексты чанков из объекта env
    question_text = env.question
    all_chunks_text = env.sentences # Это np.array, но будет работать как список

    retrieved_chunks_text = [all_chunks_text[i] for i in pred_sf_idxs]
    ground_truth_chunks_text = [all_chunks_text[i] for i in gt_sf_idxs]

    # 4. Формируем полный словарь для возврата
    return {
        'question': question_text,
        'retrieved_chunks': retrieved_chunks_text,
        'ground_truth_chunks': ground_truth_chunks_text,
        'retrieved_indices': pred_sf_idxs,
        'ground_truth_indices': gt_sf_idxs,
        'return': episode_return,
        'text_len': text_len,
        'f1': f1,
        'em': em,
    }
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Evaluate a trained PQN agent on the Babilong environment.")
    parser.add_argument("savedir", type=str,
        help=("Directory containing the training artefacts (config.yaml, " 
              "model_best.pt, model_last.pt)."),
        )
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of evaluation episodes.")
    parser.add_argument("--max_steps", type=int, default=4, help="Max steps per episode (override).")
    parser.add_argument("--num_sentences", type=int, default=50, help="Number of sentences in a sample (override).",)
    parser.add_argument("--use_last",action="store_true", help="Load weights from model_last.pt instead of model_best.pt.",)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args(argv)

    # -----------------------------------------------------------------------
    # 1. Load config
    # -----------------------------------------------------------------------
    config_path = os.path.join(args.savedir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config.yaml at {config_path}")

    cfg = OmegaConf.load(config_path)
    #cfg = prepare_config(cfg, args.max_steps, args.num_sentences)
    cfg = prepare_config(cfg, args.max_steps)
    OmegaConf.resolve(cfg)

    # hydra.utils.instantiate needs the full objects, so keep OmegaConf.

    # Set global MAX_TOKEN_LENGTH constants before tokenisers are built
    MAX_TOKEN_LENGTH["state"] = cfg.max_state_length
    MAX_TOKEN_LENGTH["action"] = cfg.max_action_length

    # -----------------------------------------------------------------------
    # 2. Set seeds & device
    # -----------------------------------------------------------------------
    set_all_seeds(args.seed)

    # Respect the device stored in the training config; fall back to CPU if absent
    torch.set_default_device(getattr(cfg, "device", "cpu"))
    torch.set_float32_matmul_precision("high")

    # -----------------------------------------------------------------------
    # 3. Build agent & load checkpoint
    # -----------------------------------------------------------------------
    agent = PQN(cfg.algo)

    ckpt_filename = "model_last.pt" if args.use_last else "model_best.pt"
    ckpt_path = os.path.join(args.savedir, ckpt_filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    agent.load(ckpt_path, strict=True)
    agent.eval()

    # -----------------------------------------------------------------------
    # 4. Build test environment
    # -----------------------------------------------------------------------
    env_test: QARetrievalEnv = instantiate(cfg.envs.test_env)

    # -----------------------------------------------------------------------
    # 5. Evaluate
    # -----------------------------------------------------------------------
    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    evaluation_results = []
    for _ in tqdm(range(args.num_samples), desc="Evaluating", ncols=80):
        # evaluate_episode теперь возвращает полный словарь с данными
        episode_data = evaluate_episode(env_test, agent)
        evaluation_results.append(episode_data)

    # Сохраняем все собранные данные в JSON-файл
    output_filename = "evaluation_with_retrieved_chunks.json"
    output_path = os.path.join(args.savedir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
    
    print(f"\nSaved detailed evaluation results to: {output_path}")

    # Теперь считаем средние метрики из собранных данных для вывода в консоль
    if not evaluation_results:
        print("Evaluation finished, but no results were collected.")
        return

    returns = [res['return'] for res in evaluation_results]
    text_lens = [res['text_len'] for res in evaluation_results]
    all_em = [res['em'] for res in evaluation_results]
    all_f1 = [res['f1'] for res in evaluation_results]
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    fact_em = sum(all_em) / len(all_em)
    fact_f1 = sum(all_f1) / len(all_f1)

    print(
        f"Evaluated on {args.num_samples} episodes, max_retrieves={args.max_steps} | "
        f"Mean return: {mean_return:.3f} ± {std_return:.3f} (std) | "
        f"Mean text len: {np.mean(text_lens):.2f} | "
        f"EM: {fact_em:.3f} | F1: {fact_f1:.3f}"
    )


if __name__ == "__main__":
    main()