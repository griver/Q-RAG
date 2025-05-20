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
from tqdm import tqdm

# ---- add repository root to PYTHONPATH (so that rl.* modules resolve) ---- #
repo_dir = os.path.dirname(os.path.abspath("./"))
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from rl.pqn import PQN  # noqa: E402
from rl.babilong_env import BabilongEnv  # noqa: E402
from rl.text_env import MAX_TOKEN_LENGTH  # noqa: E402


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


def prepare_config(cfg, max_steps: int, num_sentences: int):
    """Apply CLI overrides to the loaded config in‑place and return it."""
    # Override only test‑environment‑specific fields so that the training
    # configuration remains untouched.
    cfg.envs.test_env.max_steps = max_steps
    cfg.envs.test_env.dataset.num_sentences = num_sentences
    return cfg


@torch.no_grad()
def evaluate_episode(env: BabilongEnv, agent: PQN) -> float:
    """Run a single episode and return the cumulative reward."""
    state = env.reset()
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

    return episode_return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PQN agent on the Babilong environment."
    )
    parser.add_argument(
        "savedir",
        type=str,
        help=(
            "Directory containing the training artefacts (config.yaml, "
            "model_best.pt, model_last.pt)."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of evaluation episodes.")
    parser.add_argument("--max_steps", type=int, default=6, help="Max steps per episode (override).")
    parser.add_argument(
        "--num_sentences",
        type=int,
        default=50,
        help="Number of sentences in a sample (override).",
    )
    parser.add_argument(
        "--use_last",
        action="store_true",
        help="Load weights from model_last.pt instead of model_best.pt.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args(argv)

    # -----------------------------------------------------------------------
    # 1. Load config
    # -----------------------------------------------------------------------
    config_path = os.path.join(args.savedir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config.yaml at {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg = prepare_config(cfg, args.max_steps, args.num_sentences)
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
    env_test: BabilongEnv = instantiate(cfg.envs.test_env)

    # -----------------------------------------------------------------------
    # 5. Evaluate
    # -----------------------------------------------------------------------
    returns = []
    for _ in tqdm(range(args.num_samples), desc="Evaluating", ncols=80):
        returns.append(evaluate_episode(env_test, agent))

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))

    print(
        f"Evaluated on {args.num_samples} episodes | "
        f"Mean return: {mean_return:.3f} ± {std_return:.3f} (std)"
    )


if __name__ == "__main__":
    main()
