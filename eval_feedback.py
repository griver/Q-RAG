import argparse
import sys
import os
import json
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import compose, initialize

from rl.feedback.llm_feedback import AnswerMetricFeedback
from prompts_and_metrics.babilong import (
    BabilongPromptFormatter,
    BabilongExactMatch,
    BabilongF1,
)


def load_feedback_config(name):
    with initialize(version_base="1.3", config_path="./configs"):
        feedback_cfg = compose(
            config_name=name,
            overrides=sys.argv[1:]
        )

    OmegaConf.resolve(feedback_cfg)
    return feedback_cfg


def main():
    #example:
    #python eval_feedback.py +envs.task="qa1" +logfile=runs/Jul29_01-05-44_PQN_qa1_single-supporting-fact/eval_seed42_ns50.jsonl
    cfg = load_feedback_config('feedback/defaults.yaml')
    print("use_api:", cfg.feedback.use_api)
    feedback_model: AnswerMetricFeedback = instantiate(cfg.feedback.exact_match)
    print("use_api:", cfg.feedback.use_api)
    rewards = []
    out_f = open(cfg.output, "w") if "output" in cfg.keys() else None
    with open(cfg.logfile, "r") as f:
        lines = [ln for ln in f if ln.strip()]

    for line in tqdm(lines, desc="Feedback eval", ncols=80):
        item = json.loads(line)
        obs = {
            "question": item["question"],
            "sample_id": item.get("id"),
            "pred_idx": item.get("pred_idx", []),
            "pred_chunks": item.get("pred_text", []),
        }
        info = {"answer": item.get("answer")}
        feedback_model.reset(obs, info)
        fb_res = feedback_model.get_feedback(obs, info)
        reward = fb_res["reward"]
        rewards.append(reward)
        if out_f:
            out_item = dict(item)
            out_item["feedback_reward"] = reward
            out_f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            out_f.flush()

    if out_f:
        out_f.close()

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"Average Reward: {avg_reward:.3f}")


if __name__ == "__main__":
    main()