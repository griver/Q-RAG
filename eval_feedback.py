import argparse
import json
from tqdm import tqdm

from rl.feedback.llm_feedback import AnswerMetricFeedback
from prompts_and_metrics.babilong import (
    DEFAULT_PROMPTS,
    TEMPLATE,
    get_formatted_input,
    compute_exact_match,
    gen_f1_metric,
)


from eval_llm import prepare_messages


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval logs using ExactMatchFeedback")
    parser.add_argument("logfile", help="Path to JSON lines file with retrieval results")
    parser.add_argument("--answer_model_name", required=True, help="Model name for answer generation")
    parser.add_argument("--use_api", action="store_true", help="Use API instead of local vLLM")
    parser.add_argument("--api_base_url", default=None, help="Base URL for the API")
    parser.add_argument("--api_key", default="", help="API key for the model provider")
    parser.add_argument("--max_tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--gpu_util", type=float, default=0.8, help="GPU memory utilization for vLLM")
    parser.add_argument("--max_at_same_time", type=int, default=20, help="Max concurrent API requests")
    parser.add_argument("--output", default=None, help="Optional path to write per-sample rewards")
    args = parser.parse_args()
    args.babi_task = 'qa1'

    prompt_cfg = {
        "instruction": DEFAULT_PROMPTS[args.babi_task]["instruction"],
        "examples": DEFAULT_PROMPTS[args.babi_task]["examples"],
        "post_prompt": DEFAULT_PROMPTS[args.babi_task]["post_prompt"],
        "template": TEMPLATE,
    }

    vllm_config = {
        'gpu_memory_utilization': args.gpu_util,
        'max_model_len': 2048,
        'dtype': 'bfloat16',  # new values start here
        'quantization': None,
        'tensor_parallel_size': 1,
        'trust_remote_code': True,
    }
    sampling_params = {
        'max_tokens': args.max_tokens,
        'temperature': 0.0,
        'top_p': 0.95
    }

    feedback_model = AnswerMetricFeedback(
        use_api=args.use_api,
        answer_model_name=args.answer_model_name,
        sampling_params=sampling_params,
        vllm_config=vllm_config,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        max_at_same_time=args.max_at_same_time,
        metric=compute_exact_match,
        prepare_messages_func= lambda q, facts: prepare_messages(q, facts, prompt_cfg, prompt_cfg['template'])
    )

    rewards = []
    out_f = open(args.output, "w") if args.output else None
    with open(args.logfile, "r") as f:
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