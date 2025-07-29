import os
import argparse
import json
from typing import List
from tqdm import tqdm
from vllm import LLM, SamplingParams

from prompts_and_metrics.babilong import (
    DEFAULT_PROMPTS,
    TEMPLATE,
    get_formatted_input,
    compute_exact_match,
    gen_f1_metric,
)


def prepare_messages(question: str, facts: List[str], prompt_cfg: dict, user_template: str):
    """Create chat messages for the language model."""
    str_of_facts = " ".join(facts)
    input_text = get_formatted_input(
        str_of_facts,
        question,
        prompt_cfg["examples"],
        prompt_cfg["instruction"],
        prompt_cfg["post_prompt"],
        template=user_template,
    )
    messages = [
        {
            "role": "system",
            "content": "Your are an AI assistant, your job is to answer questions given to you by the user.",
        },
        {"role": "user", "content": input_text},
    ]
    return messages


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on retriever logs")
    parser.add_argument("retriever_logfile", help="Path to retriever log file")
    parser.add_argument("--llm_name", required=True, help="Name of LLM to load")
    parser.add_argument("--babi_task", default=None, help="Babi task name")
    parser.add_argument("--max_tokens", type=int, default=32, help="Max tokens to generate")
    args = parser.parse_args()

    prompt_cfg = {
        "instruction": DEFAULT_PROMPTS[args.babi_task]["instruction"],
        "examples": DEFAULT_PROMPTS[args.babi_task]["examples"],
        "post_prompt": DEFAULT_PROMPTS[args.babi_task]["post_prompt"],
        "template": TEMPLATE,
    }
    compute_f1 = gen_f1_metric(args.babi_task)

    llm = LLM(model=args.llm_name, gpu_memory_utilization=0.2)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.3)

    out_path = os.path.join(
        os.path.dirname(args.retriever_logfile),
        f"{os.path.basename(args.llm_name)}_{os.path.basename(args.retriever_logfile)}",
    )

    all_f1 = []
    all_em = []

    with open(args.retriever_logfile, "r") as f_in:
        lines = [ln for ln in f_in if ln.strip()]

    with open(out_path, "w") as f_out:
        for line in tqdm(lines, desc="LLM eval", ncols=80):
            item = json.loads(line)
            question = item["question"]
            answer = item["answer"]
            facts_idx = item["pred_idx"]
            facts = item.get("pred_text", [])
            facts_sorted = [f for idx, f in sorted(zip(facts_idx, facts))]

            messages = prepare_messages(question, facts_sorted, prompt_cfg, prompt_cfg["template"])
            prompt = llm.get_tokenizer().apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            outputs = llm.generate([prompt], sampling_params)
            prediction = outputs[0].outputs[0].text.strip()

            ans_f1 = compute_f1(prediction, answer)
            ans_em = compute_exact_match(prediction, answer)

            all_f1.append(ans_f1)
            all_em.append(ans_em)

            item.update({
                "prediction": prediction,
                "answer_f1": ans_f1,
                "answer_em": ans_em,
            })
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()

    print(
        f"Saved results to {out_path}. F1: {sum(all_f1)/len(all_f1):.3f}, EM: {sum(all_em)/len(all_em):.3f}"
    )


if __name__ == "__main__":
    main()