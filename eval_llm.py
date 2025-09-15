import os
import argparse
import json
import re
from typing import List
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

from prompts_and_metrics.babilong import (
    #DEFAULT_PROMPTS,
    #TEMPLATE,
    #get_formatted_input,
    BabilongExactMatch,
    BabilongF1,
    BabilongPromptFormatter,
)


def save_final_scores(results_path: str, ns_key: str, f1: float, em: float) -> None:
    """Save F1 and EM scores under the provided key into results file."""
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}

    results[ns_key] = {"f1": f1, "em": em}

    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# def prepare_messages(question: str, facts: List[str], prompt_cfg: dict, user_template: str):
#     """Create chat messages for the language model."""
#     str_of_facts = " ".join(facts)
#     input_text = get_formatted_input(
#         str_of_facts,
#         question,
#         prompt_cfg["examples"],
#         prompt_cfg["instruction"],
#         prompt_cfg["post_prompt"],
#         template=user_template,
#     )
#     messages = [
#         {
#             "role": "system",
#             "content": "Your are an AI assistant, your job is to answer questions given to you by the user.",
#         },
#         {"role": "user", "content": input_text},
#     ]
#     return messages


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on retriever logs")
    parser.add_argument("retriever_logfile", help="Path to retriever log file")
    parser.add_argument("--llm_name", required=True, help="Name of LLM to load")
    parser.add_argument("--babi_task", default=None, help="Babi task name")
    parser.add_argument("--max_tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument('--gpu_util', type=float, default=0.3, help="Max gpu memory utilization. Default: 0.3")
    parser.add_argument('--think', action="store_true", default=False, help='enable_thinking for Qwen3 models.')
    args = parser.parse_args()

    chat_template_kwargs = dict(
        add_generation_prompt=False, 
        tokenize=False
    )
    if "Qwen3" in args.llm_name:
        chat_template_kwargs['enable_thinking'] = args.think
        print(f'set enable_thinking = {args.think} for {args.llm_name}')
    
    # prompt_cfg = {
    #     "instruction": DEFAULT_PROMPTS[args.babi_task]["instruction"],
    #     "examples": DEFAULT_PROMPTS[args.babi_task]["examples"],
    #     "post_prompt": DEFAULT_PROMPTS[args.babi_task]["post_prompt"],
    #     "template": TEMPLATE,
    # }
    f1_metric = BabilongF1(args.babi_task)
    em_metric = BabilongExactMatch()
    prepare_messages = BabilongPromptFormatter(args.babi_task)

    vllm_config = {
        'gpu_memory_utilization': args.gpu_util,
        'max_model_len': 2048,
        'dtype': 'bfloat16', #new values start here
        'quantization': None,
        'tensor_parallel_size': 1,
        'trust_remote_code': True,
    }
    sampling_params = {
        'max_tokens': args.max_tokens,
        'temperature': 0.0,
        "stop": None,
        'top_p': 0.95
    }

    llm = LLM(model=args.llm_name, **vllm_config)
    sampling_params = SamplingParams(**sampling_params)

    out_path = os.path.join(
        os.path.dirname(args.retriever_logfile),
        f"{os.path.basename(args.llm_name)}_{os.path.basename(args.retriever_logfile)}",
    )

    all_f1 = []
    all_em = []
    #max_samples = 100
    with open(args.retriever_logfile, "r") as f_in:
        lines = [ln for ln in f_in if ln.strip()]

    with open(out_path, "w") as f_out:
        for i in tqdm(range(len(lines)), desc="LLM eval", ncols=80):
            # if i >= max_samples:
            #     break
            line = lines[i]
            item = json.loads(line)
            question = item["question"]
            answer = item["answer"]
            facts_idx = item["pred_idx"]
            facts = item.get("pred_text", [])
            facts_sorted = [f for idx, f in sorted(zip(facts_idx, facts))]

            #messages = prepare_messages(question, facts_sorted, prompt_cfg, prompt_cfg["template"])
            messages = prepare_messages(question, facts_sorted)
            prompt = llm.get_tokenizer().apply_chat_template(messages, **chat_template_kwargs)
            #print('Messages:', messages)
            #print('Prompt:', prompt)
            #break
            outputs = llm.generate([prompt], sampling_params)
            prediction = outputs[0].outputs[0].text.strip()

            ans_f1 = f1_metric(prediction, answer)
            ans_em = em_metric(prediction, answer)
            # ans_f1 = compute_f1(prediction, answer)
            # ans_em = compute_exact_match(prediction, answer)

            all_f1.append(ans_f1)
            all_em.append(ans_em)

            item.update({
                "prediction": prediction,
                "answer_f1": ans_f1,
                "answer_em": ans_em,
                'pred_idx': sorted(facts_idx),
                'pred_text': facts_sorted,
            })
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()

    final_f1 = sum(all_f1) / len(all_f1)
    final_em = sum(all_em) / len(all_em)

    print(f"Saved results to {out_path}. F1: {final_f1:.3f}, EM: {final_em:.3f}")

    ns_match = re.search(r"ns(\d+)", os.path.basename(args.retriever_logfile))
    ns_key = ns_match.group(1) if ns_match else "unknown"

    results_path = os.path.join(
        os.path.dirname(args.retriever_logfile),
        f"results_{os.path.basename(args.llm_name)}",
    )
    save_final_scores(results_path, ns_key, final_f1, final_em)


if __name__ == "__main__":
    main()