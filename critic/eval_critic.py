"""
Evaluation with Trained Plan Critic
====================================
Drop-in replacement for your existing eval script.
Inserts the trained Critic between Planner and Agent Loop.

Usage:
    python eval_with_critic.py \
        --file_path data/hotpotqa_dev.jsonl \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --planner_base Qwen/Qwen2.5-7B-Instruct \
        --planner_lora ./qwen_planner_lora_v2/final \
        --critic_base  Qwen/Qwen2.5-7B-Instruct \
        --critic_lora  ./critic_lora_grpo/final \
        --output_file_path results_with_critic.json
"""

import json, os, re, string, argparse, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

# Import shared constants from training script
from critic_rl_train import (
    PLANNER_SYSTEM_PROMPT, HOP_SYSTEM_PROMPT, HOP_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_PROMPT,
    CRITIC_SYSTEM_PROMPT, CRITIC_USER_PROMPT,
    parse_steps, replace_placeholders, extract_final_answer,
    build_reasoning_chain, format_plan, parse_critic_output,
    normalize_answer, compute_f1, compute_em,
    MAX_RETRIES,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file_path",        type=str, required=True)
    p.add_argument("--model_name",       type=str, default="Qwen/QwQ-32B")
    p.add_argument("--planner_base",     type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--planner_lora",     type=str, default="./qwen_planner_lora_v2/final")
    p.add_argument("--critic_base",      type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--critic_lora",      type=str, default="./critic_lora_grpo/final")
    p.add_argument("--output_file_path", type=str, default=None)
    p.add_argument("--no_critic",        action="store_true",
                   help="Disable critic for ablation (baseline)")
    return p.parse_args()


@torch.no_grad()
def run_planner(question, model, tok, feedback=None):
    user_content = f"Decompose:\n{question}"
    if feedback:
        user_content += f"\n\nPrevious plan was rejected. Feedback: {feedback}\nPlease revise."
    msgs = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    out = model.generate(
        **enc, max_new_tokens=256, do_sample=False,
        temperature=1.0, pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_steps(text)


@torch.no_grad()
def run_critic(question, plan, model, tok):
    plan_text = format_plan(plan)
    user_content = CRITIC_USER_PROMPT.format(question=question, plan=plan_text)
    msgs = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    out = model.generate(
        **enc, max_new_tokens=128, do_sample=False,
        temperature=1.0, pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_critic_output(text)


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output_file_path or (
        os.path.splitext(args.file_path)[0] + "_eval_with_critic.json"
    )

    dataset = []
    with open(args.file_path, encoding="utf-8") as f:
        for line in f: dataset.append(json.loads(line))
    print(f"Samples: {len(dataset)}")

    # Load Planner
    print("[1/3] Loading Planner ...")
    planner_tok = AutoTokenizer.from_pretrained(args.planner_base, trust_remote_code=True)
    planner_tok.pad_token = planner_tok.eos_token
    planner_tok.padding_side = "left"
    planner_base = AutoModelForCausalLM.from_pretrained(
        args.planner_base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    planner_model = PeftModel.from_pretrained(planner_base, args.planner_lora)
    planner_model.eval()

    # Load Critic
    critic_model = critic_tok = None
    if not args.no_critic:
        print("[2/3] Loading Critic ...")
        critic_tok = AutoTokenizer.from_pretrained(args.critic_base, trust_remote_code=True)
        critic_tok.pad_token = critic_tok.eos_token
        critic_base = AutoModelForCausalLM.from_pretrained(
            args.critic_base, torch_dtype=torch.bfloat16, device_map="auto"
        )
        critic_model = PeftModel.from_pretrained(critic_base, args.critic_lora)
        critic_model.eval()
    else:
        print("[2/3] Critic disabled (ablation baseline)")

    # Load QA model
    print("[3/3] Loading QA model ...")
    qa_tok = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(
        model=args.model_name, trust_remote_code=True,
        gpu_memory_utilization=0.5, max_model_len=16000,
        tensor_parallel_size=2,
    )
    sampling = SamplingParams(max_tokens=512, temperature=0.0)

    # Inference loop
    results, all_em, all_f1 = [], [], []
    critic_stats = {"total": 0, "rejected": 0, "reject_then_correct": 0}

    for i, data in enumerate(tqdm(dataset, desc="Evaluating")):
        question = data["question"]
        gt       = data["answer"]
        context  = "\n\n---\n\n".join(data["pred_texts"])

        # Planner → initial plan
        plan = run_planner(question, planner_model, planner_tok)
        verdict = "ACCEPT"
        num_retries = 0

        # Critic loop
        if critic_model is not None and plan:
            critic_stats["total"] += 1
            verdict, feedback = run_critic(question, plan, critic_model, critic_tok)

            if verdict == "REJECT":
                critic_stats["rejected"] += 1
                for _ in range(MAX_RETRIES):
                    revised = run_planner(question, planner_model, planner_tok, feedback=feedback)
                    if revised:
                        plan = revised
                        num_retries += 1
                        verdict, feedback = run_critic(question, plan, critic_model, critic_tok)
                        if verdict == "ACCEPT":
                            break

        # Agent loop: multi-hop QA
        hop_answers = []
        for sub_q_raw in plan:
            sub_q = replace_placeholders(sub_q_raw, hop_answers)
            prompt_text = HOP_PROMPT.format(context=context, question=sub_q)
            msgs = [{"role": "system", "content": HOP_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt_text}]
            text = qa_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            outputs = llm.generate([text], sampling)
            hop_answers.append(extract_final_answer(outputs[0].outputs[0].text))

        reasoning = build_reasoning_chain(plan, hop_answers)
        synth_text = SYNTHESIS_PROMPT.format(context=context, reasoning=reasoning, question=question)
        msgs = [{"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user",   "content": synth_text}]
        text = qa_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([text], sampling)
        prediction = extract_final_answer(outputs[0].outputs[0].text)

        em = compute_em(prediction, gt)
        f1 = compute_f1(prediction, gt)
        all_em.append(em)
        all_f1.append(f1)

        if verdict == "REJECT" and f1 >= 0.5:
            critic_stats["reject_then_correct"] += 1

        results.append({
            "question":      question,
            "ground_truth":  gt,
            "sub_questions": plan,
            "hop_answers":   hop_answers,
            "prediction":    prediction,
            "critic_verdict": verdict,
            "num_retries":   num_retries,
            "EM": em, "F1": f1,
        })

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(dataset)}]  EM={sum(all_em)/len(all_em):.4f}  F1={sum(all_f1)/len(all_f1):.4f}")
            with open(output_path, "w", encoding="utf-8") as f_out:
                json.dump(results, f_out, indent=2, ensure_ascii=False)

    # Final report
    avg_em = sum(all_em) / len(all_em) if all_em else 0
    avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
    reject_rate = critic_stats["rejected"] / max(critic_stats["total"], 1)

    print("\n" + "=" * 55)
    print("  EVAL RESULTS (with Plan Critic)")
    print("=" * 55)
    print(f"  Samples              : {len(results)}")
    print(f"  Mean EM              : {avg_em:.4f}")
    print(f"  Mean F1              : {avg_f1:.4f}")
    print(f"  Critic reject rate   : {reject_rate:.2%}")
    print(f"  Reject→correct rate  : "
          f"{critic_stats['reject_then_correct']/max(critic_stats['rejected'],1):.2%}")
    print("=" * 55)

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")