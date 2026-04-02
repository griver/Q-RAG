'''
这版修改了之前的占位符替换函数，修复了两个 bug:
1) 连锁污染：如果某个答案里本身包含 #N 字符串，下一轮会把它也替换掉
2) #1 误匹配 #10, #11, #12 ... 造成错误替换
这版添加了max_retries参数,允许对超过hop限制的样本进行多轮重试,增加了成功将过长分解缩短到限制内的机会,并在日志中清晰报告重试情况和最终的hop分布。

一下是一些示例命令(注意修改路径):
#### Server ####
python eval_llm_openqa_with_planner_chain_of_thought_v2.py    \
--file_path ./QRAG_hotpotqa_4090_eval_50/eval_seed42.jsonl   \
--model_name Qwen/QwQ-32B   \
--planner_base Qwen/Qwen2.5-7B-Instruct    \
--planner_lora /workspace/planner/final    \
--output_file_path ./QRAG_hotpotqa_4090_eval_50/llm-answering_qwenplanner_eval_CoT_Retires.json

### 4090 ####
CUDA_VISIBLE_DEVICES=0 python eval_llm_openqa_with_planner_chain_of_thought_v2.py    \
--file_path ./runs/QRAG_hotpotqa_4090_24h15m_50/eval_seed42.jsonl   \
--model_name Qwen/QwQ-32B   \
--planner_base Qwen/Qwen2.5-7B-Instruct    \
--planner_lora /workspace/planner/final    \
--output_file_path ./runs/QRAG_hotpotqa_4090_24h15m_50/llm-answering_qwenplanner_eval_CoT_Retires.json
'''


import json
import os
import re
import string
import argparse
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

# ─────────────────────────────────────────────
#  Planner constants (must match training code)
# ─────────────────────────────────────────────
PLANNER_SYSTEM_PROMPT = (
    "You are a multi-hop question planner. "
    "Given a complex question that requires multiple reasoning steps, "
    "decompose it into a sequence of simple, self-contained sub-questions. "
    "Each sub-question should be answerable independently or by referring to "
    "the answer of a previous step (use '#1', '#2', ... as placeholders). "
    "Output each sub-question on a new line, prefixed with 'Step N:'."
)

# ─────────────────────────────────────────────
#  QA prompts
# ─────────────────────────────────────────────

# Used for each intermediate hop sub-question
HOP_SYSTEM_PROMPT = (
    "Answer the question based on the given passages.\n"
    "Only give me the short and precise answer, do not output any other words.\n"
    "Keep your reasoning very brief and concise.\n"
    "Always end your response with \" [your final answer]\".\n"
)
HOP_PROMPT = (
    "\nGIVEN PASSAGES:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "Final answer: "
)

# Used for the final synthesis: original question + all intermediate Q&A as chain-of-thought
SYNTHESIS_SYSTEM_PROMPT = (
    "Answer the original question based on the given passages and the reasoning steps provided.\n"
    "The reasoning steps show intermediate questions and their answers that build up to the final answer.\n"
    "Use the reasoning steps as helpful hints, but always answer the ORIGINAL QUESTION.\n"
    "Only give me the short and precise answer to the ORIGINAL QUESTION, do not output any other words.\n"
    'Always end your response with "Final answer: [your final answer]".\n'
)
SYNTHESIS_PROMPT = (
    "\nGIVEN PASSAGES:\n{context}\n\n"
    "REASONING STEPS:\n{reasoning}\n\n"
    "ORIGINAL QUESTION:\n{question}\n\n"
    "Final answer: "
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-hop LLM answering with planner + vLLM")
    parser.add_argument("--file_path",        type=str, required=True)
    parser.add_argument("--model_name",       type=str, required=True)
    parser.add_argument("--planner_base",     type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--planner_lora",     type=str, default="./qwen_planner_lora_v2/final")
    parser.add_argument("--output_file_path", type=str, default=None)
    parser.add_argument("--max_hops",         type=int, default=4,
                        help="Safety cap on number of intermediate hops")
    parser.add_argument("--max_retries",      type=int, default=2,
                        help="Number of extra attempts for the planner to reduce hops below the limit")
    parser.add_argument("--planner_batch",    type=int, default=16)
    return parser.parse_args()


# ─────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower().strip())))

def compute_exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))

def recall(pred, gold):
    g, p = normalize_answer(gold).split(), normalize_answer(pred).split()
    n, good = len(g), 0
    for w in p:
        if w in g:
            good += 1
            g.remove(w)
    return good / n if n > 0 else 1

def precision(pred, gold):
    g, p = normalize_answer(gold).split(), normalize_answer(pred).split()
    n, good = len(p), 0
    for w in g:
        if w in p:
            good += 1
            p.remove(w)
    return good / n if n > 0 else 1

def compute_f1(pred, gold):
    prec, rec = precision(pred, gold), recall(pred, gold)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ─────────────────────────────────────────────
#  Planner helpers
# ─────────────────────────────────────────────

def parse_steps(text: str) -> list:
    steps = re.findall(r"Step\s+\d+:\s*(.+)", text, re.IGNORECASE)
    return [s.strip() for s in steps if s.strip()]

# ─────────────────────────────────────────────────────────────
#  旧版：顺序替换，存在两个 bug
# ─────────────────────────────────────────────────────────────

def replace_placeholders_old(question: str, hop_answers: list) -> str:
    result = question
    for i, ans in enumerate(hop_answers, start=1):
        result = result.replace(f"#{i}", ans)
    return result


# Bug 1: 连锁污染
# 如果某个答案本身包含 #N 字符串，下一轮会把它也替换掉
#
# hop_answers = ["Top #4 UK bands", "Dance Gavin Dance"]
# question    = "#1 >> has part"
#
# i=1: "#1" → "Top #4 UK bands"     →  "Top #4 UK bands >> has part"
# i=2: 没有 #2，不变
# i=3: 没有 #3，不变
# i=4: "#4" → "Dance Gavin Dance"   →  "Top Dance Gavin Dance UK bands >> has part"  ✗


# Bug 2: #1 误匹配 #10, #11, #12 ...
#
# hop_answers = ["Parasite"]
# question    = "#10 >> has part"
#
# i=1: str.replace("#1", "Parasite") 会把 "#10" 里的 "#1" 也换掉
#      "#10 >> has part"  →  "Parasite0 >> has part"             ✗


# ─────────────────────────────────────────────────────────────
#  新版：re.sub 单次替换，两个 bug 全部修复
# ─────────────────────────────────────────────────────────────

def replace_placeholders(question: str, hop_answers: list) -> str:
    def replacer(match):
        idx = int(match.group(1))
        if 1 <= idx <= len(hop_answers):
            return hop_answers[idx - 1]
        return match.group(0)   # 占位符超出范围时保持原样
    return re.sub(r'#(\d+)', replacer, question)


# Bug 1 修复：re.sub 扫描原始字符串一次性替换所有 #N，
#            替换结果不会再被扫描，答案里含 #N 也不会被二次替换
#
# hop_answers = ["Top #4 UK bands", "Dance Gavin Dance"]
# question    = "#1 >> has part"
#
# re.sub 扫描原始 "#1 >> has part"，只找到一个 #1
# → "Top #4 UK bands >> has part"                               ✓
#   （答案里的 #4 不会被继续替换）


# Bug 2 修复：正则 #(\d+) 匹配完整数字，#1 不会误匹配 #10
#
# hop_answers = ["Parasite"]
# question    = "#10 >> has part"
#
# re.sub 找到 #10，idx=10，超出 hop_answers 范围 → 保持 "#10" 原样
# → "#10 >> has part"                                           ✓
#   （而不是 "Parasite0 >> has part"）


# ─────────────────────────────────────────────────────────────
#  对比总结
# ─────────────────────────────────────────────────────────────

# ┌─────────────────────┬──────────────────────┬────────────────────────┐
# │                     │       旧版           │        新版             │
# ├─────────────────────┼──────────────────────┼────────────────────────┤
# │ 替换方式             │ 多轮 str.replace      │ 单次 re.sub            │
# │ 连锁污染             │ ✗ 有（答案含 #N）     │ ✓ 无                   │
# │ #1 误匹配 #10        │ ✗ 有                │ ✓ 无                    │
# │ 占位符超出范围        │ 静默跳过（不替换）    │ 保持原样（明确处理）     │
# └─────────────────────┴──────────────────────┴────────────────────────┘
def extract_final_answer(text: str) -> str:
    if "Final answer:" in text:
        return text.split("Final answer:")[-1].strip()
    return text.strip()

def build_reasoning_chain(sub_questions: list, hop_answers: list) -> str:
    """
    Format intermediate Q&A pairs as a readable chain-of-thought string.
    e.g.:
      Q1: Who directed Inception?  ->  A1: Christopher Nolan
      Q2: Where was Christopher Nolan born?  ->  A2: London
    """
    lines = []
    for idx, (q, a) in enumerate(zip(sub_questions, hop_answers), start=1):
        lines.append(f"Q{idx}: {q}  ->  A{idx}: {a}")
    return "\n".join(lines)

def batch_decompose(questions: list, model, tokenizer, batch_size: int = 16) -> list:
    all_steps = []
    for start in tqdm(range(0, len(questions), batch_size), desc="Planner batches"):
        batch_q = questions[start: start + batch_size]
        messages_batch = [
            [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Decompose the following question:\n\n{q}"},
            ]
            for q in batch_q
        ]
        prompts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for ids in out_ids:
            generated = ids[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            all_steps.append(parse_steps(text))
    return all_steps


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.output_file_path:
        output_file_path = args.output_file_path
    else:
        base, _ = os.path.splitext(args.file_path)
        output_file_path = base + "_eval_planner_llm.json"

    print(f"Input        : {args.file_path}")
    print(f"Output       : {output_file_path}")
    print(f"QA model     : {args.model_name}")
    print(f"Planner base : {args.planner_base}")
    print(f"Planner LoRA : {args.planner_lora}")

    # ── Load dataset ──────────────────────────
    dataset = []
    with open(args.file_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    print(f"Samples: {len(dataset)}")

    # ── Load Planner ──────────────────────────
    print("\n[1/3] Loading planner model ...")
    planner_tokenizer = AutoTokenizer.from_pretrained(args.planner_base, trust_remote_code=True)
    planner_tokenizer.pad_token = planner_tokenizer.eos_token
    planner_tokenizer.padding_side = "left"

    planner_base_model = AutoModelForCausalLM.from_pretrained(
        args.planner_base, dtype=torch.bfloat16, device_map="auto"
    )
    planner_model = PeftModel.from_pretrained(planner_base_model, args.planner_lora)
    planner_model.eval()
    print("Planner ready.")

    # ── Decompose all questions (with retry) ──
    print("\n[2/3] Decomposing questions with the planner ...")

    MAX_HOPS_LIMIT = args.max_hops                  # default 4
    MAX_RETRIES    = args.max_retries               # default 2 extra attempts = 3 total

    # per-sample retry counter; 0 = accepted on first attempt
    planner_retries = [0] * len(dataset)

    # First decomposition attempt (attempt 0)
    all_sub_questions = batch_decompose(
        questions, planner_model, planner_tokenizer, batch_size=args.planner_batch
    )

    # Retry loop — only re-runs samples that are still over the hop limit
    for retry_idx in range(1, MAX_RETRIES + 1):   # retry_idx = 1, 2
        over_limit = [i for i, sq in enumerate(all_sub_questions) if len(sq) > MAX_HOPS_LIMIT]
        if not over_limit:
            print(f"\nAll samples within {MAX_HOPS_LIMIT} hops — no retry needed.")
            break

        print(f"\n[Retry {retry_idx}/{MAX_RETRIES}] "
              f"{len(over_limit)} sample(s) still exceed {MAX_HOPS_LIMIT} hops — retrying ...")

        retry_questions = [questions[i] for i in over_limit]
        retry_results   = batch_decompose(
            retry_questions, planner_model, planner_tokenizer, batch_size=args.planner_batch
        )

        for orig_i, new_sq in zip(over_limit, retry_results):
            old_sq = all_sub_questions[orig_i]
            # Always increment retry counter for this sample
            planner_retries[orig_i] = retry_idx
            # Accept the new decomposition only if it is strictly shorter
            if len(new_sq) < len(old_sq):
                print(f"  sample {orig_i}: {len(old_sq)} hops -> {len(new_sq)} hops (accepted)")
                all_sub_questions[orig_i] = new_sq
            else:
                print(f"  sample {orig_i}: {len(old_sq)} hops -> {len(new_sq)} hops (kept old, {len(old_sq)} hops)")

    # Final check — report samples that are still over the limit after all retries
    still_over = [i for i, sq in enumerate(all_sub_questions) if len(sq) > MAX_HOPS_LIMIT]
    if still_over:
        print(f"\n[Warning] {len(still_over)} sample(s) still exceed {MAX_HOPS_LIMIT} hops "
              f"after {MAX_RETRIES} retries. Using the shortest result obtained for each.")

    # effective_max_hops: the true maximum hop count across all samples after retries.
    # This drives the QA loop — using args.max_hops here would silently truncate
    # samples whose planner could not be reduced below the limit.
    effective_max_hops = max((len(sq) for sq in all_sub_questions), default=1)
    print(f"\nEffective max_hops across all samples: {effective_max_hops}")

    # Retry distribution summary
    retry_dist = Counter(planner_retries)
    print("Planner retry distribution:")
    for k in sorted(retry_dist):
        label = "no retry" if k == 0 else f"retried {k}x"
        print(f"  {k} ({label}): {retry_dist[k]} sample(s)")

    # Hop distribution after retries
    n_hops_dist = Counter(len(sq) for sq in all_sub_questions)
    print("Final hop distribution after retries:")
    for k in sorted(n_hops_dist):
        print(f"  {k}-hop: {n_hops_dist[k]}")

    # Free planner GPU memory
    del planner_model, planner_base_model
    torch.cuda.empty_cache()

    # ── Load QA model ─────────────────────────
    print("\n[3/3] Loading QA model with vLLM ...")
    qa_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=32000,
    )
    sampling_params = SamplingParams(max_tokens=512, temperature=0.0)
    print("QA model ready.")

    # ── Intermediate hop QA ───────────────────
    # hop_answers[i] = [ans_step1, ans_step2, ...]
    print("\nStarting intermediate hop answering ...")
    hop_answers = [[] for _ in range(len(dataset))]

    # Use effective_max_hops (true max after retries) so no sample is silently truncated
    max_hops = effective_max_hops
    print(f"Running QA loop for up to {max_hops} hops.\n")

    for hop in range(max_hops):
        active_indices = [i for i, sq in enumerate(all_sub_questions) if hop < len(sq)]
        if not active_indices:
            break

        print(f"\n-- Hop {hop + 1} / {max_hops}  ({len(active_indices)} samples) --")

        prompts = []
        for i in active_indices:
            data      = dataset[i]
            sub_q_raw = all_sub_questions[i][hop]
            sub_q     = replace_placeholders(sub_q_raw, hop_answers[i])
            context   = "\n\n---\n\n".join(data["pred_texts"])
            full_prompt = HOP_PROMPT.format(context=context, question=sub_q)
            messages = [
                {"role": "system", "content": HOP_SYSTEM_PROMPT},
                {"role": "user",   "content": full_prompt},
            ]
            text = qa_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)

        outputs = llm.generate(prompts, sampling_params)

        for i, output in zip(active_indices, outputs):
            ans = extract_final_answer(output.outputs[0].text)
            hop_answers[i].append(ans)
    ##############################################################################
    # GIVEN PASSAGES:
    # ... (文章段落) ...
    # REASONING STEPS:
    # Q1: Which film won the 2020 Oscar for Best Picture?  ->  A1: Parasite
    # Q2: Who directed Parasite?                           ->  A2: Bong Joon-ho
    # ORIGINAL QUESTION:
    # The film that won the 2020 Oscar for Best Picture — who directed it?
    # Final answer: Bong Joon-ho
    ###############################################################################



    # ── Final synthesis step ──────────────────
    print(f"\n-- Final Synthesis  ({len(dataset)} samples) --")

    synthesis_prompts = []
    for i, data in enumerate(dataset):
        question  = data["question"]
        context   = "\n\n---\n\n".join(data["pred_texts"])
        sub_qs    = all_sub_questions[i]
        h_answers = hop_answers[i]

        reasoning = build_reasoning_chain(sub_qs, h_answers)

        full_prompt = SYNTHESIS_PROMPT.format(
            context=context,
            reasoning=reasoning,
            question=question,
        )
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user",   "content": full_prompt},
        ]
        text = qa_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        synthesis_prompts.append(text)

    synthesis_outputs = llm.generate(synthesis_prompts, sampling_params)
    final_predictions = [
        extract_final_answer(o.outputs[0].text) for o in synthesis_outputs
    ]

    # ── Aggregate results ─────────────────────
    print("\nAggregating results ...")
    results       = []
    all_em_scores = []
    all_f1_scores = []

    for i, data in enumerate(dataset):
        question      = data["question"]
        ground_truth  = data["answer"]
        sub_questions = all_sub_questions[i]
        h_answers     = hop_answers[i]
        prediction    = final_predictions[i]

        em = compute_exact_match(prediction, ground_truth)
        f1 = compute_f1(prediction, ground_truth)
        all_em_scores.append(em)
        all_f1_scores.append(f1)

        results.append({
            "question":                question,
            "retrieved_chunks_idx":    data.get("pred_idx"),
            "ground_truth_chunks_idx": data.get("sf_idx"),
            "ground_truth":            ground_truth,
            "sub_questions":           sub_questions,
            "hop_answers":             h_answers,
            "prediction":              prediction,
            "EM":                      em,
            "F1":                      f1,
            # ── NEW: per-sample planner diagnostics ──────────────────────
            "planner_retries":         planner_retries[i],   # 0 / 1 / 2
            "num_hops":                len(sub_questions),   # final hop count used
        })

        if (i + 1) % 100 == 0:
            avg_em = sum(all_em_scores) / len(all_em_scores)
            avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
            print(f"[{i+1}/{len(dataset)}]  EM={avg_em:.4f}  F1={avg_f1:.4f}")
            with open(output_file_path, "w", encoding="utf-8") as f_out:
                json.dump(results, f_out, indent=4, ensure_ascii=False)

    # ── Final report ──────────────────────────
    avg_em = sum(all_em_scores) / len(all_em_scores) if all_em_scores else 0
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0

    retry_dist_final = Counter(r["planner_retries"] for r in results)

    print("\n" + "=" * 50)
    print("         EVAL RESULTS (with planner)")
    print("=" * 50)
    print(f"Num samples          : {len(results)}")
    print(f"Mean Exact Match (EM): {avg_em:.4f}")
    print(f"Mean F1-Score        : {avg_f1:.4f}")
    print(f"Effective max hops   : {effective_max_hops}")
    print("Planner retry counts :")
    for k in sorted(retry_dist_final):
        label = "accepted on 1st attempt" if k == 0 else f"needed {k} retry/retries"
        print(f"  retries={k} ({label}): {retry_dist_final[k]} sample(s)")
    print("=" * 50)

    with open(output_file_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_file_path}")