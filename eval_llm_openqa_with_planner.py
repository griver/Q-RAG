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
#  QA prompt (unchanged from original)
# ─────────────────────────────────────────────
QA_SYSTEM_PROMPT = (
    "Answer the question based on the given passages.\n"
    "Only give me the short and precise answer, do not output any other words.\n"
    "Keep your reasoning very brief and concise.\n"
    'Always end your response with "Final answer: [your final answer]".\n'
)
QA_PROMPT = (
    "\nGIVEN PASSAGES:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "Final answer: "
)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-hop LLM answering with planner + vLLM")
    parser.add_argument("--file_path",      type=str, required=True,
                        help="Path to input JSONL (with pred_texts, answer, etc.)")
    parser.add_argument("--model_name",     type=str, required=True,
                        help="Path/name of the QA model (vLLM)")
    parser.add_argument("--planner_base",   type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model id for the planner")
    parser.add_argument("--planner_lora",   type=str, default="./qwen_planner_lora_v2/final",
                        help="Path to the planner LoRA checkpoint")
    parser.add_argument("--output_file_path", type=str, default=None,
                        help="Output JSON path (default: <input_stem>_eval_planner_llm.json)")
    parser.add_argument("--max_hops",       type=int, default=4,
                        help="Maximum number of hops to execute (safety cap)")
    parser.add_argument("--planner_batch",  type=int, default=16,
                        help="Number of questions to decompose at once (reduce if OOM)")
    return parser.parse_args()

# ─────────────────────────────────────────────
#  Helpers
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

def parse_steps(text: str) -> list:
    """Extract sub-questions from planner output."""
    steps = re.findall(r"Step\s+\d+:\s*(.+)", text, re.IGNORECASE)
    return [s.strip() for s in steps if s.strip()]

def replace_placeholders(question: str, hop_answers: list) -> str:
    """Replace #1, #2, ... with previously computed answers."""
    result = question
    for i, ans in enumerate(hop_answers, start=1):
        result = result.replace(f"#{i}", ans)
    return result

def extract_final_answer(text: str) -> str:
    if "Final answer:" in text:
        return text.split("Final answer:")[-1].strip()
    return text.strip()

def batch_decompose(questions: list, model, tokenizer, batch_size: int = 16) -> list:
    """Return a list of sub-question lists, one per input question."""
    all_steps = []
    for start in tqdm(range(0, len(questions), batch_size), desc="Planner batches"):
        batch_q = questions[start: start + batch_size]
        messages_batch = [
            [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Decompose:\n{q}"},
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


if __name__ == "__main__":
    args = parse_args()

    if args.output_file_path:
        output_file_path = args.output_file_path
    else:
        base, _ = os.path.splitext(args.file_path)
        output_file_path = base + "_eval_planner_llm.json"

    print(f"Input  : {args.file_path}")
    print(f"Output : {output_file_path}")
    print(f"QA model     : {args.model_name}")
    print(f"Planner base : {args.planner_base}")
    print(f"Planner LoRA : {args.planner_lora}")

    # ─────────────────────────────────────────────
    #  Load dataset
    # ─────────────────────────────────────────────
    dataset = []
    with open(args.file_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    print(f"Samples: {len(dataset)}")

    # ─────────────────────────────────────────────
    #  Load Planner (HuggingFace + LoRA)
    # ─────────────────────────────────────────────
    print("\n[1/3] Loading planner model …")
    planner_tokenizer = AutoTokenizer.from_pretrained(args.planner_base, trust_remote_code=True)
    planner_tokenizer.pad_token = planner_tokenizer.eos_token
    planner_tokenizer.padding_side = "left"

    planner_base_model = AutoModelForCausalLM.from_pretrained(
        args.planner_base, dtype=torch.bfloat16, device_map="auto"
    )
    planner_model = PeftModel.from_pretrained(planner_base_model, args.planner_lora)
    planner_model.eval()
    print("Planner ready.")

    # ─────────────────────────────────────────────
    #  Decompose all questions with the planner
    # ─────────────────────────────────────────────
    print("\n[2/3] Decomposing questions with the planner …")
    questions = [d["question"] for d in dataset]
    all_sub_questions = batch_decompose(
        questions, planner_model, planner_tokenizer, batch_size=args.planner_batch
    )

    n_hops_dist = Counter(len(sq) for sq in all_sub_questions)
    print("Decomposition hop distribution:")
    for k in sorted(n_hops_dist):
        print(f"  {k}-hop: {n_hops_dist[k]}")

    # Free planner GPU memory before loading vLLM
    del planner_model, planner_base_model
    torch.cuda.empty_cache()

    # ─────────────────────────────────────────────
    #  Load QA model (vLLM)
    # ─────────────────────────────────────────────
    print("\n[3/3] Loading QA model with vLLM …")
    qa_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=32000,
    )
    sampling_params = SamplingParams(max_tokens=512, temperature=0.0)
    print("QA model ready.")

    # ─────────────────────────────────────────────
    #  Multi-hop QA loop
    # ─────────────────────────────────────────────
    print("\nStarting multi-hop answering …")
    hop_answers = [[] for _ in range(len(dataset))]
    max_hops = min(args.max_hops, max((len(sq) for sq in all_sub_questions), default=1))

    for hop in range(max_hops):
        active_indices = [i for i, sq in enumerate(all_sub_questions) if hop < len(sq)]
        if not active_indices:
            break

        print(f"\n── Hop {hop + 1} / {max_hops}  ({len(active_indices)} samples) ──")

        prompts = []
        for i in active_indices:
            data        = dataset[i]
            sub_q_raw   = all_sub_questions[i][hop]
            sub_q       = replace_placeholders(sub_q_raw, hop_answers[i])
            context     = "\n\n---\n\n".join(data["pred_texts"])
            full_prompt = QA_PROMPT.format(context=context, question=sub_q)
            messages    = [
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user",   "content": full_prompt},
            ]
            text = qa_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)

        outputs = llm.generate(prompts, sampling_params)

        for i, output in zip(active_indices, outputs):
            raw_ans = output.outputs[0].text
            ans     = extract_final_answer(raw_ans)
            hop_answers[i].append(ans)

    # ─────────────────────────────────────────────
    #  Aggregate results
    # ─────────────────────────────────────────────
    print("\nAggregating results …")
    results       = []
    all_em_scores = []
    all_f1_scores = []

    for i, data in enumerate(dataset):
        question     = data["question"]
        ground_truth = data["answer"]
        sub_questions = all_sub_questions[i]
        answers       = hop_answers[i]

        final_prediction = answers[-1] if answers else ""

        em = compute_exact_match(final_prediction, ground_truth)
        f1 = compute_f1(final_prediction, ground_truth)
        all_em_scores.append(em)
        all_f1_scores.append(f1)

        results.append({
            "question":                question,
            "retrieved_chunks_idx":    data.get("pred_idx"),
            "ground_truth_chunks_idx": data.get("sf_idx"),
            "ground_truth":            ground_truth,
            "sub_questions":           sub_questions,
            "hop_answers":             answers,
            "prediction":              final_prediction,
            "EM":                      em,
            "F1":                      f1,
        })

        if (i + 1) % 100 == 0:
            avg_em = sum(all_em_scores) / len(all_em_scores)
            avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
            print(f"[{i+1}/{len(dataset)}]  EM={avg_em:.4f}  F1={avg_f1:.4f}")
            with open(output_file_path, "w", encoding="utf-8") as f_out:
                json.dump(results, f_out, indent=4, ensure_ascii=False)

    # ─────────────────────────────────────────────
    #  Final report
    # ─────────────────────────────────────────────
    avg_em = sum(all_em_scores) / len(all_em_scores) if all_em_scores else 0
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0

    print("\n" + "=" * 50)
    print("         EVAL RESULTS (with planner)")
    print("=" * 50)
    print(f"Num samples          : {len(results)}")
    print(f"Mean Exact Match (EM): {avg_em:.4f}")
    print(f"Mean F1-Score        : {avg_f1:.4f}")
    print("=" * 50)

    with open(output_file_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_file_path}")