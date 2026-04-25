import argparse
import json
import os
import re
import string
from typing import Any, Dict, List, Optional, Set

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "./qwen_planner_lora/final"

PLANNER_SYSTEM_PROMPT = """You are a planning agent for open-domain QA.
Given QUESTION and numbered PASSAGES, output STRICT JSON only:
{"selected_idx":[...], "sub_questions":[...], "confidence":0.0}
Rules:
- selected_idx must contain only integer indices that appear in PASSAGES
- choose evidence that is sufficient to answer the question
- prefer keeping 2-6 passages; do not return an empty list
- do not output any extra text outside JSON
"""

PLANNER_USER_PROMPT = """QUESTION:
{question}

PASSAGES:
{numbered_passages}
"""

qa_instruction_prompt = """Answer the question based on the given passages.
Only give me the short and precise answer, do not output any other words.
Keep your reasoning very brief and concise.
Always end your response with "Final answer: [your final answer]".
"""

qa_prompt = """
GIVEN PASSAGES:
{context}

QUESTION:
{question}

Final answer: """

def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower().strip())))

def compute_exact_match(prediction: str, target: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(target))

def recall(prediction: str, target: str) -> float:
    target_tokens = normalize_answer(target).split()
    prediction_tokens = normalize_answer(prediction).split()
    len_true = len(target_tokens)
    len_good = 0
    for word in prediction_tokens:
        if word in target_tokens:
            len_good += 1
            target_tokens.remove(word)
    return len_good / len_true if len_true > 0 else 1.0

def precision(prediction: str, target: str) -> float:
    target_tokens = normalize_answer(target).split()
    prediction_tokens = normalize_answer(prediction).split()
    len_gen = len(prediction_tokens)
    len_good = 0
    for word in target_tokens:
        if word in prediction_tokens:
            len_good += 1
            prediction_tokens.remove(word)
    return len_good / len_gen if len_gen > 0 else 1.0

def compute_f1(prediction: str, target: str) -> float:
    p = precision(prediction, target)
    r = recall(prediction, target)
    if (p + r) == 0.0:
        return 0.0
    return (2.0 * p * r) / (p + r)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_pred_chunks(sample: Dict[str, Any]) -> List[str]:
    chunks = sample.get("pred_texts")
    if chunks is None:
        chunks = sample.get("pred_text", [])
    return list(chunks)

def build_numbered_passages(chunks: List[str]) -> str:
    return "\n\n".join([f"[{i}] {chunk}" for i, chunk in enumerate(chunks)])

def extract_final_answer(text: str) -> str:
    markers = ["Final answer:", "final answer:", "FINAL ANSWER:"]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[-1]
            break
    text = text.strip().strip('"').strip("'")
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    return text

def safe_parse_plan(plan_text: str) -> Dict[str, Any]:
    parsed = {
        "selected_idx": [],
        "sub_questions": [],
        "confidence": 0.0,
        "parse_ok": False,
    }

    json_match = re.search(r"\{[\s\S]*?\}", plan_text)
    if not json_match:
        return parsed

    try:
        obj = json.loads(json_match.group(0))
        raw_idx = obj.get("selected_idx", [])
        valid_idx: List[int] = []
        for x in raw_idx:
            if isinstance(x, int):
                valid_idx.append(x)
            elif isinstance(x, str) and x.isdigit():
                valid_idx.append(int(x))

        parsed["selected_idx"] = valid_idx
        parsed["sub_questions"] = obj.get("sub_questions", [])
        parsed["confidence"] = float(obj.get("confidence", 0.0))
        parsed["parse_ok"] = True
    except Exception:
        return parsed

    return parsed

def select_context_by_plan(chunks: List[str], selected_idx: List[int], top_k: int) -> Dict[str, Any]:
    valid = []
    seen = set()
    for idx in selected_idx:
        if 0 <= idx < len(chunks) and idx not in seen:
            valid.append(idx)
            seen.add(idx)

    if not valid:
        valid = list(range(min(top_k, len(chunks))))

    return {
        "selected_idx": valid,
        "selected_texts": [chunks[i] for i in valid],
    }

def select_context_hybrid(
    chunks: List[str],
    planner_selected_idx: List[int],
    top_k_fallback: int,
    force_include_top_n: int,
    min_keep: int,
    max_keep: int,
) -> Dict[str, Any]:
    """Build a robust context: planner picks + top retriever anchors + bounded size."""
    keep: List[int] = []
    seen: Set[int] = set()

    for idx in planner_selected_idx:
        if 0 <= idx < len(chunks) and idx not in seen:
            keep.append(idx)
            seen.add(idx)

    # Always preserve earliest retriever chunks as anchors.
    anchor_n = max(0, min(force_include_top_n, len(chunks)))
    for idx in range(anchor_n):
        if idx not in seen:
            keep.append(idx)
            seen.add(idx)

    # If still too few, expand with fallback prefix.
    if len(keep) < min_keep:
        fallback_n = max(min_keep, top_k_fallback)
        for idx in range(min(fallback_n, len(chunks))):
            if idx not in seen:
                keep.append(idx)
                seen.add(idx)
            if len(keep) >= min_keep:
                break

    keep = keep[:max_keep] if max_keep > 0 else keep
    keep_sorted = sorted(keep)
    return {
        "selected_idx": keep_sorted,
        "selected_texts": [chunks[i] for i in keep_sorted],
    }

def merge_with_full_context(chunks: List[str], selected_idx: List[int]) -> Dict[str, Any]:
    """Keep full context while moving planner-selected chunks to the front."""
    selected = []
    seen: Set[int] = set()
    for idx in selected_idx:
        if 0 <= idx < len(chunks) and idx not in seen:
            selected.append(idx)
            seen.add(idx)

    remaining = [idx for idx in range(len(chunks)) if idx not in seen]
    final_idx = selected + remaining
    return {
        "selected_idx": final_idx,
        "selected_texts": [chunks[i] for i in final_idx],
    }

def apply_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenQA evaluation with optional Planner Agent")
    parser.add_argument("--file_path", type=str, required=True, help="Path to input JSONL from retriever")
    parser.add_argument("--model_name", type=str, required=True, help="Reader model name/path")
    parser.add_argument("--output_file_path", type=str, default=None, help="Path for output JSON")

    parser.add_argument("--use_planner", action="store_true", help="Enable planner->reader two-stage pipeline")
    parser.add_argument("--planner_model_name", type=str, default=BASE_MODEL, help="Planner base model name/path")
    parser.add_argument("--planner_lora_path", type=str, default=LORA_PATH, help="Planner LoRA adapter path")
    parser.add_argument(
        "--allow_base_planner_without_lora",
        action="store_true",
        help="Allow planner without LoRA adapter (disabled by default to avoid quality drop)",
    )
    parser.add_argument("--planner_top_k", type=int, default=4, help="Fallback top-k chunks when plan parse fails")
    parser.add_argument("--planner_min_keep", type=int, default=3, help="Minimum chunks kept for answering")
    parser.add_argument("--planner_max_keep", type=int, default=8, help="Maximum chunks kept for answering")
    parser.add_argument("--planner_force_include_top_n", type=int, default=2, help="Always include first N retrieved chunks")
    parser.add_argument("--planner_conf_threshold", type=float, default=0.35, help="If planner confidence below this, expand fallback context")
    parser.add_argument(
        "--planner_context_mode",
        type=str,
        default="full_with_priority",
        choices=["selected_only", "full_with_priority"],
        help=(
            "How planner affects reader context: "
            "selected_only = pass only filtered chunks; "
            "full_with_priority = keep all chunks but move planner-selected ones first"
        ),
    )
    parser.add_argument("--planner_max_tokens", type=int, default=256, help="Max tokens for planner generation")
    parser.add_argument("--planner_temperature", type=float, default=0.0, help="Planner generation temperature")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="vLLM GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=32000, help="vLLM max model length")

    args = parser.parse_args()

    file_path = args.file_path
    model_name = args.model_name
    planner_model_name = args.planner_model_name or BASE_MODEL
    normalized_lora_path = (args.planner_lora_path or "").strip()

    use_planner_runtime = args.use_planner
    if use_planner_runtime and not normalized_lora_path and not args.allow_base_planner_without_lora:
        print(
            "[Planner] --planner_lora_path is empty. "
            "Auto-disabling planner to protect EM/F1. "
            "Use --allow_base_planner_without_lora to force-enable."
        )
        use_planner_runtime = False
    elif use_planner_runtime and normalized_lora_path and not os.path.isdir(normalized_lora_path):
        if args.allow_base_planner_without_lora:
            print(
                f"[Planner] LoRA directory not found: {normalized_lora_path}. "
                "Falling back to base planner because --allow_base_planner_without_lora is set."
            )
            normalized_lora_path = ""
        else:
            print(
                f"[Planner] LoRA directory not found: {normalized_lora_path}. "
                "Auto-disabling planner to protect EM/F1. "
                "Use --allow_base_planner_without_lora to force base planner."
            )
            use_planner_runtime = False

    if args.output_file_path:
        output_file_path = args.output_file_path
    else:
        base, _ = os.path.splitext(file_path)
        output_file_path = base + "_eval_llm.json"

    print(f"Input: {file_path}")
    print(f"Output: {output_file_path}")
    print(f"Reader model: {model_name}")
    print(f"Use planner: {use_planner_runtime}")
    if use_planner_runtime:
        print(f"Planner model: {planner_model_name}")
        print(f"Planner LoRA: {normalized_lora_path or '<none>'}")

    dataset = load_jsonl(file_path)
    print(f"Samples in dataset: {len(dataset)}")

    # Reuse one engine when planner and reader share the same base model.
    same_model_for_planner = use_planner_runtime and (planner_model_name == model_name)
    need_lora_on_reader = same_model_for_planner and bool(normalized_lora_path)

    reader_tokenizer = AutoTokenizer.from_pretrained(model_name)
    reader_llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_lora=need_lora_on_reader,
        max_lora_rank = 64,
        # max_num_loras=8
    )

    planner_tokenizer = reader_tokenizer
    planner_llm = None
    planner_lora_request = None
    if use_planner_runtime:
        if same_model_for_planner:
            planner_tokenizer = reader_tokenizer
            planner_llm = reader_llm
        else:
            planner_tokenizer = AutoTokenizer.from_pretrained(planner_model_name)
            planner_llm = LLM(
                model=planner_model_name,
                trust_remote_code=True,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                enable_lora=bool(normalized_lora_path),
                max_lora_rank = 64,
                # max_num_loras=8
            )
        if normalized_lora_path:
            planner_lora_request = LoRARequest("planner_lora", 1, normalized_lora_path)

    planner_outputs: List[Optional[str]] = [None] * len(dataset)
    planner_infos: List[Dict[str, Any]] = [
        {
            "selected_idx": [],
            "selected_texts": [],
            "sub_questions": [],
            "confidence": 0.0,
            "parse_ok": False,
            "raw": None,
        }
        for _ in dataset
    ]

    if use_planner_runtime:
        planner_prompts: List[str] = []
        for sample in tqdm(dataset, desc="Preparing planner prompts"):
            chunks = get_pred_chunks(sample)
            numbered_passages = build_numbered_passages(chunks)
            planner_user = PLANNER_USER_PROMPT.format(
                question=sample["question"],
                numbered_passages=numbered_passages,
            )
            planner_prompts.append(
                apply_template(planner_tokenizer, PLANNER_SYSTEM_PROMPT, planner_user)
            )

        planner_sampling_params = SamplingParams(
            max_tokens=args.planner_max_tokens,
            temperature=args.planner_temperature,
        )

        print("Starting planner generation...")
        planner_gen_outputs = planner_llm.generate(
            planner_prompts,
            planner_sampling_params,
            lora_request=planner_lora_request,
        )
        print("Planner generation completed.")

        for i, output in enumerate(tqdm(planner_gen_outputs, desc="Parsing planner outputs")):
            raw_plan = output.outputs[0].text
            parsed = safe_parse_plan(raw_plan)
            chunks = get_pred_chunks(dataset[i])

            # Low confidence or parse failure gets a wider fallback context.
            dynamic_top_k = args.planner_top_k
            if (not parsed["parse_ok"]) or (parsed["confidence"] < args.planner_conf_threshold):
                dynamic_top_k = max(dynamic_top_k, args.planner_min_keep + args.planner_force_include_top_n)

            chosen = select_context_hybrid(
                chunks=chunks,
                planner_selected_idx=parsed["selected_idx"],
                top_k_fallback=dynamic_top_k,
                force_include_top_n=args.planner_force_include_top_n,
                min_keep=args.planner_min_keep,
                max_keep=args.planner_max_keep,
            )

            if args.planner_context_mode == "full_with_priority":
                chosen = merge_with_full_context(chunks, chosen["selected_idx"])

            planner_outputs[i] = raw_plan
            planner_infos[i] = {
                "selected_idx": chosen["selected_idx"],
                "selected_texts": chosen["selected_texts"],
                "sub_questions": parsed["sub_questions"],
                "confidence": parsed["confidence"],
                "parse_ok": parsed["parse_ok"],
                "raw": raw_plan,
            }

    answer_prompts: List[str] = []
    for i, sample in enumerate(tqdm(dataset, desc="Preparing answer prompts")):
        question = sample["question"]
        pred_chunks = get_pred_chunks(sample)

        if use_planner_runtime:
            selected_texts = planner_infos[i]["selected_texts"]
            context = "\n\n---\n\n".join(selected_texts)
        else:
            context = "\n\n---\n\n".join(pred_chunks)

        answer_user = qa_prompt.format(context=context, question=question)
        answer_prompts.append(
            apply_template(reader_tokenizer, qa_instruction_prompt, answer_user)
        )

    answer_sampling_params = SamplingParams(
        max_tokens=4000,
        temperature=0.0,
    )

    print("Starting answer generation...")
    answer_outputs = reader_llm.generate(answer_prompts, answer_sampling_params)
    print("Answer generation completed.")

    results: List[Dict[str, Any]] = []
    all_em_scores: List[int] = []
    all_f1_scores: List[float] = []

    for i, (sample, output) in enumerate(
        tqdm(zip(dataset, answer_outputs), total=len(dataset), desc="Processing results")
    ):
        ground_truth_answer = sample["answer"]
        decoded_output = output.outputs[0].text
        llm_prediction = extract_final_answer(decoded_output)

        em_score = compute_exact_match(llm_prediction, ground_truth_answer)
        f1_score = compute_f1(llm_prediction, ground_truth_answer)

        all_em_scores.append(em_score)
        all_f1_scores.append(f1_score)

        result_entry: Dict[str, Any] = {
            "question": sample["question"],
            "retrieved_chunks_idx": sample.get("pred_idx", []),
            "ground_truth_chunks_idx": sample.get("sf_idx", []),
            "ground_truth": ground_truth_answer,
            "prediction": llm_prediction,
            "full_model_output": decoded_output,
            "EM": em_score,
            "F1": f1_score,
        }

        if use_planner_runtime:
            result_entry.update(
                {
                    "planner_raw": planner_infos[i]["raw"],
                    "planner_parse_ok": planner_infos[i]["parse_ok"],
                    "planner_selected_idx": planner_infos[i]["selected_idx"],
                    "planner_selected_texts": planner_infos[i]["selected_texts"],
                    "planner_sub_questions": planner_infos[i]["sub_questions"],
                    "planner_confidence": planner_infos[i]["confidence"],
                }
            )

        results.append(result_entry)

        if (i + 1) % 100 == 0:
            with open(output_file_path, "w", encoding="utf-8") as f_out:
                json.dump(results, f_out, indent=2, ensure_ascii=False)
            print(f"--- {i + 1}/{len(dataset)}: Intermediate results saved. ---")

    avg_em = sum(all_em_scores) / len(all_em_scores) if all_em_scores else 0.0
    avg_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0

    print("\n" + "=" * 50)
    print("             EVAL RESULTS")
    print("=" * 50)
    print(f"Num samples: {len(results)}")
    print(f"Mean Exact Match (EM): {avg_em:.4f}")
    print(f"Mean F1-Score: {avg_f1:.4f}")
    print("=" * 50)

    with open(output_file_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2, ensure_ascii=False)

    print(f"All results saved to {output_file_path}")

if __name__ == "__main__":
    main()


