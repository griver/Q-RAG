"""
Critic Agent RL Training via GRPO — v2 (CIV + Stratified + Dual-Objective)
===========================================================================
Changes from v1:
  1. Counterfactual Advantage (CIV): advantage is r̄_A - r̄_R per verdict,
     not standard group-wise z-score.
  2. Stratified Sampling: K-2 free samples + 1 guided ACCEPT + 1 guided REJECT.
     Guided prefix tokens are masked from log-prob computation.
  3. Dual-Objective Reward: REJECT samples get
       r = r_verdict + λ * (score_after_replan - score_before_replan)
  4. Per-token average log-prob (not sum) to stabilise loss scale.
  5. GRPO_GROUP_SIZE raised to 8.

Dual-GPU optimized (ideal for dual RTX 4090).
"""

import json
import os
import re
import string
import argparse
import atexit
import multiprocessing as mp
import random
import hashlib
import pickle
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm


# ─────────────────────────────────────────────────────────────
#  Constants — Planner (keep identical to eval script)
# ─────────────────────────────────────────────────────────────
PLANNER_SYSTEM_PROMPT = (
    "You are a multi-hop question planner. "
    "Given a complex question that requires multiple reasoning steps, "
    "decompose it into a sequence of simple, self-contained sub-questions. "
    "Each sub-question should be answerable independently or by referring to "
    "the answer of a previous step (use '#1', '#2', ... as placeholders). "
    "Output each sub-question on a new line, prefixed with 'Step N:'"
)

HOP_SYSTEM_PROMPT = (
    "Answer the question based on the given passages.\n"
    "Only give me the short and precise answer, do not output any other words.\n"
    "Keep your reasoning very brief and concise.\n"
    'Always end your response with "Final answer: [your final answer]".\n'
)
HOP_PROMPT = (
    "\nGIVEN PASSAGES:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "Final answer: "
)

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

# ─────────────────────────────────────────────────────────────
#  Critic system prompt
# ─────────────────────────────────────────────────────────────
CRITIC_SYSTEM_PROMPT = (
    "You are a Plan Quality Critic. "
    "Your job is to evaluate whether the given decomposition plan correctly "
    "breaks down a complex multi-hop question into solvable sub-steps.\n\n"
    "A GOOD plan:\n"
    "  - Has sub-questions that are self-contained or refer to prior answers with #N\n"
    "  - Covers ALL aspects needed to answer the original question\n"
    "  - Has no circular dependencies between sub-questions\n"
    "  - Has an appropriate number of steps (not too many, not too few)\n\n"
    "Output format (strictly follow this):\n"
    "<verdict>ACCEPT</verdict>\n"
    "or\n"
    "<verdict>REJECT</verdict>\n"
    "<feedback>One concise sentence explaining what is wrong and how to fix it.</feedback>"
)

CRITIC_USER_PROMPT = (
    "ORIGINAL QUESTION:\n{question}\n\n"
    "PROPOSED PLAN:\n{plan}\n\n"
    "Evaluate this plan carefully."
)

# ─────────────────────────────────────────────────────────────
#  Training hyperparameters
# ─────────────────────────────────────────────────────────────
GRPO_GROUP_SIZE = 8          # ← was 4; raised for stratified sampling headroom
MAX_RETRIES = 2
LR = 1e-5
WARMUP_STEPS = 50
GRAD_CLIP = 1.0
KL_COEF = 0.05
REWARD_ACCEPT_CORRECT = 1.0
REWARD_ACCEPT_WRONG = -1.0
REWARD_REJECT_CORRECT = 1.0
REWARD_REJECT_STILL_BAD = -0.5

# ── v2 new hyper-params ─────────────────────────────────────
LAMBDA_FEEDBACK = 0.5        # weight of r_feedback in dual-objective reward
CIV_MARGIN = 0.1             # ignore CIV advantage when |r̄_A - r̄_R| < margin

# ─────────────────────────────────────────────────────────────
#  Curriculum stage thresholds
# ─────────────────────────────────────────────────────────────
STAGE_THRESHOLDS = {
    1: {"EM": 0.45, "F1": 0.55},
    2: {"EM": 0.38, "F1": 0.48},
    3: {"EM": 0.30, "F1": 0.40},
}
MAX_STAGE_RETRAIN = 2

QA_MAX_TOKENS = 48
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.35
DEFAULT_VLLM_MAX_MODEL_LEN = 16000


# ─────────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="JSONL with 'question', 'answer', 'pred_texts' fields",
    )
    p.add_argument(
        "--critic_base",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model for critic (trained from scratch with LoRA)",
    )
    p.add_argument("--planner_base", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--planner_lora", type=str, default="./planner/qwen_planner_lora_v2")
    p.add_argument(
        "--qa_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="QA model for agent loop (loaded via vLLM)",
    )
    p.add_argument("--output_dir", type=str, default="./critic_lora_grpo")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--stop_stage_after_pass",
        action="store_true",
        default=True,
        help="Check stage threshold after each epoch and stop that stage as soon as it passes.",
    )
    p.add_argument(
        "--disable_stop_stage_after_pass",
        action="store_false",
        dest="stop_stage_after_pass",
        help="Force the old behavior: always run all epochs inside each stage attempt.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Questions per gradient update",
    )
    p.add_argument(
        "--critic_temp",
        type=float,
        default=0.8,
        help="Sampling temperature for GRPO group sampling",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument(
        "--planner_cache_path",
        type=str,
        default=None,
        help="Persistent planner cache path. Defaults to <output_dir>/planner_cache.pkl",
    )
    p.add_argument(
        "--qa_cache_path",
        type=str,
        default=None,
        help="Persistent QA-result cache path. Defaults to <output_dir>/qa_cache.pkl",
    )
    p.add_argument(
        "--qa_gpu_id",
        type=int,
        default=0,
        help="GPU id used by vLLM / QA inference",
    )
    p.add_argument(
        "--train_gpu_id",
        type=int,
        default=1,
        help="GPU id used by Planner + Critic training",
    )
    p.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    )
    p.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=DEFAULT_VLLM_MAX_MODEL_LEN,
    )
    # ── v2 CLI args ──────────────────────────────────────────
    p.add_argument(
        "--lambda_feedback",
        type=float,
        default=LAMBDA_FEEDBACK,
        help="Weight of r_feedback in dual-objective reward for REJECT samples",
    )
    p.add_argument(
        "--civ_margin",
        type=float,
        default=CIV_MARGIN,
        help="Dead-zone margin for CIV advantage; below this |r̄_A-r̄_R| advantage is zeroed",
    )
    p.add_argument(
        "--group_size",
        type=int,
        default=GRPO_GROUP_SIZE,
        help="GRPO group size (must be >= 4 for stratified sampling)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────
def get_model_device(model):
    return next(model.parameters()).device


def normalize_feedback(feedback: str | None):
    if feedback is None:
        return None
    return " ".join(feedback.strip().split())


def stable_hash(payload) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def planner_cache_key(question: str, feedback: str | None) -> str:
    return stable_hash({"question": question, "feedback": normalize_feedback(feedback)})


def agent_loop_cache_key(question: str, final_plan: list, context: str) -> str:
    return stable_hash({"question": question, "final_plan": final_plan, "context": context})


def critic_prompt_cache_key(question: str, plan: list) -> str:
    return stable_hash({"question": question, "plan": plan})


def load_pickle_cache(path: str) -> dict:
    if path is None or (not os.path.exists(path)):
        return {}
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception as exc:
        print(f"Warning: failed to load cache {path}: {exc}")
    return {}


def save_pickle_cache(cache: dict, path: str):
    if path is None:
        return
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


# ─────────────────────────────────────────────────────────────
#  vLLM worker on dedicated QA GPU
# ─────────────────────────────────────────────────────────────
class _ProxyOutputItem:
    def __init__(self, text: str):
        self.text = text


class _ProxyRequestOutput:
    def __init__(self, prompt: str, texts):
        self.prompt = prompt
        self.outputs = [_ProxyOutputItem(t) for t in texts]


def _serialize_sampling_params(sampling_params):
    if isinstance(sampling_params, dict):
        return dict(sampling_params)

    payload = {
        "max_tokens": getattr(sampling_params, "max_tokens", 256),
        "temperature": getattr(sampling_params, "temperature", 0.0),
    }
    top_p = getattr(sampling_params, "top_p", None)
    if top_p is not None:
        payload["top_p"] = top_p
    top_k = getattr(sampling_params, "top_k", None)
    if top_k is not None:
        payload["top_k"] = top_k
    stop = getattr(sampling_params, "stop", None)
    if stop is not None:
        payload["stop"] = stop
    return payload


def _vllm_worker_main(
    visible_gpu_id, model_name, trust_remote_code,
    gpu_memory_utilization, max_model_len, req_q, resp_q, ready_q,
):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu_id)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_name,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        ready_q.put({"ok": True})

        while True:
            msg = req_q.get()
            if msg is None:
                break
            cmd = msg.get("cmd", "")
            if cmd != "generate":
                resp_q.put({"ok": False, "error": f"Unknown cmd: {cmd}"})
                continue

            sampling = SamplingParams(**msg["sampling"])
            outputs = llm.generate(msg["prompts"], sampling)
            payload = []
            for out in outputs:
                texts = [x.text for x in out.outputs]
                payload.append({"prompt": out.prompt, "texts": texts})
            resp_q.put({"ok": True, "payload": payload})

    except Exception as e:
        ready_q.put({"ok": False, "error": repr(e)})


class VLLMProxy:
    def __init__(
        self, visible_gpu_id, model_name, trust_remote_code=True,
        gpu_memory_utilization=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len=DEFAULT_VLLM_MAX_MODEL_LEN,
    ):
        self.ctx = mp.get_context("spawn")
        self.req_q = self.ctx.Queue()
        self.resp_q = self.ctx.Queue()
        self.ready_q = self.ctx.Queue()
        self.proc = self.ctx.Process(
            target=_vllm_worker_main,
            args=(
                visible_gpu_id, model_name, trust_remote_code,
                gpu_memory_utilization, max_model_len,
                self.req_q, self.resp_q, self.ready_q,
            ),
        )
        self.proc.start()
        self._ready = False

    def wait_until_ready(self):
        if self._ready:
            return
        msg = self.ready_q.get()
        if not msg["ok"]:
            raise RuntimeError(f"vLLM worker failed to start: {msg['error']}")
        self._ready = True

    def generate(self, prompts, sampling_params):
        self.wait_until_ready()
        sampling_dict = _serialize_sampling_params(sampling_params)
        self.req_q.put({"cmd": "generate", "prompts": prompts, "sampling": sampling_dict})
        msg = self.resp_q.get()
        if not msg["ok"]:
            raise RuntimeError(f"vLLM generate failed: {msg['error']}")
        results = []
        for item in msg["payload"]:
            results.append(_ProxyRequestOutput(item["prompt"], item["texts"]))
        return results

    def shutdown(self):
        try:
            self.req_q.put(None)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                self.proc.terminate()


# ─────────────────────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t):
        return " ".join(t.split())
    def remove_punc(t):
        exc = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exc)
    return white_space_fix(remove_articles(remove_punc(s.lower().strip())))


def counter_common(p, g):
    from collections import Counter
    pc = Counter(p)
    gc = Counter(g)
    return sum((pc & gc).values())


def compute_f1(pred, gold):
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    common = counter_common(p_toks, g_toks)
    if not common:
        return 0.0
    prec = common / len(p_toks) if p_toks else 0.0
    rec = common / len(g_toks) if g_toks else 0.0
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_em(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def score_answer(pred: str, gold: str) -> float:
    em = compute_em(pred, gold)
    f1 = compute_f1(pred, gold)
    return max(em, f1)


# ─────────────────────────────────────────────────────────────
#  Planner helpers
# ─────────────────────────────────────────────────────────────
def parse_steps(text: str) -> list:
    steps = re.findall(r"Step\s+\d+:\s*(.+)", text, re.IGNORECASE)
    return [s.strip() for s in steps if s.strip()]


def replace_placeholders(question: str, hop_answers: list) -> str:
    result = question
    for i, ans in enumerate(hop_answers, start=1):
        result = result.replace(f"#{i}", ans)
    return result


def extract_final_answer(text: str) -> str:
    if "Final answer:" in text:
        return text.split("Final answer:")[-1].strip()
    return text.strip()


def format_plan(steps: list) -> str:
    return "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))


def build_reasoning_chain(sub_questions, hop_answers):
    lines = []
    for idx, (q, a) in enumerate(zip(sub_questions, hop_answers), start=1):
        lines.append(f"Q{idx}: {q}  ->  A{idx}: {a}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  Critic output parser
# ─────────────────────────────────────────────────────────────
def parse_critic_output(text: str):
    verdict_match = re.search(
        r"<verdict>\s*(ACCEPT|REJECT)\s*</verdict>",
        text,
        re.IGNORECASE,
    )
    verdict = verdict_match.group(1).upper() if verdict_match else "ACCEPT"

    feedback = None
    if verdict == "REJECT":
        fb_match = re.search(r"<feedback>(.*?)</feedback>", text, re.IGNORECASE | re.DOTALL)
        feedback = fb_match.group(1).strip() if fb_match else "The plan needs revision."

    return verdict, feedback


# ─────────────────────────────────────────────────────────────
#  Planner inference (one question at a time, greedy)
# ─────────────────────────────────────────────────────────────
@torch.inference_mode()
def planner_decompose(question, planner_model, planner_tok, feedback=None):
    user_content = f"Decompose:\n{question}"
    if feedback:
        user_content += (
            f"\n\nPrevious plan was rejected. Feedback: {feedback}\nPlease revise the plan."
        )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = planner_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    planner_device = get_model_device(planner_model)
    enc = planner_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(planner_device)
    out = planner_model.generate(
        **enc, max_new_tokens=256, do_sample=False, pad_token_id=planner_tok.pad_token_id,
    )
    text = planner_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_steps(text)


# ─────────────────────────────────────────────────────────────
#  Agent Loop (QA execution using vLLM)
# ─────────────────────────────────────────────────────────────
def run_agent_loop(question, sub_questions, context, llm, qa_tok):
    sampling = {"max_tokens": QA_MAX_TOKENS, "temperature": 0.0}
    hop_answers = []

    for sub_q_raw in sub_questions:
        sub_q = replace_placeholders(sub_q_raw, hop_answers)
        prompt_text = HOP_PROMPT.format(context=context, question=sub_q)
        messages = [
            {"role": "system", "content": HOP_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        text = qa_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([text], sampling)
        ans = extract_final_answer(outputs[0].outputs[0].text)
        hop_answers.append(ans)

    reasoning = build_reasoning_chain(sub_questions, hop_answers)
    synth_text = SYNTHESIS_PROMPT.format(context=context, reasoning=reasoning, question=question)
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": synth_text},
    ]
    text = qa_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([text], sampling)
    return extract_final_answer(outputs[0].outputs[0].text)


# ─────────────────────────────────────────────────────────────
#  Critic reward assignment (base — used inside dual-objective)
# ─────────────────────────────────────────────────────────────
def compute_critic_reward(verdict, answer_score, threshold=0.5):
    correct = answer_score >= threshold
    if verdict == "ACCEPT":
        return REWARD_ACCEPT_CORRECT if correct else REWARD_ACCEPT_WRONG
    else:
        return REWARD_REJECT_CORRECT if correct else REWARD_REJECT_STILL_BAD


# ─────────────────────────────────────────────────────────────
#  Critic tokenisation helpers
# ─────────────────────────────────────────────────────────────
def build_critic_prompt(question, plan, critic_tok):
    plan_text = format_plan(plan)
    user_content = CRITIC_USER_PROMPT.format(question=question, plan=plan_text)
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return critic_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─────────────────────────────────────────────────────────────
#  Critic sampling — free samples (unchanged from v1)
# ─────────────────────────────────────────────────────────────
def sample_critic_responses(
    prompt, critic_model, critic_tok, num_samples, temperature, max_new_tokens,
):
    critic_device = get_model_device(critic_model)
    enc = critic_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(critic_device)
    input_len = enc["input_ids"].shape[1]

    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": critic_tok.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    else:
        gen_kwargs["do_sample"] = False

    batched_enc = {k: v.repeat(num_samples, 1) for k, v in enc.items()}
    shared_input_ids = enc["input_ids"].clone()
    shared_attention_mask = enc["attention_mask"].clone()

    with torch.inference_mode():
        out = critic_model.generate(**batched_enc, **gen_kwargs)

    results = []
    for sequence in out.sequences:
        response_ids = sequence[input_len:]
        text = critic_tok.decode(response_ids, skip_special_tokens=True).strip()
        results.append({
            "text": text,
            "input_ids": shared_input_ids,
            "attention_mask": shared_attention_mask,
            "response_ids": response_ids.clone(),
            "guided_prefix_len": 0,          # ← v2: free sample
        })
    return results


# ─────────────────────────────────────────────────────────────
#  v2: Guided sample generation (for stratified sampling)
# ─────────────────────────────────────────────────────────────
@torch.inference_mode()
def _generate_guided_sample(prompt, guided_prefix, critic_model, critic_tok, max_new_tokens):
    """
    Append *guided_prefix* (e.g. "<verdict>ACCEPT") to the prompt, then
    let the model continue with greedy decoding.
    Returns a sample dict whose ``guided_prefix_len`` records how many
    response tokens belong to the forced prefix so they can be masked
    from log-prob computation later.
    """
    critic_device = get_model_device(critic_model)

    # tokenise prompt
    enc = critic_tok(
        prompt, return_tensors="pt", truncation=True, max_length=1024,
    ).to(critic_device)
    prompt_input_ids = enc["input_ids"]            # (1, P)
    prompt_attention_mask = enc["attention_mask"]   # (1, P)
    prompt_len = prompt_input_ids.shape[1]

    # tokenise guided prefix (no special tokens)
    prefix_enc = critic_tok(
        guided_prefix, add_special_tokens=False, return_tensors="pt",
    )
    prefix_ids = prefix_enc["input_ids"].to(critic_device)     # (1, G)
    prefix_len = prefix_ids.shape[1]

    # concat prompt + prefix → new "input" for generate()
    full_input_ids = torch.cat([prompt_input_ids, prefix_ids], dim=1)
    full_attention_mask = torch.ones_like(full_input_ids)

    out = critic_model.generate(
        input_ids=full_input_ids,
        attention_mask=full_attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=critic_tok.pad_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )

    # response = everything after original prompt (prefix + continuation)
    response_ids = out.sequences[0][prompt_len:]
    text = critic_tok.decode(response_ids, skip_special_tokens=True).strip()

    return {
        "text": text,
        "input_ids": prompt_input_ids.clone(),
        "attention_mask": prompt_attention_mask.clone(),
        "response_ids": response_ids.clone(),
        "guided_prefix_len": prefix_len,       # ← tokens to mask in log-prob
    }


# ─────────────────────────────────────────────────────────────
#  v2: Stratified sampling  (改动二)
# ─────────────────────────────────────────────────────────────
def sample_critic_responses_stratified(
    prompt, critic_model, critic_tok, group_size, temperature, max_new_tokens,
):
    """
    Returns *group_size* critic samples guaranteed to contain at least
    one ACCEPT and one REJECT verdict:
      - (group_size - 2) free samples  (temperature sampling)
      - 1 guided ACCEPT
      - 1 guided REJECT
    If a guided sample produces a broken format, we fall back to a free
    sample so the total count stays at group_size.
    """
    assert group_size >= 4, (
        f"group_size must be >= 4 for stratified sampling, got {group_size}"
    )

    free_count = group_size - 2

    # ── free samples ─────────────────────────────────────────
    free_samples = sample_critic_responses(
        prompt, critic_model, critic_tok, free_count, temperature, max_new_tokens,
    )

    # ── guided ACCEPT ────────────────────────────────────────
    try:
        guided_accept = _generate_guided_sample(
            prompt, "<verdict>ACCEPT", critic_model, critic_tok, max_new_tokens,
        )
        v_a, _ = parse_critic_output(guided_accept["text"])
        if v_a != "ACCEPT":
            # format broke — replace with a free sample
            guided_accept = sample_critic_responses(
                prompt, critic_model, critic_tok, 1, temperature, max_new_tokens,
            )[0]
    except Exception:
        guided_accept = sample_critic_responses(
            prompt, critic_model, critic_tok, 1, temperature, max_new_tokens,
        )[0]

    # ── guided REJECT ────────────────────────────────────────
    try:
        guided_reject = _generate_guided_sample(
            prompt, "<verdict>REJECT", critic_model, critic_tok, max_new_tokens,
        )
        v_r, _ = parse_critic_output(guided_reject["text"])
        if v_r != "REJECT":
            guided_reject = sample_critic_responses(
                prompt, critic_model, critic_tok, 1, temperature, max_new_tokens,
            )[0]
    except Exception:
        guided_reject = sample_critic_responses(
            prompt, critic_model, critic_tok, 1, temperature, max_new_tokens,
        )[0]

    return free_samples + [guided_accept, guided_reject]


# ─────────────────────────────────────────────────────────────
#  Recompute log p(response | prompt) — v2: per-token avg + prefix mask
# ─────────────────────────────────────────────────────────────
def _forward_for_logprob(model, full_ids, full_attention_mask, use_base_without_adapter=False):
    if use_base_without_adapter:
        if not hasattr(model, "disable_adapter"):
            raise RuntimeError(
                "critic_model does not expose disable_adapter(); "
                "cannot compute base reference logprob."
            )
        with model.disable_adapter():
            return model(input_ids=full_ids, attention_mask=full_attention_mask, use_cache=False)
    return model(input_ids=full_ids, attention_mask=full_attention_mask, use_cache=False)


def compute_batched_response_logprobs(
    model,
    input_ids,
    attention_mask,
    response_ids_list,
    use_base_without_adapter=False,
    guided_prefix_lens=None,
):
    """
    v2 changes vs v1
    ----------------
    1. Accepts *guided_prefix_lens* — a list[int] parallel to
       *response_ids_list*.  Tokens in the guided prefix are masked
       from the log-prob computation (but still attend during the
       forward pass so the continuation is properly conditioned).
    2. Returns **per-token average** log-prob instead of sum, which
       keeps the loss scale independent of response length.
    """
    model_device = get_model_device(model)

    if len(response_ids_list) == 0:
        return torch.empty(0, device=model_device, dtype=torch.float32)

    input_ids = input_ids.to(model_device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=model_device)
    else:
        attention_mask = attention_mask.to(model_device)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    batch_size = len(response_ids_list)
    prompt_ids = input_ids.expand(batch_size, -1)
    prompt_attention_mask = attention_mask.expand(batch_size, -1)

    response_lengths = [int(r.numel()) for r in response_ids_list]
    max_response_len = max(response_lengths)

    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", 0)

    response_batch = torch.full(
        (batch_size, max_response_len),
        fill_value=pad_token_id,
        dtype=prompt_ids.dtype,
        device=model_device,
    )
    # attention mask for forward pass — includes guided prefix
    response_attn = torch.zeros(
        (batch_size, max_response_len),
        dtype=prompt_attention_mask.dtype,
        device=model_device,
    )

    for idx, response_ids in enumerate(response_ids_list):
        response_ids = response_ids.to(model_device).view(-1)
        cur_len = response_ids.numel()
        if cur_len == 0:
            continue
        response_batch[idx, :cur_len] = response_ids
        response_attn[idx, :cur_len] = 1

    # ── logprob mask: exclude padding AND guided-prefix tokens ──
    logprob_mask = response_attn.clone()
    if guided_prefix_lens is not None:
        for idx, prefix_len in enumerate(guided_prefix_lens):
            if prefix_len > 0:
                mask_end = min(prefix_len, max_response_len)
                logprob_mask[idx, :mask_end] = 0

    # ── forward pass — chunked to avoid OOM on large groups ──
    full_ids = torch.cat([prompt_ids, response_batch], dim=1)
    full_attention_mask = torch.cat([prompt_attention_mask, response_attn], dim=1)
    prompt_len = prompt_ids.shape[1]

    MICRO_BS = 2  # process 2 samples at a time to stay within VRAM
    all_token_log_probs = []

    for chunk_start in range(0, batch_size, MICRO_BS):
        chunk_end = min(chunk_start + MICRO_BS, batch_size)

        chunk_ids = full_ids[chunk_start:chunk_end]
        chunk_mask = full_attention_mask[chunk_start:chunk_end]

        outputs = _forward_for_logprob(
            model, chunk_ids, chunk_mask,
            use_base_without_adapter=use_base_without_adapter,
        )

        chunk_logits = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + max_response_len, :]
        chunk_log_probs = F.log_softmax(chunk_logits, dim=-1)

        chunk_token_lp = chunk_log_probs.gather(
            dim=-1,
            index=response_batch[chunk_start:chunk_end].unsqueeze(-1),
        ).squeeze(-1)

        all_token_log_probs.append(chunk_token_lp)

        # free intermediate tensors eagerly
        del outputs, chunk_logits, chunk_log_probs
        torch.cuda.empty_cache()

    token_log_probs = torch.cat(all_token_log_probs, dim=0)

    # apply logprob mask (zeros out padding + guided-prefix positions)
    token_log_probs = token_log_probs * logprob_mask.to(token_log_probs.dtype)

    # ── per-token average (v2) ──────────────────────────────
    effective_lengths = logprob_mask.sum(dim=1).clamp(min=1)
    return token_log_probs.sum(dim=1) / effective_lengths


# ─────────────────────────────────────────────────────────────
#  v2: GRPO loss with CIV advantage  (改动一)
# ─────────────────────────────────────────────────────────────
def grpo_loss_civ(
    samples,
    rewards,
    verdicts,
    critic_model,
    kl_coef=KL_COEF,
    civ_margin=CIV_MARGIN,
):
    """
    Counterfactual Instrumental Variable (CIV) advantage:
        A_k =  (r̄_A - r̄_R)   if verdict_k == ACCEPT
        A_k = -(r̄_A - r̄_R)   if verdict_k == REJECT
    with a dead-zone margin to avoid oscillation on noisy signal.
    """
    critic_device = get_model_device(critic_model)

    # ── CIV advantage ───────────────────────────────────────
    accept_rewards = [r for r, v in zip(rewards, verdicts) if v == "ACCEPT"]
    reject_rewards = [r for r, v in zip(rewards, verdicts) if v == "REJECT"]

    r_bar_A = sum(accept_rewards) / len(accept_rewards) if accept_rewards else 0.0
    r_bar_R = sum(reject_rewards) / len(reject_rewards) if reject_rewards else 0.0

    raw_adv = r_bar_A - r_bar_R
    if abs(raw_adv) < civ_margin:
        raw_adv = 0.0

    advantages = []
    for v in verdicts:
        advantages.append(raw_adv if v == "ACCEPT" else -raw_adv)

    # ── verdict-balanced weighting ─────────────────────────
    # Without this, 7 ACCEPT vs 1 REJECT means the REJECT gradient
    # signal is drowned out.  We give each verdict TYPE equal total
    # weight: w_k = 1/count(verdict_k), then normalise so Σw = N.
    n_accept = sum(1 for v in verdicts if v == "ACCEPT")
    n_reject = sum(1 for v in verdicts if v == "REJECT")
    per_sample_weight = []
    for v in verdicts:
        if v == "ACCEPT" and n_accept > 0:
            per_sample_weight.append(1.0 / n_accept)
        elif v == "REJECT" and n_reject > 0:
            per_sample_weight.append(1.0 / n_reject)
        else:
            per_sample_weight.append(1.0)
    # normalise so weights sum to len(verdicts)
    w_sum = sum(per_sample_weight)
    if w_sum > 0:
        factor = len(verdicts) / w_sum
        per_sample_weight = [w * factor for w in per_sample_weight]

    # ── collect valid samples ───────────────────────────────
    valid_response_ids = []
    valid_advantages = []
    valid_guided_prefix_lens = []
    valid_weights = []
    prompt_input_ids = None
    prompt_attention_mask = None

    for sample, adv, w in zip(samples, advantages, per_sample_weight):
        response_ids = sample["response_ids"]
        if response_ids.numel() == 0:
            continue
        if prompt_input_ids is None:
            prompt_input_ids = sample["input_ids"]
            prompt_attention_mask = sample["attention_mask"]
        valid_response_ids.append(response_ids)
        valid_advantages.append(adv)
        valid_guided_prefix_lens.append(sample.get("guided_prefix_len", 0))
        valid_weights.append(w)

    if len(valid_response_ids) == 0:
        dummy = None
        for p in critic_model.parameters():
            if p.requires_grad:
                dummy = p
                break
        if dummy is None:
            raise RuntimeError("No trainable parameters found in critic_model.")
        return dummy.sum() * 0.0

    # ── policy log-prob (per-token avg) ─────────────────────
    avg_log_probs = compute_batched_response_logprobs(
        critic_model,
        prompt_input_ids,
        prompt_attention_mask,
        valid_response_ids,
        use_base_without_adapter=False,
        guided_prefix_lens=valid_guided_prefix_lens,
    )

    # ── reference log-prob (base model w/o adapter) ─────────
    with torch.no_grad():
        ref_avg_log_probs = compute_batched_response_logprobs(
            critic_model,
            prompt_input_ids,
            prompt_attention_mask,
            valid_response_ids,
            use_base_without_adapter=True,
            guided_prefix_lens=valid_guided_prefix_lens,
        )

    valid_advantages_t = torch.tensor(
        valid_advantages, dtype=torch.float32, device=critic_device,
    )
    valid_weights_t = torch.tensor(
        valid_weights, dtype=torch.float32, device=critic_device,
    )
    kl = avg_log_probs - ref_avg_log_probs.to(avg_log_probs.device)
    pg_loss = -valid_advantages_t * avg_log_probs
    sample_losses = pg_loss + kl_coef * kl
    # weighted mean: gives ACCEPT and REJECT equal total influence
    return (sample_losses * valid_weights_t).sum() / valid_weights_t.sum()


# ─────────────────────────────────────────────────────────────
#  Single-stage training function  (改动一+二+三 integrated)
# ─────────────────────────────────────────────────────────────
def train_one_stage(
    stage_data, stage_id, critic_model, critic_tok,
    planner_model, planner_tok, llm, qa_tok,
    optimizer, scheduler, args, global_step,
    planner_cache, agent_loop_cache, critic_prompt_cache,
):
    random.shuffle(stage_data)

    log_stats = defaultdict(list)
    batch_loss_acc = None
    optimizer.zero_grad(set_to_none=True)

    critic_review_cache = {}
    group_size = args.group_size

    def cached_planner(question, feedback=None):
        key = planner_cache_key(question, feedback)
        if key not in planner_cache:
            planner_cache[key] = planner_decompose(
                question, planner_model, planner_tok, feedback=feedback,
            )
        return planner_cache[key]

    def cached_zero_temp_critic(question, final_plan):
        prompt = build_critic_prompt(question, final_plan, critic_tok)
        if prompt not in critic_review_cache:
            re_samples = sample_critic_responses(
                prompt, critic_model, critic_tok,
                num_samples=1, temperature=0.0,
                max_new_tokens=args.max_new_tokens,
            )
            critic_review_cache[prompt] = parse_critic_output(re_samples[0]["text"])
        return critic_review_cache[prompt]

    def cached_agent_loop(question, final_plan, context):
        key = agent_loop_cache_key(question, final_plan, context)
        if key not in agent_loop_cache:
            agent_loop_cache[key] = run_agent_loop(
                question, final_plan, context, llm, qa_tok,
            )
        return agent_loop_cache[key]

    for sample_idx, data in enumerate(tqdm(stage_data, desc=f"  Stage {stage_id} training")):
        question = data["question"]
        ground_truth = data["answer"]
        context = data["context_str"]

        # ── Step 1: Planner → initial plan ───────────────────
        initial_plan = cached_planner(question)
        if not initial_plan:
            continue

        # ── Step 2: Run agent loop on ORIGINAL plan (baseline for dual-objective) ──
        original_pred = cached_agent_loop(question, initial_plan, context)
        original_score = score_answer(original_pred, ground_truth)

        # ── Step 3: Build critic prompt ──────────────────────
        prompt_key = critic_prompt_cache_key(question, initial_plan)
        if prompt_key not in critic_prompt_cache:
            critic_prompt_cache[prompt_key] = build_critic_prompt(
                question, initial_plan, critic_tok,
            )
        critic_prompt = critic_prompt_cache[prompt_key]

        # ── Step 4: Stratified group sampling  (改动二) ──────
        critic_model.train()
        group_samples = sample_critic_responses_stratified(
            critic_prompt,
            critic_model,
            critic_tok,
            group_size=group_size,
            temperature=args.critic_temp,
            max_new_tokens=args.max_new_tokens,
        )

        # ── Step 5: Evaluate each sample, compute dual-objective reward  (改动三) ──
        group_rewards = []
        group_verdicts = []

        for sample in group_samples:
            verdict, feedback = parse_critic_output(sample["text"])
            group_verdicts.append(verdict)

            if verdict == "ACCEPT":
                # ACCEPT → use original plan; reward is just outcome reward
                pred_answer = original_pred
                ans_score = original_score
                reward = compute_critic_reward(verdict, ans_score)
            else:
                # REJECT → replan, then evaluate
                final_plan = initial_plan
                for _ in range(MAX_RETRIES):
                    revised = cached_planner(question, feedback=feedback)
                    if revised:
                        final_plan = revised
                        v2, feedback = cached_zero_temp_critic(question, final_plan)
                        if v2 == "ACCEPT":
                            break

                pred_answer = cached_agent_loop(question, final_plan, context)
                ans_score = score_answer(pred_answer, ground_truth)

                # ── dual-objective reward ────────────────────
                r_verdict = compute_critic_reward(verdict, ans_score)
                r_feedback = ans_score - original_score   # marginal contribution of feedback
                reward = r_verdict + args.lambda_feedback * r_feedback

            group_rewards.append(reward)

            log_stats["em"].append(compute_em(pred_answer, ground_truth))
            log_stats["f1"].append(compute_f1(pred_answer, ground_truth))
            log_stats["reward"].append(reward)
            log_stats["verdict"].append(1 if verdict == "ACCEPT" else 0)

        # ── Step 6: CIV GRPO loss  (改动一) ─────────────────
        loss = grpo_loss_civ(
            group_samples,
            group_rewards,
            group_verdicts,
            critic_model,
            kl_coef=KL_COEF,
            civ_margin=args.civ_margin,
        )
        loss = loss / args.batch_size
        loss.backward()

        detached_loss = loss.detach()
        if batch_loss_acc is None:
            batch_loss_acc = detached_loss
        else:
            batch_loss_acc = batch_loss_acc + detached_loss

        # ── Step 7: Gradient update ──────────────────────────
        if (sample_idx + 1) % args.batch_size == 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, critic_model.parameters()),
                GRAD_CLIP,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_every == 0:
                window = 50
                avg_em = sum(log_stats["em"][-window:]) / max(len(log_stats["em"][-window:]), 1)
                avg_f1 = sum(log_stats["f1"][-window:]) / max(len(log_stats["f1"][-window:]), 1)
                avg_rew = sum(log_stats["reward"][-window:]) / max(len(log_stats["reward"][-window:]), 1)
                acc_rate = sum(log_stats["verdict"][-window:]) / max(len(log_stats["verdict"][-window:]), 1)

                loss_value = 0.0 if batch_loss_acc is None else float(batch_loss_acc.item())
                print(
                    f"    Step {global_step:5d} | loss={loss_value:.4f} | "
                    f"EM={avg_em:.3f}  F1={avg_f1:.3f} | "
                    f"reward={avg_rew:.3f} | accept_rate={acc_rate:.2f}"
                )
                batch_loss_acc = None

            if global_step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"stage{stage_id}_step{global_step}")
                critic_model.save_pretrained(ckpt)
                critic_tok.save_pretrained(ckpt)
                print(f"    Checkpoint saved → {ckpt}")

    final_metrics = {
        "EM": sum(log_stats["em"]) / max(len(log_stats["em"]), 1),
        "F1": sum(log_stats["f1"]) / max(len(log_stats["f1"]), 1),
        "avg_reward": sum(log_stats["reward"]) / max(len(log_stats["reward"]), 1),
        "accept_rate": sum(log_stats["verdict"]) / max(len(log_stats["verdict"]), 1),
    }
    return global_step, final_metrics


# ─────────────────────────────────────────────────────────────
#  Main training loop (dual-GPU, curriculum-gated)
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This script expects at least 2 GPUs.")
    if args.qa_gpu_id < 0 or args.qa_gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid --qa_gpu_id={args.qa_gpu_id}. Found {torch.cuda.device_count()} visible GPU(s)."
        )
    if args.train_gpu_id < 0 or args.train_gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid --train_gpu_id={args.train_gpu_id}. Found {torch.cuda.device_count()} visible GPU(s)."
        )
    if args.qa_gpu_id == args.train_gpu_id:
        raise RuntimeError(
            f"qa_gpu_id and train_gpu_id must be different. Got both={args.qa_gpu_id}."
        )

    # ── Load dataset, split by stage ─────────────────────────
    print(f"Loading {args.train_file} ...")
    stage_data = defaultdict(list)
    with open(args.train_file, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            s = d.get("curriculum_stage", 1)
            d["context_str"] = "\n\n---\n\n".join(d["pred_texts"])
            stage_data[s].append(d)

    for s in sorted(stage_data):
        print(f"  Stage {s}: {len(stage_data[s])} samples")

    planner_cache_path = args.planner_cache_path or os.path.join(args.output_dir, "planner_cache.pkl")
    qa_cache_path = args.qa_cache_path or os.path.join(args.output_dir, "qa_cache.pkl")
    planner_cache = load_pickle_cache(planner_cache_path)
    agent_loop_cache = load_pickle_cache(qa_cache_path)
    critic_prompt_cache = {}

    print(f"Loaded planner cache: {len(planner_cache)} entries")
    print(f"Loaded QA cache: {len(agent_loop_cache)} entries")

    # ── Step 1: vLLM on dedicated QA GPU ─────────────────────
    print(f"\n[1/3] Loading QA model via vLLM on GPU{args.qa_gpu_id} ...")
    qa_tok = AutoTokenizer.from_pretrained(args.qa_model, trust_remote_code=True)

    llm = VLLMProxy(
        visible_gpu_id=args.qa_gpu_id,
        model_name=args.qa_model,
        trust_remote_code=True,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
    )
    atexit.register(llm.shutdown)

    llm.wait_until_ready()
    print(f"QA model ready on GPU{args.qa_gpu_id}.")

    # ── Step 2: main process uses dedicated training GPU ─────
    torch.cuda.set_device(args.train_gpu_id)

    # ── Load Planner (frozen) on training GPU ────────────────
    print(f"\n[2/3] Loading Planner (frozen) on GPU{args.train_gpu_id} ...")
    planner_tok = AutoTokenizer.from_pretrained(args.planner_base, trust_remote_code=True)
    planner_tok.pad_token = planner_tok.eos_token
    planner_tok.padding_side = "left"

    planner_base_model = AutoModelForCausalLM.from_pretrained(
        args.planner_base,
        torch_dtype=torch.bfloat16,
        device_map={"": args.train_gpu_id},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    planner_model = PeftModel.from_pretrained(planner_base_model, args.planner_lora)
    planner_model.eval()
    for p in planner_model.parameters():
        p.requires_grad_(False)
    print(f"Planner ready on GPU{args.train_gpu_id} (frozen).")

    # ── Load Critic (trainable) on training GPU ──────────────
    print(f"\n[3/3] Loading Critic on GPU{args.train_gpu_id} ...")
    critic_tok = AutoTokenizer.from_pretrained(args.critic_base, trust_remote_code=True)
    critic_tok.pad_token = critic_tok.eos_token
    critic_tok.padding_side = "left"

    critic_base_model = AutoModelForCausalLM.from_pretrained(
        args.critic_base,
        torch_dtype=torch.bfloat16,
        device_map={"": args.train_gpu_id},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    critic_model = get_peft_model(critic_base_model, lora_cfg)
    critic_model.print_trainable_parameters()
    print(f"Critic ready on GPU{args.train_gpu_id}.")

    # ── Optimiser & scheduler ────────────────────────────────
    total_samples = sum(len(v) for v in stage_data.values())
    total_steps = max((total_samples * args.epochs) // args.batch_size, 1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, critic_model.parameters()),
        lr=LR,
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    # ── Curriculum-gated training ────────────────────────────
    print(
        f"\nStarting curriculum GRPO v2 (CIV + Stratified + Dual-Obj) | "
        f"epochs={args.epochs} | batch={args.batch_size} | G={args.group_size} | "
        f"λ_fb={args.lambda_feedback} | civ_margin={args.civ_margin} | "
        f"qa_gpu=GPU{args.qa_gpu_id} | train_gpu=GPU{args.train_gpu_id}"
    )

    global_step = 0
    training_log = []

    for stage_id in sorted(stage_data.keys()):
        threshold = STAGE_THRESHOLDS.get(stage_id, {"EM": 0.0, "F1": 0.0})
        retrain_ct = 0
        passed = False

        print(f"\n{'=' * 55}")
        print(f"  STAGE {stage_id}  |  target EM≥{threshold['EM']}  F1≥{threshold['F1']}")
        print(f"  samples: {len(stage_data[stage_id])}")
        print(f"{'=' * 55}")

        while not passed and retrain_ct <= MAX_STAGE_RETRAIN:
            if retrain_ct > 0:
                print(
                    f"\n  [Stage {stage_id}] Threshold not met — retraining "
                    f"(attempt {retrain_ct}/{MAX_STAGE_RETRAIN}) ..."
                )

            for epoch in range(args.epochs):
                print(f"\n  Epoch {epoch + 1}/{args.epochs}")
                global_step, metrics = train_one_stage(
                    stage_data=stage_data[stage_id],
                    stage_id=stage_id,
                    critic_model=critic_model,
                    critic_tok=critic_tok,
                    planner_model=planner_model,
                    planner_tok=planner_tok,
                    llm=llm,
                    qa_tok=qa_tok,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    global_step=global_step,
                    planner_cache=planner_cache,
                    agent_loop_cache=agent_loop_cache,
                    critic_prompt_cache=critic_prompt_cache,
                )

                save_pickle_cache(planner_cache, planner_cache_path)
                save_pickle_cache(agent_loop_cache, qa_cache_path)

                em_ok = metrics["EM"] >= threshold["EM"]
                f1_ok = metrics["F1"] >= threshold["F1"]
                passed = em_ok and f1_ok
                if passed and args.stop_stage_after_pass:
                    print(
                        f"\n  Stage {stage_id} passed after epoch {epoch + 1}; "
                        f"stopping remaining epochs for this attempt."
                    )
                    break

            em_ok = metrics["EM"] >= threshold["EM"]
            f1_ok = metrics["F1"] >= threshold["F1"]
            passed = em_ok and f1_ok

            status = "PASSED ✓" if passed else "FAILED ✗"
            print(
                f"\n  Stage {stage_id} result: "
                f"EM={metrics['EM']:.4f} (≥{threshold['EM']}) {'✓' if em_ok else '✗'}  |  "
                f"F1={metrics['F1']:.4f} (≥{threshold['F1']}) {'✓' if f1_ok else '✗'}  "
                f"→ {status}"
            )
            print(
                f"  avg_reward={metrics['avg_reward']:.3f}  "
                f"accept_rate={metrics['accept_rate']:.2f}"
            )

            training_log.append({
                "stage": stage_id,
                "retrain_attempt": retrain_ct,
                **metrics,
                "passed": passed,
            })

            stage_ckpt = os.path.join(
                args.output_dir,
                f"stage{stage_id}_attempt{retrain_ct}_{'pass' if passed else 'fail'}",
            )
            critic_model.save_pretrained(stage_ckpt)
            critic_tok.save_pretrained(stage_ckpt)
            save_pickle_cache(planner_cache, planner_cache_path)
            save_pickle_cache(agent_loop_cache, qa_cache_path)
            print(f"  Checkpoint saved → {stage_ckpt}")

            retrain_ct += 1

        if not passed:
            print(
                f"\n  WARNING: Stage {stage_id} did not meet threshold after "
                f"{MAX_STAGE_RETRAIN} retrains. Proceeding to next stage anyway."
            )

    # ── Final save ────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    critic_model.save_pretrained(final_dir)
    critic_tok.save_pretrained(final_dir)
    save_pickle_cache(planner_cache, planner_cache_path)
    save_pickle_cache(agent_loop_cache, qa_cache_path)

    # ── Training summary ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE — CURRICULUM SUMMARY (v2)")
    print("=" * 55)
    for entry in training_log:
        tag = "PASS" if entry["passed"] else "FAIL"
        print(
            f"  Stage {entry['stage']} attempt {entry['retrain_attempt']} | "
            f"EM={entry['EM']:.4f}  F1={entry['F1']:.4f}  "
            f"reward={entry['avg_reward']:.3f}  [{tag}]"
        )

    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    print(f"\n  Log saved → {log_path}")
    print(f"  Final model → {final_dir}")
    print("=" * 55)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
