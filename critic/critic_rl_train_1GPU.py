"""
Critic Agent RL Training via GRPO
===================================
Train a Plan-Critic from scratch using only outcome reward (EM/F1).
The Critic learns to Accept/Reject Planner output; its reward is derived
entirely from whether the final multi-hop answer is correct.

Single-GPU version:
    - No split-card setup
    - vLLM QA model, Planner, and Critic all use the same GPU
    - No separately loaded ref model; reference log-prob is computed by
      temporarily disabling the Critic's LoRA adapter
"""

import json
import os
import re
import string
import argparse
import atexit
import multiprocessing as mp
import random
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
GRPO_GROUP_SIZE = 4
MAX_RETRIES = 2
LR = 1e-5
WARMUP_STEPS = 50
GRAD_CLIP = 1.0
KL_COEF = 0.05
REWARD_ACCEPT_CORRECT = 1.0
REWARD_ACCEPT_WRONG = -1.0
REWARD_REJECT_CORRECT = 1.0
REWARD_REJECT_STILL_BAD = -0.5

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
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="Single GPU id used by vLLM + planner + critic",
    )
    p.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        help="Single-GPU mode: keep this conservative or vLLM may抢占太多显存",
    )
    p.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=DEFAULT_VLLM_MAX_MODEL_LEN,
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


# ─────────────────────────────────────────────────────────────
#  vLLM worker on same GPU (single-GPU mode)
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
    visible_gpu_id: int,
    model_name: str,
    trust_remote_code: bool,
    gpu_memory_utilization: float,
    max_model_len: int,
    req_q,
    resp_q,
    ready_q,
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
                payload.append(
                    {
                        "prompt": out.prompt,
                        "texts": texts,
                    }
                )

            resp_q.put({"ok": True, "payload": payload})

    except Exception as e:
        ready_q.put({"ok": False, "error": repr(e)})


class VLLMProxy:
    def __init__(
        self,
        visible_gpu_id: int,
        model_name: str,
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN,
    ):
        self.ctx = mp.get_context("spawn")
        self.req_q = self.ctx.Queue()
        self.resp_q = self.ctx.Queue()
        self.ready_q = self.ctx.Queue()

        self.proc = self.ctx.Process(
            target=_vllm_worker_main,
            args=(
                visible_gpu_id,
                model_name,
                trust_remote_code,
                gpu_memory_utilization,
                max_model_len,
                self.req_q,
                self.resp_q,
                self.ready_q,
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

        self.req_q.put(
            {
                "cmd": "generate",
                "prompts": prompts,
                "sampling": sampling_dict,
            }
        )

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
@torch.no_grad()
def planner_decompose(question: str, planner_model, planner_tok, feedback: str = None) -> list:
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
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    planner_device = get_model_device(planner_model)
    enc = planner_tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(planner_device)

    out = planner_model.generate(
        **enc,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=planner_tok.pad_token_id,
    )

    text = planner_tok.decode(
        out[0][enc["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return parse_steps(text)


# ─────────────────────────────────────────────────────────────
#  Agent Loop (QA execution using vLLM)
# ─────────────────────────────────────────────────────────────
def run_agent_loop(question: str, sub_questions: list, context: str, llm, qa_tok) -> str:
    sampling = {
        "max_tokens": QA_MAX_TOKENS,
        "temperature": 0.0,
    }

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
    synth_text = SYNTHESIS_PROMPT.format(
        context=context,
        reasoning=reasoning,
        question=question,
    )
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": synth_text},
    ]
    text = qa_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([text], sampling)
    return extract_final_answer(outputs[0].outputs[0].text)


# ─────────────────────────────────────────────────────────────
#  Critic reward assignment
# ─────────────────────────────────────────────────────────────
def compute_critic_reward(verdict: str, answer_score: float, threshold: float = 0.5) -> float:
    correct = answer_score >= threshold

    if verdict == "ACCEPT":
        return REWARD_ACCEPT_CORRECT if correct else REWARD_ACCEPT_WRONG
    else:
        return REWARD_REJECT_CORRECT if correct else REWARD_REJECT_STILL_BAD


# ─────────────────────────────────────────────────────────────
#  Critic tokenisation helpers
# ─────────────────────────────────────────────────────────────
def build_critic_prompt(question: str, plan: list, critic_tok) -> str:
    plan_text = format_plan(plan)
    user_content = CRITIC_USER_PROMPT.format(question=question, plan=plan_text)
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return critic_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─────────────────────────────────────────────────────────────
#  Critic sampling
#  Important: only sample here, do NOT use generate() scores for training.
# ─────────────────────────────────────────────────────────────
def sample_critic_responses(
    prompt: str,
    critic_model,
    critic_tok,
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
) -> list:
    critic_device = get_model_device(critic_model)

    enc = critic_tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(critic_device)

    input_len = enc["input_ids"].shape[1]
    results = []

    do_sample = temperature is not None and temperature > 0.0

    for _ in range(num_samples):
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

        with torch.no_grad():
            out = critic_model.generate(
                **enc,
                **gen_kwargs,
            )

        response_ids = out.sequences[0][input_len:]
        text = critic_tok.decode(response_ids, skip_special_tokens=True).strip()

        results.append(
            {
                "text": text,
                "input_ids": enc["input_ids"].clone(),
                "attention_mask": enc["attention_mask"].clone(),
                "response_ids": response_ids.clone(),
            }
        )

    return results


# ─────────────────────────────────────────────────────────────
#  Recompute log p(response | prompt)
# ─────────────────────────────────────────────────────────────
def _forward_for_logprob(model, full_ids, full_attention_mask, use_base_without_adapter=False):
    if use_base_without_adapter:
        if not hasattr(model, "disable_adapter"):
            raise RuntimeError(
                "critic_model does not expose disable_adapter(); cannot compute base reference logprob."
            )
        with model.disable_adapter():
            outputs = model(
                input_ids=full_ids,
                attention_mask=full_attention_mask,
                use_cache=False,
            )
        return outputs

    outputs = model(
        input_ids=full_ids,
        attention_mask=full_attention_mask,
        use_cache=False,
    )
    return outputs



def compute_response_logprob(
    model,
    input_ids,
    attention_mask,
    response_ids,
    use_base_without_adapter: bool = False,
):
    model_device = get_model_device(model)

    input_ids = input_ids.to(model_device)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=model_device)
    else:
        attention_mask = attention_mask.to(model_device)

    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    response_ids = response_ids.to(model_device)

    full_ids = torch.cat([input_ids, response_ids], dim=1)

    response_attention_mask = torch.ones_like(response_ids, device=model_device)
    full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=1)

    outputs = _forward_for_logprob(
        model,
        full_ids,
        full_attention_mask,
        use_base_without_adapter=use_base_without_adapter,
    )

    prompt_len = input_ids.shape[1]

    logits = outputs.logits[:, prompt_len - 1 : -1, :]
    log_probs = F.log_softmax(logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=response_ids.unsqueeze(-1),
    ).squeeze(-1)

    return token_log_probs.sum()


# ─────────────────────────────────────────────────────────────
#  GRPO loss
# ─────────────────────────────────────────────────────────────
def grpo_loss(
    samples: list,
    rewards: list,
    critic_model,
    kl_coef: float = KL_COEF,
) -> torch.Tensor:
    critic_device = get_model_device(critic_model)

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=critic_device)
    mean_r = rewards_t.mean()
    std_r = rewards_t.std(unbiased=False)
    advantages = (rewards_t - mean_r) / (std_r + 1e-8)

    total_loss = None
    valid_count = 0

    for sample, adv in zip(samples, advantages):
        response_ids = sample["response_ids"]
        if response_ids.numel() == 0:
            continue

        sum_log_prob = compute_response_logprob(
            critic_model,
            sample["input_ids"],
            sample["attention_mask"],
            response_ids,
            use_base_without_adapter=False,
        )

        with torch.no_grad():
            ref_sum_log_prob = compute_response_logprob(
                critic_model,
                sample["input_ids"],
                sample["attention_mask"],
                response_ids,
                use_base_without_adapter=True,
            )

        kl = sum_log_prob - ref_sum_log_prob.to(sum_log_prob.device)

        pg_loss = -adv * sum_log_prob
        sample_loss = pg_loss + kl_coef * kl

        if total_loss is None:
            total_loss = sample_loss
        else:
            total_loss = total_loss + sample_loss

        valid_count += 1

    if valid_count == 0:
        dummy = None
        for p in critic_model.parameters():
            if p.requires_grad:
                dummy = p
                break
        if dummy is None:
            raise RuntimeError("No trainable parameters found in critic_model.")
        return dummy.sum() * 0.0

    return total_loss / valid_count


# ─────────────────────────────────────────────────────────────
#  Single-stage training function
# ─────────────────────────────────────────────────────────────
def train_one_stage(
    stage_data: list,
    stage_id: int,
    critic_model,
    critic_tok,
    planner_model,
    planner_tok,
    llm,
    qa_tok,
    optimizer,
    scheduler,
    args,
    global_step: int,
):
    random.shuffle(stage_data)

    log_stats = defaultdict(list)
    batch_loss_acc = 0.0
    optimizer.zero_grad()

    planner_cache = {}
    critic_review_cache = {}
    agent_loop_cache = {}

    def cached_planner(question, feedback=None):
        key = (question, normalize_feedback(feedback))
        if key not in planner_cache:
            planner_cache[key] = planner_decompose(
                question,
                planner_model,
                planner_tok,
                feedback=feedback,
            )
        return planner_cache[key]

    def cached_zero_temp_critic(question, final_plan):
        prompt = build_critic_prompt(question, final_plan, critic_tok)
        if prompt not in critic_review_cache:
            re_samples = sample_critic_responses(
                prompt,
                critic_model,
                critic_tok,
                num_samples=1,
                temperature=0.0,
                max_new_tokens=args.max_new_tokens,
            )
            critic_review_cache[prompt] = parse_critic_output(re_samples[0]["text"])
        return critic_review_cache[prompt]

    def cached_agent_loop(question, final_plan, context):
        key = (question, tuple(final_plan), context)
        if key not in agent_loop_cache:
            agent_loop_cache[key] = run_agent_loop(
                question,
                final_plan,
                context,
                llm,
                qa_tok,
            )
        return agent_loop_cache[key]

    for sample_idx, data in enumerate(tqdm(stage_data, desc=f"  Stage {stage_id} training")):
        question = data["question"]
        ground_truth = data["answer"]
        context = "\n\n---\n\n".join(data["pred_texts"])

        # Step 1: Planner → initial plan
        initial_plan = cached_planner(question)
        if not initial_plan:
            continue

        # Step 2: Build critic prompt
        critic_prompt = build_critic_prompt(question, initial_plan, critic_tok)

        # Step 3: Sample G critic responses
        critic_model.train()
        group_samples = sample_critic_responses(
            critic_prompt,
            critic_model,
            critic_tok,
            num_samples=GRPO_GROUP_SIZE,
            temperature=args.critic_temp,
            max_new_tokens=args.max_new_tokens,
        )

        # Step 4: Run full pipeline for each sample, collect rewards
        group_rewards = []
        for sample in group_samples:
            verdict, feedback = parse_critic_output(sample["text"])

            final_plan = initial_plan
            for _ in range(MAX_RETRIES if verdict == "REJECT" else 0):
                revised = cached_planner(question, feedback=feedback)
                if revised:
                    final_plan = revised
                    verdict, feedback = cached_zero_temp_critic(question, final_plan)
                    if verdict == "ACCEPT":
                        break

            pred_answer = cached_agent_loop(question, final_plan, context)
            ans_score = score_answer(pred_answer, ground_truth)
            reward = compute_critic_reward(verdict, ans_score)
            group_rewards.append(reward)

            log_stats["em"].append(compute_em(pred_answer, ground_truth))
            log_stats["f1"].append(compute_f1(pred_answer, ground_truth))
            log_stats["reward"].append(reward)
            log_stats["verdict"].append(1 if verdict == "ACCEPT" else 0)

        # Step 5: GRPO loss
        loss = grpo_loss(
            group_samples,
            group_rewards,
            critic_model,
        )
        loss = loss / args.batch_size
        loss.backward()
        batch_loss_acc += float(loss.item())

        # Step 6: Gradient update
        if (sample_idx + 1) % args.batch_size == 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, critic_model.parameters()),
                GRAD_CLIP,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.log_every == 0:
                window = 50
                avg_em = sum(log_stats["em"][-window:]) / max(len(log_stats["em"][-window:]), 1)
                avg_f1 = sum(log_stats["f1"][-window:]) / max(len(log_stats["f1"][-window:]), 1)
                avg_rew = sum(log_stats["reward"][-window:]) / max(len(log_stats["reward"][-window:]), 1)
                acc_rate = sum(log_stats["verdict"][-window:]) / max(len(log_stats["verdict"][-window:]), 1)

                print(
                    f"    Step {global_step:5d} | loss={batch_loss_acc:.4f} | "
                    f"EM={avg_em:.3f}  F1={avg_f1:.3f} | "
                    f"reward={avg_rew:.3f} | accept_rate={acc_rate:.2f}"
                )
                batch_loss_acc = 0.0

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
#  Main training loop (single-GPU, curriculum-gated)
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("This script expects at least 1 GPU.")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid --gpu_id={args.gpu_id}. Found {torch.cuda.device_count()} visible GPU(s)."
        )

    # ── Load dataset, split by stage ─────────────────────────
    print(f"Loading {args.train_file} ...")
    stage_data = defaultdict(list)
    with open(args.train_file, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            s = d.get("curriculum_stage", 1)
            stage_data[s].append(d)

    for s in sorted(stage_data):
        print(f"  Stage {s}: {len(stage_data[s])} samples")

    # ── Step 1: vLLM on same GPU ─────────────────────────────
    print(f"\n[1/3] Loading QA model via vLLM on GPU{args.gpu_id} ...")
    qa_tok = AutoTokenizer.from_pretrained(args.qa_model, trust_remote_code=True)

    llm = VLLMProxy(
        visible_gpu_id=args.gpu_id,
        model_name=args.qa_model,
        trust_remote_code=True,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
    )
    atexit.register(llm.shutdown)

    llm.wait_until_ready()
    print(f"QA model ready on GPU{args.gpu_id}.")

    # ── Step 2: main process also use same GPU ───────────────
    torch.cuda.set_device(args.gpu_id)

    # ── Load Planner (frozen) on same GPU ────────────────────
    print(f"\n[2/3] Loading Planner (frozen) on GPU{args.gpu_id} ...")
    planner_tok = AutoTokenizer.from_pretrained(args.planner_base, trust_remote_code=True)
    planner_tok.pad_token = planner_tok.eos_token
    planner_tok.padding_side = "left"

    planner_base_model = AutoModelForCausalLM.from_pretrained(
        args.planner_base,
        torch_dtype=torch.bfloat16,
        device_map={"": args.gpu_id},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    planner_model = PeftModel.from_pretrained(planner_base_model, args.planner_lora)
    planner_model.eval()
    for p in planner_model.parameters():
        p.requires_grad_(False)
    print(f"Planner ready on GPU{args.gpu_id} (frozen).")

    # ── Load Critic (trainable) on same GPU ──────────────────
    print(f"\n[3/3] Loading Critic on GPU{args.gpu_id} ...")
    critic_tok = AutoTokenizer.from_pretrained(args.critic_base, trust_remote_code=True)
    critic_tok.pad_token = critic_tok.eos_token
    critic_tok.padding_side = "left"

    critic_base_model = AutoModelForCausalLM.from_pretrained(
        args.critic_base,
        torch_dtype=torch.bfloat16,
        device_map={"": args.gpu_id},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
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
    print(f"Critic ready on GPU{args.gpu_id}.")

    # ── Optimiser & scheduler ────────────────────────────────
    total_samples = sum(len(v) for v in stage_data.values())
    total_steps = max((total_samples * args.epochs) // args.batch_size, 1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, critic_model.parameters()),
        lr=LR,
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        WARMUP_STEPS,
        total_steps,
    )

    # ── Curriculum-gated training ────────────────────────────
    print(
        f"\nStarting curriculum GRPO | epochs={args.epochs} | "
        f"batch={args.batch_size} | G={GRPO_GROUP_SIZE} | "
        f"single_gpu=GPU{args.gpu_id} | qa_max_tokens={QA_MAX_TOKENS} | "
        f"vllm_mem_util={args.vllm_gpu_memory_utilization}"
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
                )

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

            training_log.append(
                {
                    "stage": stage_id,
                    "retrain_attempt": retrain_ct,
                    **metrics,
                    "passed": passed,
                }
            )

            stage_ckpt = os.path.join(
                args.output_dir,
                f"stage{stage_id}_attempt{retrain_ct}_{'pass' if passed else 'fail'}",
            )
            critic_model.save_pretrained(stage_ckpt)
            critic_tok.save_pretrained(stage_ckpt)
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

    # ── Training summary ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE — CURRICULUM SUMMARY")
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
