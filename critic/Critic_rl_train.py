"""
Critic Agent RL Training via GRPO
===================================
Train a Plan-Critic from scratch using only outcome reward (EM/F1).
The Critic learns to Accept/Reject Planner output; its reward is derived
entirely from whether the final multi-hop answer is correct.

Architecture:
    Question
      → Planner (LoRA, frozen during critic training)
      → [Plan Critic]  ← trained here with GRPO
           ↓ Accept          ↓ Reject (max MAX_RETRIES)
      Agent Loop          Planner (re-plan with feedback)
           ↓
      Final Answer  →  Reward (EM / F1)  →  GRPO update on Critic
"""

import json
import os
import re
import string
import argparse
import copy
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

# ─────────────────────────────────────────────────────────────
#  Constants — Planner (keep identical to eval script)
# ─────────────────────────────────────────────────────────────
PLANNER_SYSTEM_PROMPT = (
    "You are a multi-hop question planner. "
    "Given a complex question that requires multiple reasoning steps, "
    "decompose it into a sequence of simple, self-contained sub-questions. "
    "Each sub-question should be answerable independently or by referring to "
    "the answer of a previous step (use '#1', '#2', ... as placeholders). "
    "Output each sub-question on a new line, prefixed with 'Step N:'."
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
#  The Critic ONLY sees: original question + proposed plan.
#  It must output a structured verdict so we can parse it reliably.
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
GRPO_GROUP_SIZE   = 4       # G: number of critic samples per question
MAX_RETRIES       = 2       # max Planner re-plans on Reject
LR                = 1e-5
WARMUP_STEPS      = 50
GRAD_CLIP         = 1.0
KL_COEF           = 0.05    # KL penalty coefficient (keep Critic close to ref)
REWARD_ACCEPT_CORRECT   =  1.0
REWARD_ACCEPT_WRONG     = -1.0
REWARD_REJECT_CORRECT   =  1.0
REWARD_REJECT_STILL_BAD = -0.5

# ─────────────────────────────────────────────────────────────
#  Curriculum stage thresholds
#  每个 stage 训练完后，在该 stage 的数据上评估
#  EM 和 F1 都必须达标，才能进入下一阶段
#  如果未达标，当前 stage 最多重复训练 MAX_STAGE_RETRAIN 次
# ─────────────────────────────────────────────────────────────
STAGE_THRESHOLDS = {
    1: {"EM": 0.45, "F1": 0.55},
    2: {"EM": 0.38, "F1": 0.48},
    3: {"EM": 0.30, "F1": 0.40},
}
MAX_STAGE_RETRAIN = 2   # 同一 stage 最多重训几次再强制进入下一阶段


# ─────────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file",        type=str, required=True,
                   help="JSONL with 'question', 'answer', 'pred_texts' fields")
    p.add_argument("--critic_base",       type=str, default="Qwen/Qwen2.5-7B-Instruct",
                   help="Base model for critic (trained from scratch with LoRA)")
    p.add_argument("--planner_base",      type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--planner_lora",      type=str, default="./qwen_planner_lora_v2/final")
    p.add_argument("--qa_model",          type=str, default="Qwen/QwQ-32B",
                   help="QA model for agent loop (loaded via vLLM)")
    p.add_argument("--output_dir",        type=str, default="./critic_lora_grpo")
    p.add_argument("--epochs",            type=int, default=3)
    p.add_argument("--batch_size",        type=int, default=4,
                   help="Questions per gradient update")
    p.add_argument("--critic_temp",       type=float, default=0.8,
                   help="Sampling temperature for GRPO group sampling")
    p.add_argument("--max_new_tokens",    type=int, default=128)
    p.add_argument("--save_every",        type=int, default=200)
    p.add_argument("--log_every",         type=int, default=20)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
#  Metric helpers (identical to eval script)
# ─────────────────────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t):
        exc = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exc)
    return white_space_fix(remove_articles(remove_punc(s.lower().strip())))

def compute_f1(pred, gold):
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    common = Counter_common(p_toks, g_toks)
    if not common: return 0.0
    prec = common / len(p_toks) if p_toks else 0
    rec  = common / len(g_toks) if g_toks else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def Counter_common(p, g):
    from collections import Counter
    pc, gc = Counter(p), Counter(g)
    return sum((pc & gc).values())

def compute_em(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))

def score_answer(pred: str, gold: str) -> float:
    """Combined EM + F1 reward in [0, 1]."""
    em = compute_em(pred, gold)
    f1 = compute_f1(pred, gold)
    return max(em, f1)   # take best of EM / F1


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
    """
    Returns (verdict: str, feedback: str | None)
    verdict is 'ACCEPT' or 'REJECT'
    """
    verdict_match = re.search(r"<verdict>\s*(ACCEPT|REJECT)\s*</verdict>", text, re.IGNORECASE)
    verdict  = verdict_match.group(1).upper() if verdict_match else "ACCEPT"  # default accept if malformed

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
        user_content += f"\n\nPrevious plan was rejected. Feedback: {feedback}\nPlease revise the plan."

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    prompt = planner_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = planner_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(planner_model.device)
    out = planner_model.generate(
        **enc,
        max_new_tokens=256,
        do_sample=False,
        temperature=1.0,
        pad_token_id=planner_tok.pad_token_id,
    )
    text = planner_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_steps(text)


# ─────────────────────────────────────────────────────────────
#  Agent Loop (QA execution using vLLM)
# ─────────────────────────────────────────────────────────────
def run_agent_loop(question: str, sub_questions: list, context: str, llm, qa_tok) -> str:
    """Run multi-hop QA and return the final synthesised answer."""
    sampling = SamplingParams(max_tokens=256, temperature=0.0)
    hop_answers = []

    # Intermediate hops
    for sub_q_raw in sub_questions:
        sub_q = replace_placeholders(sub_q_raw, hop_answers)
        prompt_text = HOP_PROMPT.format(context=context, question=sub_q)
        messages = [
            {"role": "system", "content": HOP_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text},
        ]
        text = qa_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([text], sampling)
        ans = extract_final_answer(outputs[0].outputs[0].text)
        hop_answers.append(ans)

    # Synthesis
    reasoning = build_reasoning_chain(sub_questions, hop_answers)
    synth_text = SYNTHESIS_PROMPT.format(context=context, reasoning=reasoning, question=question)
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user",   "content": synth_text},
    ]
    text = qa_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate([text], sampling)
    return extract_final_answer(outputs[0].outputs[0].text)


# ─────────────────────────────────────────────────────────────
#  Critic reward assignment
# ─────────────────────────────────────────────────────────────
def compute_critic_reward(verdict: str, answer_score: float, threshold: float = 0.5) -> float:
    """
    Compute scalar reward for the Critic's verdict.

    Key design: Critic is rewarded for interventions that *causally* improve outcome.
    We use a soft threshold (F1 >= 0.5 = "correct") for robustness.
    """
    correct = answer_score >= threshold
    if verdict == "ACCEPT":
        return REWARD_ACCEPT_CORRECT if correct else REWARD_ACCEPT_WRONG
    else:  # REJECT
        return REWARD_REJECT_CORRECT if correct else REWARD_REJECT_STILL_BAD


# ─────────────────────────────────────────────────────────────
#  Critic tokenisation helpers
# ─────────────────────────────────────────────────────────────
def build_critic_prompt(question: str, plan: list, critic_tok) -> str:
    plan_text = format_plan(plan)
    user_content = CRITIC_USER_PROMPT.format(question=question, plan=plan_text)
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return critic_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def sample_critic_responses(
    prompt: str,
    critic_model,
    critic_tok,
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
) -> list[dict]:
    """
    Sample `num_samples` responses from the Critic for one prompt.
    Returns list of dicts with keys: text, input_ids, response_ids, log_probs
    """
    enc = critic_tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(critic_model.device)
    input_len = enc["input_ids"].shape[1]

    results = []
    for _ in range(num_samples):
        with torch.no_grad():
            out = critic_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=critic_tok.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        response_ids = out.sequences[0][input_len:]
        text = critic_tok.decode(response_ids, skip_special_tokens=True).strip()

        # Compute per-token log probs for GRPO loss
        # scores: list of (vocab_size,) tensors, one per generated token
        log_probs = []
        for step_idx, score_t in enumerate(out.scores):
            if step_idx >= len(response_ids): break
            token_id = response_ids[step_idx]
            lp = F.log_softmax(score_t[0], dim=-1)[token_id]
            log_probs.append(lp)

        results.append({
            "text":         text,
            "input_ids":    enc["input_ids"].clone(),
            "response_ids": response_ids.clone(),
            "log_probs":    log_probs,  # list of scalar tensors
        })
    return results


# ─────────────────────────────────────────────────────────────
#  GRPO loss
# ─────────────────────────────────────────────────────────────
def grpo_loss(
    samples: list[dict],
    rewards: list[float],
    ref_model,
    critic_tok,
    kl_coef: float = KL_COEF,
) -> torch.Tensor:
    """
    GRPO (Group Relative Policy Optimisation) loss for one group.

    Advantage = (r - mean(r)) / (std(r) + eps)   — normalised within group
    Loss = -mean_over_group( advantage * sum_log_probs ) + kl_coef * KL(policy || ref)
    """
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean_r    = rewards_t.mean()
    std_r     = rewards_t.std() + 1e-8
    advantages = (rewards_t - mean_r) / std_r   # shape (G,)

    total_loss = torch.tensor(0.0, requires_grad=True)
    total_loss = total_loss.to(next(ref_model.parameters()).device)

    for sample, adv in zip(samples, advantages):
        if not sample["log_probs"]:
            continue
        sum_log_prob = torch.stack(sample["log_probs"]).sum()   # policy log prob

        # KL penalty: compare to reference model
        with torch.no_grad():
            input_ids = sample["input_ids"]
            resp_ids  = sample["response_ids"].unsqueeze(0)
            full_ids  = torch.cat([input_ids, resp_ids], dim=1)
            ref_out   = ref_model(full_ids)
            ref_logits = ref_out.logits[0, input_ids.shape[1]-1:-1]  # align to response
            ref_lp    = F.log_softmax(ref_logits, dim=-1)
            ref_sum_lp = ref_lp[range(len(sample["response_ids"])), sample["response_ids"]].sum()

        kl = sum_log_prob - ref_sum_lp   # positive if policy diverges from ref

        # GRPO policy gradient loss
        pg_loss   = -adv.to(sum_log_prob.device) * sum_log_prob
        sample_loss = pg_loss + kl_coef * kl

        total_loss = total_loss + sample_loss

    return total_loss / max(len(samples), 1)


# ─────────────────────────────────────────────────────────────
#  Single-stage training function
# ─────────────────────────────────────────────────────────────
def train_one_stage(
    stage_data: list,
    stage_id: int,
    critic_model,
    critic_tok,
    ref_model,
    planner_model,
    planner_tok,
    llm,
    qa_tok,
    optimizer,
    scheduler,
    args,
    global_step: int,
) -> tuple[int, dict]:
    """
    训练一个 stage，返回 (更新后的 global_step, 该 stage 的最终指标)。
    指标格式: {"EM": float, "F1": float, "avg_reward": float, "accept_rate": float}
    """
    import random
    random.shuffle(stage_data)

    log_stats      = defaultdict(list)
    batch_loss_acc = torch.tensor(0.0)
    optimizer.zero_grad()

    for sample_idx, data in enumerate(tqdm(stage_data, desc=f"  Stage {stage_id} training")):
        question     = data["question"]
        ground_truth = data["answer"]
        context      = "\n\n---\n\n".join(data["pred_texts"])

        # Step 1: Planner → initial plan
        initial_plan = planner_decompose(question, planner_model, planner_tok)
        if not initial_plan:
            continue

        # Step 2: Build critic prompt
        critic_prompt = build_critic_prompt(question, initial_plan, critic_tok)

        # Step 3: Sample G critic responses (GRPO group)
        critic_model.train()
        group_samples = sample_critic_responses(
            critic_prompt, critic_model, critic_tok,
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
                revised = planner_decompose(
                    question, planner_model, planner_tok, feedback=feedback
                )
                if revised:
                    final_plan = revised
                    re_prompt  = build_critic_prompt(question, final_plan, critic_tok)
                    re_samples = sample_critic_responses(
                        re_prompt, critic_model, critic_tok,
                        num_samples=1, temperature=0.0,
                        max_new_tokens=args.max_new_tokens,
                    )
                    verdict, feedback = parse_critic_output(re_samples[0]["text"])
                    if verdict == "ACCEPT":
                        break

            pred_answer = run_agent_loop(question, final_plan, context, llm, qa_tok)
            ans_score   = score_answer(pred_answer, ground_truth)
            reward      = compute_critic_reward(verdict, ans_score)
            group_rewards.append(reward)

            log_stats["em"].append(compute_em(pred_answer, ground_truth))
            log_stats["f1"].append(compute_f1(pred_answer, ground_truth))
            log_stats["reward"].append(reward)
            log_stats["verdict"].append(1 if verdict == "ACCEPT" else 0)

        # Step 5: GRPO loss
        loss = grpo_loss(group_samples, group_rewards, ref_model, critic_tok)
        loss = loss / args.batch_size
        loss.backward()
        batch_loss_acc += loss.item()

        # Step 6: Gradient update
        if (sample_idx + 1) % args.batch_size == 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, critic_model.parameters()),
                GRAD_CLIP
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.log_every == 0:
                window    = 50
                avg_em    = sum(log_stats["em"][-window:])     / max(len(log_stats["em"][-window:]), 1)
                avg_f1    = sum(log_stats["f1"][-window:])     / max(len(log_stats["f1"][-window:]), 1)
                avg_rew   = sum(log_stats["reward"][-window:]) / max(len(log_stats["reward"][-window:]), 1)
                acc_rate  = sum(log_stats["verdict"][-window:])/ max(len(log_stats["verdict"][-window:]), 1)
                print(
                    f"    Step {global_step:5d} | loss={batch_loss_acc.item():.4f} | "
                    f"EM={avg_em:.3f}  F1={avg_f1:.3f} | "
                    f"reward={avg_rew:.3f} | accept_rate={acc_rate:.2f}"
                )
                batch_loss_acc = torch.tensor(0.0)

            if global_step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"stage{stage_id}_step{global_step}")
                critic_model.save_pretrained(ckpt)
                critic_tok.save_pretrained(ckpt)
                print(f"    Checkpoint saved → {ckpt}")

    # ── Compute final stage metrics (over ALL samples seen this stage) ──
    final_metrics = {
        "EM":          sum(log_stats["em"])      / max(len(log_stats["em"]), 1),
        "F1":          sum(log_stats["f1"])      / max(len(log_stats["f1"]), 1),
        "avg_reward":  sum(log_stats["reward"])  / max(len(log_stats["reward"]), 1),
        "accept_rate": sum(log_stats["verdict"]) / max(len(log_stats["verdict"]), 1),
    }
    return global_step, final_metrics


# ─────────────────────────────────────────────────────────────
#  Main training loop  (curriculum-gated)
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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

    # ── Load Planner (frozen) ─────────────────────────────────
    print("\n[1/4] Loading Planner (frozen) ...")
    planner_tok = AutoTokenizer.from_pretrained(args.planner_base, trust_remote_code=True)
    planner_tok.pad_token = planner_tok.eos_token
    planner_tok.padding_side = "left"
    planner_base_model = AutoModelForCausalLM.from_pretrained(
        args.planner_base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    planner_model = PeftModel.from_pretrained(planner_base_model, args.planner_lora)
    planner_model.eval()
    for p in planner_model.parameters():
        p.requires_grad_(False)
    print("Planner ready (frozen).")

    # ── Load QA model ─────────────────────────────────────────
    print("\n[2/4] Loading QA model via vLLM ...")
    qa_tok = AutoTokenizer.from_pretrained(args.qa_model)
    llm = LLM(
        model=args.qa_model, trust_remote_code=True,
        gpu_memory_utilization=0.5, max_model_len=16000,
        tensor_parallel_size=2,
    )
    print("QA model ready.")

    # ── Load Critic (trainable) + reference ──────────────────
    print("\n[3/4] Loading Critic model ...")
    critic_tok = AutoTokenizer.from_pretrained(args.critic_base, trust_remote_code=True)
    critic_tok.pad_token = critic_tok.eos_token

    critic_base_model = AutoModelForCausalLM.from_pretrained(
        args.critic_base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    critic_model = get_peft_model(critic_base_model, lora_cfg)
    critic_model.print_trainable_parameters()

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.critic_base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    print("Critic ready.")

    # ── Optimiser & scheduler ─────────────────────────────────
    total_samples = sum(len(v) for v in stage_data.values())
    total_steps   = (total_samples * args.epochs) // args.batch_size
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, critic_model.parameters()),
        lr=LR, weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    # ── Curriculum-gated training ─────────────────────────────
    print(f"\n[4/4] Starting curriculum GRPO | epochs={args.epochs} | "
          f"batch={args.batch_size} | G={GRPO_GROUP_SIZE}")

    global_step   = 0
    training_log  = []   # records per-stage metrics for the final report

    for stage_id in sorted(stage_data.keys()):
        threshold  = STAGE_THRESHOLDS.get(stage_id, {"EM": 0.0, "F1": 0.0})
        retrain_ct = 0
        passed     = False

        print(f"\n{'='*55}")
        print(f"  STAGE {stage_id}  |  target EM≥{threshold['EM']}  F1≥{threshold['F1']}")
        print(f"  samples: {len(stage_data[stage_id])}")
        print(f"{'='*55}")

        while not passed and retrain_ct <= MAX_STAGE_RETRAIN:
            if retrain_ct > 0:
                print(f"\n  [Stage {stage_id}] Threshold not met — retraining "
                      f"(attempt {retrain_ct}/{MAX_STAGE_RETRAIN}) ...")

            for epoch in range(args.epochs):
                print(f"\n  Epoch {epoch+1}/{args.epochs}")
                global_step, metrics = train_one_stage(
                    stage_data   = stage_data[stage_id],
                    stage_id     = stage_id,
                    critic_model = critic_model,
                    critic_tok   = critic_tok,
                    ref_model    = ref_model,
                    planner_model= planner_model,
                    planner_tok  = planner_tok,
                    llm          = llm,
                    qa_tok       = qa_tok,
                    optimizer    = optimizer,
                    scheduler    = scheduler,
                    args         = args,
                    global_step  = global_step,
                )

            # ── Stage evaluation ──────────────────────────────
            em_ok = metrics["EM"] >= threshold["EM"]
            f1_ok = metrics["F1"] >= threshold["F1"]
            passed = em_ok and f1_ok

            status = "PASSED ✓" if passed else "FAILED ✗"
            print(f"\n  Stage {stage_id} result: "
                  f"EM={metrics['EM']:.4f} (≥{threshold['EM']}) {'✓' if em_ok else '✗'}  |  "
                  f"F1={metrics['F1']:.4f} (≥{threshold['F1']}) {'✓' if f1_ok else '✗'}  "
                  f"→ {status}")
            print(f"  avg_reward={metrics['avg_reward']:.3f}  "
                  f"accept_rate={metrics['accept_rate']:.2f}")

            training_log.append({
                "stage":       stage_id,
                "retrain_attempt": retrain_ct,
                **metrics,
                "passed": passed,
            })

            # Save stage checkpoint regardless of pass/fail
            stage_ckpt = os.path.join(
                args.output_dir,
                f"stage{stage_id}_attempt{retrain_ct}_{'pass' if passed else 'fail'}"
            )
            critic_model.save_pretrained(stage_ckpt)
            critic_tok.save_pretrained(stage_ckpt)
            print(f"  Checkpoint saved → {stage_ckpt}")

            retrain_ct += 1

        if not passed:
            print(f"\n  WARNING: Stage {stage_id} did not meet threshold after "
                  f"{MAX_STAGE_RETRAIN} retrains. Proceeding to next stage anyway.")

    # ── Final save ────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    critic_model.save_pretrained(final_dir)
    critic_tok.save_pretrained(final_dir)

    # ── Training summary ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE — CURRICULUM SUMMARY")
    print("=" * 55)
    for entry in training_log:
        tag = "PASS" if entry["passed"] else "FAIL"
        print(f"  Stage {entry['stage']} attempt {entry['retrain_attempt']} | "
              f"EM={entry['EM']:.4f}  F1={entry['F1']:.4f}  "
              f"reward={entry['avg_reward']:.3f}  [{tag}]")

    # Save log to JSON
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\n  Log saved → {log_path}")
    print(f"  Final model → {final_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()