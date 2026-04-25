import json
import os
import argparse
import numpy as np
import re
import string
import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser(description="LLM answering with vLLM + Planner Agent")
parser.add_argument("--file_path", type=str, required=True,
                    help="Path to the input JSONL file with extracted chunks")
parser.add_argument("--model_name", type=str, required=True,
                    help="Path to the Reader model (e.g. /mnt/Qwen3-8B)")
# [新增] 传入你合并好的 Planner 模型路径
parser.add_argument("--planner_model_name", type=str, required=True,
                    help="Path to the fine-tuned Planner model")
parser.add_argument("--output_file_path", type=str, default=None,
                    help="Path to save the output JSON")
args = parser.parse_args()

file_path = args.file_path
model_name = args.model_name
planner_model_name = args.planner_model_name

if args.output_file_path:
    output_file_path = args.output_file_path
else:
    base, _ = os.path.splitext(file_path)
    output_file_path = base + "_agent_eval.json"

print(f"Input: {file_path}")
print(f"Output: {output_file_path}")
print(f"Reader Model: {model_name}")
print(f"Planner Model: {planner_model_name}")

dataset = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))
print(f"Samples in dataset: {len(dataset)}")

# --- Prompts 配置 ---
# [新增] Planner 的 System Prompt (必须和你训练时一字不差)
PLANNER_SYSTEM_PROMPT = """You are an expert Query Planner for an advanced Agentic Machine Reading Comprehension (MRC) system.
The system has already retrieved a pool of context documents. Your task is to decompose a complex, multi-hop user query into a sequence of atomic sub-queries. A downstream Reader Agent will execute these sub-queries sequentially against the retrieved documents to find the final answer.

### INSTRUCTIONS & CONSTRAINTS:
1. You MUST output ONLY a valid JSON object. No markdown formatting outside the JSON.
2. The JSON must contain two keys: "thought_process" and "plan".
3. "thought_process" is a string analyzing the logical hops.
4. "plan" is an array of objects. Each object MUST contain:
   - "step_id": an integer starting from 1.
   - "sub_query": a natural language question representing the atomic step. Use placeholders like "the entity identified in #1" to refer to previous steps.
   - "dependency": an array of integers representing the step_ids this step depends on. If independent, use []."""

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

# --- 工具函数保持不变 ---
def normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))

def compute_exact_match(prediction, target):
    return int(normalize_answer(target) == normalize_answer(prediction))

def recall(prediction, target):
    target, prediction = normalize_answer(target).split(), normalize_answer(prediction).split()
    len_true = len(target)
    len_good = sum(1 for word in prediction if word in target and not target.remove(word))
    return len_good / len_true if len_true > 0 else 1

def precision(prediction, target):
    target, prediction = normalize_answer(target).split(), normalize_answer(prediction).split()
    len_gen = len(prediction)
    len_good = sum(1 for word in target if word in prediction and not prediction.remove(word))
    return len_good / len_gen if len_gen > 0 else 1

def compute_f1(prediction, target):
    prec, rec = precision(prediction, target), recall(prediction, target)
    return (2. * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# [新增] 显存切分策略加载两个大模型
# ==========================================
# 由于要在同一张/两张卡上加载两个模型，必须严格限制显存比例，否则会 OOM
print("🧠 正在加载 Planner 模型...")
planner_tokenizer = AutoTokenizer.from_pretrained(planner_model_name)
planner_llm = LLM(
    model=planner_model_name, 
    trust_remote_code=True, 
    gpu_memory_utilization=0.25, # 给 Planner 分配 25% 显存
    max_model_len=4096
)

print("📖 正在加载 Reader 模型...")
reader_tokenizer = AutoTokenizer.from_pretrained(model_name)
reader_llm = LLM(
    model=model_name, 
    trust_remote_code=True, 
    gpu_memory_utilization=0.65, # 给 Reader 分配 65% 显存 (需吞吐长文本)
    max_model_len=32000
)

# ==========================================
# 阶段 1: 批量生成所有 Plan
# ==========================================
print("\n🚀 [阶段 1] Planner 开始批量拆解问题...")
planner_prompts = []
for data in dataset:
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": data['question']}
    ]
    planner_prompts.append(planner_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

plan_outputs = planner_llm.generate(planner_prompts, SamplingParams(temperature=0.0, max_tokens=1024))

# 解析 JSON Plans
all_plans = []
for out in plan_outputs:
    try:
        plan_json = json.loads(out.outputs[0].text)
        all_plans.append(plan_json.get("plan", []))
    except Exception as e:
        # 如果解析失败，回退到把原问题当成唯一的 Step
        print(f"JSON 解析警告，回退为单步推理...")
        all_plans.append([]) 

# ==========================================
# 阶段 2: Agent Loop 执行
# ==========================================
print("\n🚀 [阶段 2] Reader 开始执行 Agent Loop...")
results = []
all_em_scores = []
all_f1_scores = []

# 这里只能循环执行，因为每一步依赖上一步的答案
for i, (data, steps) in enumerate(tqdm(zip(dataset, all_plans), total=len(dataset), desc="Agent Loop")):
    question = data['question']
    ground_truth_answer = data['answer']
    try:
        context = "\n\n---\n\n".join(data['pred_texts'])
    except:
        context = "\n\n---\n\n".join(data.get('pred_text', []))

    # 容错：如果 plan 是空的，直接把原问题当作 step 1
    if not steps:
        steps = [{"step_id": 1, "sub_query": question, "dependency": []}]

    step_answers = {}
    final_llm_prediction = ""
    full_model_output = "" # 记录最后一步的完整输出

    # === 核心：循环执行 Plan ===
    for step in steps:
        curr_sub_query = step["sub_query"]
        
        # 占位符替换逻辑：将 "#1" 替换为 step_id 为 1 的答案
        for prev_id, prev_ans in step_answers.items():
            curr_sub_query = curr_sub_query.replace(f"#{prev_id}", prev_ans)

        # 构造 Reader 的 Prompt
        reader_messages = [
            {"role": "system", "content": qa_instruction_prompt},
            {"role": "user", "content": qa_prompt.format(context=context, question=curr_sub_query)}
        ]
        reader_text = reader_tokenizer.apply_chat_template(reader_messages, tokenize=False, add_generation_prompt=True)
        
        # 跑单条推理
        step_output = reader_llm.generate([reader_text], SamplingParams(temperature=0.0, max_tokens=1024), use_tqdm=False)
        raw_ans = step_output[0].outputs[0].text
        
        # 清洗 Final answer
        clean_ans = raw_ans.split("Final answer:")[-1].strip() if "" in raw_ans else raw_ans.strip()
        
        # 记录到记忆字典中
        step_answers[step["step_id"]] = clean_ans
        final_llm_prediction = clean_ans
        full_model_output = raw_ans

    # === 计算得分并记录 ===
    em_score = compute_exact_match(final_llm_prediction, ground_truth_answer)
    f1_score = compute_f1(final_llm_prediction, ground_truth_answer)

    all_em_scores.append(em_score)
    all_f1_scores.append(f1_score)

    result_entry = {
        "question": question,
        "generated_plan": steps, # 存下来方便后面复盘分析
        "intermediate_steps": step_answers, # 存下每一步查到了什么
        "ground_truth": ground_truth_answer,
        "prediction": final_llm_prediction,
        "full_model_output": full_model_output,
        "EM": em_score,
        "F1": f1_score
    }
    results.append(result_entry)

    if (i + 1) % 100 == 0:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(results, f_out, indent=4, ensure_ascii=False)

# --- Final metric calculation ---
avg_em = sum(all_em_scores) / len(all_em_scores) if all_em_scores else 0
avg_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0

print("\n" + "=" * 50)
print("             EVAL RESULTS")
print("=" * 50)
print(f"Num samples: {len(results)}")
print(f"Mean Exact Match (EM): {avg_em:.4f}")
print(f"Mean F1-Score: {avg_f1:.4f}")
print("=" * 50)

with open(output_file_path, 'w', encoding='utf-8') as f_out:
    json.dump(results, f_out, indent=4, ensure_ascii=False)
print(f"All results saved to {output_file_path}")