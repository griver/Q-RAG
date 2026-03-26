import json
import os
import argparse
import numpy as np
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
import torch.utils
from nltk.probability import gt_demo
from torch.utils.data import Dataset
import json
import re
import string
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
import sys


parser = argparse.ArgumentParser(description="LLM answering with vLLM")
parser.add_argument("--file_path", type=str, required=True,
                    help="Path to the input JSONL file with extracted chunks")
parser.add_argument("--model_name", type=str, required=True,
                    help="Path to the model (e.g. /mnt/Qwen3-8B)")
parser.add_argument("--output_file_path", type=str, default=None,
                    help="Path to save the output JSON (default: <input_dir>/<input_stem>_eval_llm.json)")
args = parser.parse_args()

file_path = args.file_path
model_name = args.model_name
if args.output_file_path:
    output_file_path = args.output_file_path
else:
    base, _ = os.path.splitext(file_path)
    output_file_path = base + "_eval_llm.json"

print(f"Input: {file_path}")
print(f"Output: {output_file_path}")
print(f"Model: {model_name}")



dataset = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

print(f"Samples in dataset: {len(dataset)}")

#os.environ["VLLM_USE_TORCH_COMPILE"] = "0"
#os.environ["TORCH_COMPILE_DISABLE"] = "1"
#os.environ["VLLM_DISABLE_CUDA_GRAPHS"] = "1"


'''
qa_instruction_prompt = """You are a question-answer long-context system.
Carefully read all context, pay attention on crucial facts and accurately answer the given question.
Your answer must be a short and direct answer to the QUESTION.
If you need Chain of Thoughts, you can write it, but your answer must be finished with the following template:

Final answer: your final SHORT AND DIRECT answer."""

qa_prompt = """QUESTION:
{question}

CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER: """
'''


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



# def build_messages(prompt):
#     messages = [
#         {"role": "system", "content": "You are Qwen, a helpful assistant. You need to answer the question briefly."},
#         {"role": "user", "content":f"{prompt}"}
#         ]
#     return messages


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))


def compute_exact_match(prediction, target):
    target, prediction = normalize_answer(target), normalize_answer(prediction)
    return int(target == prediction)


def recall(prediction, target):
    target, prediction = normalize_answer(target).split(), normalize_answer(prediction).split()
    len_true = len(target)
    len_good = 0
    for word in prediction:
        if word in target:
            len_good += 1
            target.remove(word)
    return len_good / len_true if len_true > 0 else 1


def precision(prediction, target):
    target, prediction = normalize_answer(target).split(), normalize_answer(prediction).split()
    len_gen = len(prediction)
    len_good = 0
    for word in target:
        if word in prediction:
            len_good += 1
            prediction.remove(word)
    return len_good / len_gen if len_gen > 0 else 1


def compute_f1(prediction, target):
    prec = precision(prediction, target)
    rec = recall(prediction, target)
    if (prec + rec) == 0.:
        return 0.

    f1 = (2. * prec * rec) / (prec + rec)
    return f1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---  Loading the model via vLLM ---

# The tokenizer is still needed to correctly apply the chat template
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Load the model using the LLM class from vLLM
# If you have multiple GPUs, you can add: tensor_parallel_size=N
llm = LLM(model=model_name,
          trust_remote_code=True,
          gpu_memory_utilization=0.95,
          max_model_len=32000,)
print(f"Model {model_name} loaded successfully with vLLM.")




# Lists for storing results
results = []
all_em_scores = []
all_f1_scores = []

all_prompts = []
for data in tqdm(dataset, desc="Preparing prompts"):
    question = data['question']
    try:
        context = "\n\n---\n\n".join(data['pred_texts'])
    except (KeyError, TypeError):
        context = "\n\n---\n\n".join(data['pred_text'])
    full_prompt_for_model = qa_prompt.format(context=context, question=question)

    messages = [
        {"role": "system", "content": qa_instruction_prompt},
        {"role": "user", "content": full_prompt_for_model}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    all_prompts.append(text)

print(f"Prepared {len(all_prompts)} prompts for batch processing.")


# 2. Run generation for all prompts in a SINGLE call
# Set generation parameters
sampling_params = SamplingParams(
    max_tokens=4000,
    temperature=0.0, # temperature=0.0 for greedy generation
)

print("Starting batch answer generation...")
outputs = llm.generate(all_prompts, sampling_params)
print("Generation completed.")


# 3. Results processing
results = []
all_em_scores = []
all_f1_scores = []


for i, (data, output) in enumerate(tqdm(zip(dataset, outputs), total=len(dataset), desc="Processing results")):
    question = data['question']
    ground_truth_answer = data['answer']

    # the answer text in output.outputs[0].text
    decoded_output = output.outputs[0].text

    if "Final answer:" in decoded_output:
        llm_prediction = decoded_output.split("Final answer:")[-1].strip()
    else:
        llm_prediction = decoded_output.strip()

    em_score = compute_exact_match(llm_prediction, ground_truth_answer)
    f1_score = compute_f1(llm_prediction, ground_truth_answer)

    all_em_scores.append(em_score)
    all_f1_scores.append(f1_score)

    result_entry = {
        "question": question,
        "retrieved_chunks_idx": data['pred_idx'],
        'ground_truth_chunks_idx': data["sf_idx"],
        "ground_truth": ground_truth_answer,
        "prediction": llm_prediction,
        "full_model_output": decoded_output,
        "EM": em_score,
        "F1": f1_score
    }
    results.append(result_entry)

    if (i + 1) % 100 == 0:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            json.dump(results, f_out, indent=4, ensure_ascii=False)
        print(f"--- {i+1}/{len(dataset)}: Intermediate results saved. ---")

    # if (i + 1) % 10 == 0:
    #     avg_em = sum(all_em_scores) / len(all_em_scores)
    #     avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
    #     print("=" * 50)
    #     print(f"Samples processed: {len(all_em_scores)}")
    #     print(f"Average Exact Match (EM): {avg_em:.4f}")
    #     print(f"Average F1-Score: {avg_f1:.4f}")
    #     print("=" * 50)


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

# Final Save
with open(output_file_path, 'w', encoding='utf-8') as f_out:
    json.dump(results, f_out, indent=4, ensure_ascii=False)
print(f"All results saved to {output_file_path}")


