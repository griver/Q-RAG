import os
import argparse
import json
import re
from typing import List
from tqdm import tqdm
from vllm import LLM, SamplingParams
from contextlib import redirect_stderr
from prompts_and_metrics.general_qa import compute_exact_match, compute_f1
from envs.qa_dataset_adapter import CHUNK_SIZE

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation. Please provide a direct conversational answer without any function definitions, JSON, or code blocks. \n\nQuestion: {input}",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, give the short and precise answer only and do not output any other words.\n\nQuestion: {input}",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
}

dataset = "narrativeqa"
#dataset = "hotpotqa"



def save_final_scores(results_path: str, ns_key: str, em: float, f1: float) -> None:
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}
    results[ns_key] = {"em": em, "f1": f1}
    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write('\n' + str(CHUNK_SIZE))



def prepare_messages(question: str, facts: List[str], prompt_format: str):
    facts_str = " ".join(facts)
    input_text = prompt_format.format(context=facts_str, input=question)
    messages = [
        {"role": "system", "content": "You are a expert AI assistant. Your task is to answer the user's question based exclusively on the provided text context."},
        {"role": "user",   "content": input_text},
    ]
    return messages



def delete_reasoning(s, substring="</think>"):
    idx = s.find(substring)
    if idx == -1:
        return s
    return s[idx+len(substring):]



def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on retriever logs")
    parser.add_argument("retriever_logfile", help="Path to retriever log file")
    parser.add_argument("--llm_name", help="Name of LLM to load")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument('--think', action="store_true", default=False, help='enable_thinking for Qwen3 models.')
    args = parser.parse_args()

    chat_template_kwargs = dict(
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    if "Qwen3" in args.llm_name:
        chat_template_kwargs['enable_thinking'] = args.think
        print(f"Set enable_thinking = {args.think} for {args.llm_name}")

    vllm_config = {
        'gpu_memory_utilization': 0.95,
        'max_model_len': 10000,
        'dtype': 'bfloat16',
        'quantization': None,
        'tensor_parallel_size': 1,
        'trust_remote_code': True,
    }

    sampling_params = {
        'max_tokens': args.max_tokens,
        'temperature': 0.01,
        "stop": None,
        'top_k': 3,
        'top_p': 0.7
    }

    os.environ["VLLM_NO_PROGRESS_BARS"] = "1"
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    devnull = open(os.devnull, 'w')

    llm = LLM(model=args.llm_name, **vllm_config)
    sampling_params = SamplingParams(**sampling_params)

    out_path = os.path.join(
        os.path.dirname(args.retriever_logfile),
        f"{os.path.basename(args.retriever_logfile)}_{os.path.basename(args.llm_name)}",
    )

    with open(args.retriever_logfile, "r") as f_in:
        lines = [ln for ln in f_in if ln.strip()]

    all_f1 = []
    all_em = []

    with open(out_path, "w") as f_out:
        for line in tqdm(lines, desc="LLM eval", ncols=80):
            item = json.loads(line)
            question = item["question"]
            answer = item["answer"]
            facts = item.get("pred_text", [])

            prompt_format = dataset2prompt[dataset]
            messages = prepare_messages(question, facts, prompt_format)
            prompt = llm.get_tokenizer().apply_chat_template(messages, **chat_template_kwargs)

            with redirect_stderr(devnull):
                outputs = llm.generate([prompt], sampling_params)
            prediction = outputs[0].outputs[0].text.strip()
            if "QwQ" in args.llm_name:
                prediction = delete_reasoning(prediction).strip()

            ans_em = compute_exact_match(prediction, answer)
            ans_f1 = compute_f1(prediction, answer)
            all_f1.append(ans_f1)
            all_em.append(ans_em)

            item.update({
                "prediction": prediction,
                'pred_text': facts,
                "answer_em": ans_em,
                "answer_f1": ans_f1,
            })
            del item["sf_idx"];  del item["pred_idx"];  del item["sf_texts"]
            del item["em"];  del item["f1"];

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()

    final_em = sum(all_em) / len(all_em) * 100
    final_f1 = sum(all_f1) / len(all_f1) * 100

    print(f"Saved results to {out_path}.")
    print(f"EM: {final_em}, F1: {final_f1}")

    ns_match = re.search(r"ns(\d+)", os.path.basename(args.retriever_logfile))
    ns_key = ns_match.group(1) if ns_match else "unknown"

    results_path = os.path.join(os.path.dirname(args.retriever_logfile),
        f"results_{os.path.basename(args.retriever_logfile)}_{os.path.basename(args.llm_name)}.txt")
    save_final_scores(results_path, ns_key, final_em, final_f1)



if __name__ == "__main__":
    main()
