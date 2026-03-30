import json

def analyze_retrieval(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)
    retrieval_miss = 0
    retrieval_partial = 0
    retrieval_perfect = 0

    for item in data:
        gt_chunks = set(item.get("ground_truth_chunks_idx", []))
        retrieved_chunks = set(item.get("retrieved_chunks_idx", []))
        
        # Some samples might not have GT chunks defined
        if not gt_chunks:
            retrieval_perfect += 1
            continue
            
        inter = gt_chunks.intersection(retrieved_chunks)
        
        if len(inter) == len(gt_chunks):
            # All GT chunks were retrieved
            retrieval_perfect += 1
        elif len(inter) == 0:
            # None of the GT chunks were retrieved
            retrieval_miss += 1
        else:
            # Only some GT chunks were retrieved
            retrieval_partial += 1

    print("=" * 50)
    print(f"总体检索质量分析")
    print(f"总样本数: {total_samples}")
    print("=" * 50)
    print(f"1. 完美检索 (包含所有Ground Truth): {retrieval_perfect} 条 ({retrieval_perfect/total_samples*100:.2f}%)")
    print(f"2. 部分检索 (缺失部分Ground Truth): {retrieval_partial} 条 ({retrieval_partial/total_samples*100:.2f}%)")
    print(f"3. 彻底检错 (完全没碰到Ground Truth): {retrieval_miss} 条 ({retrieval_miss/total_samples*100:.2f}%)")
    
    total_error = retrieval_partial + retrieval_miss
    print("-" * 50)
    print(f"-> 检索本身存在缺失/错误的总数: {total_error} 条 ({total_error/total_samples*100:.2f}%)")

if __name__ == "__main__":
    analyze_retrieval("/home/ai-faculty/workspace/Q-RAG/runs/QRAG_hotpotqa_4090_24h15m_50/llm-answering_planner_eval.json")
