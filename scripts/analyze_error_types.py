import json
from collections import defaultdict

def analyze_errors(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = defaultdict(list)
    
    refusal_words = [
        "unable", "not mentioned", "not provided", "none",
        "unknown", "doesn't provide", "do not provide", 
        "cannot determine", "not possible", "no information",
        "does not support"
    ]

    for item in data:
        # 1. Check Retrieval Failure
        # For HotpotQA, usually GT chunks need to be in retrieved chunks
        gt_chunks = set(item.get("ground_truth_chunks_idx", []))
        retrieved_chunks = set(item.get("retrieved_chunks_idx", []))
        
        # We only count retrieval failure if gt_chunks exists and wasn't fully retrieved
        if gt_chunks and not gt_chunks.issubset(retrieved_chunks):
            categories["1. 检索失败 (Retrieval Failure): 没召回正确的文本块"].append(item)
            continue
            
        # 2. Check Planner / Hop Failures
        hops = item.get("hop_answers", [])
        if not hops: # Although this shouldn't happen based on code, just in case
            pass
        elif any(str(h).strip() == "" for h in hops):
            categories["2. Planner断链 (Empty Hop): 中间某个子问题直接回答为空"].append(item)
            continue
            
        is_hop_refusal = False
        for h in hops:
            h_lower = str(h).lower()
            # If the answer is literally just "none" or contains refusal phrases
            if str(h).strip() == "none" or any(rw in h_lower for rw in refusal_words):
                is_hop_refusal = True
                break
        if is_hop_refusal:
            categories["3. 中间步拒答 (Hop Refusal): 模型在回答中间问题时表示文段中没有答案"].append(item)
            continue

        # 3. Check Synthesis / Final Failures
        pred_lower = str(item.get("prediction", "")).lower().strip()
        if not pred_lower or pred_lower == "none" or any(rw in pred_lower for rw in refusal_words):
            categories["4. 综合阶段拒答 (Synthesis Refusal): 中间步有答案，但最后拒绝回答"].append(item)
            continue
            
        # 4. Rest are Logic / Extraction errors (Hallucinations, Wrong entity extracted)
        categories["5. 推理/幻觉错误 (Reasoning/Extraction Error): 流程全都走通了，但答案给错了"].append(item)

    print("=" * 60)
    print(f"Total Low F1 Records Analyzed: {len(data)}")
    print("=" * 60)
    
    for category in sorted(categories.keys()):
        items = categories[category]
        pct = (len(items) / len(data)) * 100
        print(f"{category}")
        print(f"数量: {len(items)} 条 ({pct:.1f}%)")
        print("-" * 60)

    # We will save the breakdown so you can inspect them easily
    output_path = file_path.replace(".json", "_categorized.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    print(f"\n分类详情已保存至: {output_path}")

if __name__ == "__main__":
    analyze_errors("/home/ai-faculty/workspace/Q-RAG/runs/QRAG_hotpotqa_4090_24h15m_50/llm-answering_planner_eval_f1_under_0.3.json")
