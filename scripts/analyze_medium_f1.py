import json
from collections import defaultdict

def analyze_medium_f1(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 提取 0.3 <= F1 < 0.6
    filtered_data = []
    for item in data:
        try:
            f1 = float(item.get("F1", 0.0))
        except:
            f1 = 0.0
        if 0.3 <= f1 < 0.6:
            filtered_data.append(item)

    # 保存提取后的数据
    out_filtered = input_file.replace(".json", "_f1_0.3_to_0.6.json")
    with open(out_filtered, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    # 2. 错误原因归类
    categories = defaultdict(list)
    refusal_words = [
        "unable", "not mentioned", "not provided", "none",
        "unknown", "doesn't provide", "do not provide", 
        "cannot determine", "not possible", "no information",
        "does not support"
    ]

    for item in filtered_data:
        gt_chunks = set(item.get("ground_truth_chunks_idx", []))
        retrieved_chunks = set(item.get("retrieved_chunks_idx", []))
        
        # 1. 检索失败
        if gt_chunks and not gt_chunks.issubset(retrieved_chunks):
            categories["1. 检索缺失 (Retrieval Miss): 缺失部分或全部关键文本块"].append(item)
            continue
            
        hops = item.get("hop_answers", [])
        # 2. Planner 断链为空
        if any(str(h).strip() == "" for h in hops):
            categories["2. Planner断链 (Empty Hop): 中间子问题回答为空 (但侥幸猜对部分单词)"].append(item)
            continue
            
        # 3. 中间步拒答
        is_hop_refusal = False
        for h in hops:
            if str(h).strip().lower() == "none" or any(rw in str(h).lower() for rw in refusal_words):
                is_hop_refusal = True
                break
        if is_hop_refusal:
            categories["3. 中间步拒答 (Hop Refusal): 中间步自称找不到答案 (但最后侥幸命中单词)"].append(item)
            continue

        # 4. 综合步拒答
        pred_lower = str(item.get("prediction", "")).lower().strip()
        if not pred_lower or pred_lower == "none" or any(rw in pred_lower for rw in refusal_words):
            categories["4. 综合阶段拒答 (Synthesis Refusal): 模型输出拒答但包含原问题词汇导致F1>0"].append(item)
            continue
            
        # 5. 回答过于冗长或包含多余信息（中等F1分段的典型特征）
        categories["5. 边界粗糙/过度生成 (Partial/Verbose Match): 流程没报错，找对了核心，但回答带了多余的无用信息/句子长"].append(item)

    print("=" * 60)
    print(f"分析区间: 0.3 <= F1 < 0.6")
    print(f"符合该区间的数据总量: {len(filtered_data)}")
    print("=" * 60)
    
    for category in sorted(categories.keys()):
        items = categories[category]
        pct = (len(items) / len(filtered_data)) * 100 if filtered_data else 0
        print(f"{category}")
        print(f"数量: {len(items)} 条 ({pct:.1f}%)")
        print("-" * 60)

    # 保存归类文件
    out_cat = input_file.replace(".json", "_f1_0.3_to_0.6_categorized.json")
    with open(out_cat, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    print(f"原数据已保存至: {out_filtered}")
    print(f"分类详情已保存至: {out_cat}")

if __name__ == "__main__":
    analyze_medium_f1("/home/ai-faculty/workspace/Q-RAG/runs/QRAG_hotpotqa_4090_24h15m_50/llm-answering_planner_eval.json")
