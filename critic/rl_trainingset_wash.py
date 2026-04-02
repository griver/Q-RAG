"""
MuSiQue Curriculum Data Preparation
=====================================
从本地 20000 条 MuSiQue 数据中筛选出三阶段课程学习训练集。

输入格式（MuSiQue 原始格式，每行一个 JSON）：
  {
    "id": "2hop__123456",
    "question": "...",
    "answer": "...",
    "question_decomposition": [{"question": ..., "answer": ...}, ...],
    "paragraphs": [{"paragraph_text": ..., "is_supporting": ...}, ...],
    "answerable": true
  }

输出格式（curriculum_train.jsonl，每行一个 JSON）：
  {
    "question": "...",
    "answer": "...",
    "pred_texts": ["passage1", "passage2", ...],   # 所有段落文本
    "supporting_texts": ["support1", ...],          # 仅 supporting 段落
    "num_hops": 2,
    "curriculum_stage": 1,
    "gold_sub_questions": ["Step 1: ...", "Step 2: ..."],  # 来自 question_decomposition
    "id": "2hop__123456"
  }

阶段划分：
  Stage 1:  500 条，全 2-hop      → Critic 学基础 Accept/Reject
  Stage 2: 1000 条，2+3-hop 混合  → Critic 学中等复杂度
  Stage 3: 1500 条，全难度        → Critic 泛化到复杂问题

目标 EM/F1 参考：
  Stage 1: EM ≥ 0.45, F1 ≥ 0.55
  Stage 2: EM ≥ 0.38, F1 ≥ 0.48
  Stage 3: EM ≥ 0.30, F1 ≥ 0.40
"""

import json
import random
import argparse
import re
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────
#  Stage 配置
#  每个 stage 的目标数量和 hop 组成
# ─────────────────────────────────────────────
STAGE_CONFIG = {
    1: {"total": 500,  "hops": {2: 1.0}},               # 100% 2-hop
    2: {"total": 1000, "hops": {2: 0.4, 3: 0.6}},       # 40% 2-hop, 60% 3-hop
    3: {"total": 1500, "hops": {2: 0.2, 3: 0.4, 4: 0.4}},# 20/40/40
}

# 每个 stage 达标才进入下一阶段的 EM/F1 参考阈值
STAGE_THRESHOLDS = {
    1: {"EM": 0.45, "F1": 0.55},
    2: {"EM": 0.38, "F1": 0.48},
    3: {"EM": 0.30, "F1": 0.40},
}


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def get_num_hops(record: dict) -> int:
    """
    从 MuSiQue 记录中获取 hop 数。
    优先从 id 字段解析（最可靠），fallback 到 question_decomposition 长度。
    """
    rid = record.get("id", "")
    match = re.match(r"(\d+)hop", rid)
    if match:
        return int(match.group(1))
    # fallback
    decomp = record.get("question_decomposition", [])
    return len(decomp) if decomp else 0


def extract_pred_texts(record: dict) -> list[str]:
    """提取所有段落文本（作为检索结果传给 agent loop）。"""
    return [
        p["paragraph_text"].strip()
        for p in record.get("paragraphs", [])
        if p.get("paragraph_text", "").strip()
    ]


def extract_supporting_texts(record: dict) -> list[str]:
    """只提取 supporting 段落（gold evidence）。"""
    return [
        p["paragraph_text"].strip()
        for p in record.get("paragraphs", [])
        if p.get("is_supporting", False) and p.get("paragraph_text", "").strip()
    ]


def extract_gold_sub_questions(record: dict) -> list[str]:
    """
    从 question_decomposition 提取 gold sub-questions。
    格式化为 'Step N: question' 供 Planner 参考（不是训练信号，仅做分析用）。
    """
    steps = []
    for i, step in enumerate(record.get("question_decomposition", []), start=1):
        q = step.get("question", "").strip()
        if q:
            steps.append(f"Step {i}: {q}")
    return steps


def convert_record(record: dict, stage: int) -> dict:
    """把 MuSiQue 原始记录转换成训练格式。"""
    return {
        "id":                record.get("id", ""),
        "question":          record.get("question", "").strip(),
        "answer":            record.get("answer", "").strip(),
        "pred_texts":        extract_pred_texts(record),
        "supporting_texts":  extract_supporting_texts(record),
        "num_hops":          get_num_hops(record),
        "gold_sub_questions":extract_gold_sub_questions(record),
        "answerable":        record.get("answerable", True),
        "curriculum_stage":  stage,
    }


def is_valid(record: dict) -> bool:
    """过滤掉质量差的样本。"""
    if not record.get("answerable", True):
        return False                          # 跳过不可回答的问题
    if not record.get("question", "").strip():
        return False
    if not record.get("answer", "").strip():
        return False
    if len(record.get("paragraphs", [])) == 0 and len(record.get("pred_texts", [])) == 0:
        return False
    if record.get("num_hops", 0) < 2:
        return False
    return True


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, required=True,
                        help="原始 MuSiQue JSONL 文件路径")
    parser.add_argument("--output", type=str, default="curriculum_train.jsonl",
                        help="输出文件路径")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 读入数据 ──────────────────────────────
    print(f"Loading {args.input} ...")
    raw_data = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))
    print(f"Raw records: {len(raw_data)}")

    # ── 按 hop 数分桶 ─────────────────────────
    buckets = defaultdict(list)
    skip_count = 0
    for record in raw_data:
        record["num_hops"] = get_num_hops(record)
        if is_valid(record):
            hop = min(record["num_hops"], 4)  # 4+ 都归入 4-hop 桶
            buckets[hop].append(record)
        else:
            skip_count += 1

    print(f"\nSkipped (invalid/unanswerable): {skip_count}")
    print("Bucket distribution:")
    for hop in sorted(buckets.keys()):
        print(f"  {hop}-hop: {len(buckets[hop])} samples")

    # ── 检查数据是否足够 ─────────────────────
    needed = defaultdict(int)
    for stage_id, cfg in STAGE_CONFIG.items():
        for hop, ratio in cfg["hops"].items():
            needed[hop] += int(cfg["total"] * ratio)

    print("\nRequired per hop:")
    all_ok = True
    for hop, n in sorted(needed.items()):
        available = len(buckets[hop])
        status = "✓" if available >= n else "✗ INSUFFICIENT"
        print(f"  {hop}-hop: need {n}, have {available}  {status}")
        if available < n:
            all_ok = False

    if not all_ok:
        print("\nWARNING: Not enough data in some buckets.")
        print("Will use all available samples for those hops (may be less than target).")

    # ── 打乱每个桶 ───────────────────────────
    for hop in buckets:
        random.shuffle(buckets[hop])

    # ── 按阶段采样（不重叠）─────────────────
    # 用 pointer 追踪每个桶已经用了多少
    pointers = defaultdict(int)
    all_output = []

    for stage_id, cfg in STAGE_CONFIG.items():
        stage_records = []
        print(f"\nStage {stage_id} (target: {cfg['total']} samples):")

        for hop, ratio in cfg["hops"].items():
            target_n = int(cfg["total"] * ratio)
            start    = pointers[hop]
            end      = start + target_n
            sampled  = buckets[hop][start:end]
            pointers[hop] = end

            actual_n = len(sampled)
            print(f"  {hop}-hop: {actual_n}/{target_n} samples")

            for record in sampled:
                stage_records.append(convert_record(record, stage=stage_id))

        # 打乱阶段内顺序（混合不同 hop）
        random.shuffle(stage_records)
        all_output.extend(stage_records)
        print(f"  Stage {stage_id} total: {len(stage_records)} samples")

    # ── 写出 ─────────────────────────────────
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_output:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── 统计报告 ─────────────────────────────
    print("\n" + "=" * 55)
    print("  CURRICULUM DATASET SUMMARY")
    print("=" * 55)
    stage_counts = defaultdict(int)
    hop_counts   = defaultdict(int)
    for r in all_output:
        stage_counts[r["curriculum_stage"]] += 1
        hop_counts[r["num_hops"]] += 1

    for s in sorted(stage_counts):
        thresh = STAGE_THRESHOLDS[s]
        print(f"  Stage {s}: {stage_counts[s]:4d} samples  "
              f"(target EM≥{thresh['EM']:.2f}, F1≥{thresh['F1']:.2f})")

    print()
    for h in sorted(hop_counts):
        print(f"  {h}-hop: {hop_counts[h]} samples")

    print(f"\n  Total: {len(all_output)} samples")
    print(f"  Saved to: {output_path.resolve()}")
    print("=" * 55)

    # ── 输出示例 ─────────────────────────────
    if args.verbose:
        print("\n--- Example record (Stage 1) ---")
        for r in all_output:
            if r["curriculum_stage"] == 1:
                print(json.dumps(r, indent=2, ensure_ascii=False)[:800] + "...")
                break


if __name__ == "__main__":
    main()