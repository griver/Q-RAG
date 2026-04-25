#!/usr/bin/env python3
"""提取 F1 分数小于 0.3 的评测数据"""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Extract records where F1 < 0.3")
    parser.add_argument("-i", "--input", required=True, type=Path, help="输入评测结果文件 (.json)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出文件 (.json)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_name(f"{input_path.stem}_f1_under_0.3.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 筛选 F1 小于 0.3 的样本（包含 None/缺省情况算作 0）
    low_f1_records = []
    for item in data:
        try:
            f1 = float(item.get("F1", 0.0))
        except (TypeError, ValueError):
            f1 = 0.0
            
        if f1 < 0.3:
            low_f1_records.append(item)

    # 保存文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(low_f1_records, f, ensure_ascii=False, indent=2)

    print(f"数据处理完成！")
    print(f"总计输入样本: {len(data)} 条")
    print(f"F1 < 0.3 样本: {len(low_f1_records)} 条")
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
