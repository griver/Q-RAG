import json
import os


def load_data(file_path):
    """
    Load data from a .json or .jsonl file.
    """
    if file_path.endswith(".jsonl"):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                obj = json.loads(line)
                data.append(obj)
        return data

    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # 如果你的 json 最外层是 dict，这里尽量兼容一下
            # 例如 {"data": [...]} 这种格式
            if "data" in data and isinstance(data["data"], list):
                return data["data"]

            raise ValueError("JSON file is a dict, but no list found under key 'data'.")

    raise ValueError("Only .json and .jsonl files are supported.")


def is_empty_sub_questions(value):
    """
    Check whether sub_questions is effectively empty.
    Empty cases include:
    - key missing (handled outside)
    - None
    - ""
    - []
    - list with only empty / blank strings
    """
    if value is None:
        return True

    if isinstance(value, str):
        if value.strip() == "":
            return True
        return False

    if isinstance(value, list):
        if len(value) == 0:
            return True

        has_non_empty_item = False

        for item in value:
            if item is None:
                continue

            if isinstance(item, str):
                if item.strip() != "":
                    has_non_empty_item = True
                    break
            else:
                # 如果列表里有非字符串内容，也视为“不是空”
                has_non_empty_item = True
                break

        if not has_non_empty_item:
            return True

        return False

    return False


def extract_empty_hop_answers(data):
    """
    Extract samples whose hop_answers field contains an empty string.
    """
    empty_samples = []
    total_count = len(data)

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        hop_answers = item.get("hop_answers", [])
        
        # 只要 hop_answers 里有任何一个元素是纯空字符串，就归为包含空回答的情况
        has_empty = any(isinstance(ans, str) and ans.strip() == "" for ans in hop_answers)

        if has_empty:
            item_copy = dict(item)
            item_copy["_sample_index"] = idx
            empty_samples.append(item_copy)

    empty_count = len(empty_samples)
    empty_ratio = 0.0

    if total_count > 0:
        empty_ratio = empty_count / total_count

    return empty_samples, empty_count, total_count, empty_ratio


def save_as_json(output_path, data):
    """
    Save extracted samples as JSON.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_as_jsonl(output_path, data):
    """
    Save extracted samples as JSONL.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + "\n")


def main():
    input_path = "/home/ai-faculty/workspace/Q-RAG/runs/QRAG_hotpotqa_4090_24h15m_50/llm-answering_planner_eval.json"
    output_json_path = "empty_hop_answers_samples.json"
    output_jsonl_path = "empty_hop_answers_samples.jsonl"

    data = load_data(input_path)

    empty_samples, empty_count, total_count, empty_ratio = extract_empty_hop_answers(data)

    print("=" * 60)
    print("Hop answers empty check summary")
    print(f"Input file          : {input_path}")
    print(f"Total samples       : {total_count}")
    print(f"Empty hop_answers   : {empty_count} (Samples containing at least one empty string)")
    print(f"Empty ratio         : {empty_ratio:.2%}")
    print("=" * 60)

    save_as_json(output_json_path, empty_samples)
    save_as_jsonl(output_jsonl_path, empty_samples)

    print(f"Saved JSON  : {output_json_path}")
    print(f"Saved JSONL : {output_jsonl_path}")


if __name__ == "__main__":
    main()