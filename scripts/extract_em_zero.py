#!/usr/bin/env python3
"""Extract records with EM == 0 from evaluation result files.

Supports:
- JSON files containing a top-level list of records
- JSONL files containing one JSON object per line
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _to_float(value: Any) -> float | None:
    """Convert value to float when possible; return None on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_em_zero(record: dict[str, Any]) -> bool:
    """True if record has an EM value numerically equal to 0."""
    if "EM" not in record:
        return False
    em = _to_float(record.get("EM"))
    return em == 0.0


def read_records(input_path: Path) -> list[dict[str, Any]]:
    """Load records from JSON or JSONL file."""
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Line {line_no} in {input_path} is not a JSON object"
                    )
                records.append(obj)
        return records

    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{input_path} must contain a top-level JSON list")
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Item {idx} in {input_path} is not a JSON object"
                )
        return data

    raise ValueError(
        f"Unsupported file extension: {input_path.suffix}. Use .json or .jsonl"
    )


def write_json(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write records as formatted JSON array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract records where EM == 0 from eval result file"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input file path (.json or .jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <input_stem>_em0.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_name(f"{input_path.stem}_em0.json")

    records = read_records(input_path)
    em_zero_records = [record for record in records if is_em_zero(record)]

    write_json(em_zero_records, output_path)

    print(f"Input: {input_path}")
    print(f"Total records: {len(records)}")
    print(f"EM=0 records: {len(em_zero_records)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
