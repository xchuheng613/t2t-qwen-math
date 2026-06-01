#!/usr/bin/env python3
"""Normalize a QA dataset into id/instruction/output JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lora_reasoning_common import load_rows, normalize_row, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/public.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/normalized/public_normalized.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--raw-copy",
        type=Path,
        default=Path("data/raw/public.jsonl"),
        help="Optional copy of the loaded raw rows for reproducibility. Use '' to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input, limit=args.limit)

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for index, row in enumerate(rows):
        item = normalize_row(row, index)
        if item["id"] in seen:
            duplicates.append(item["id"])
        seen.add(item["id"])
        normalized.append(item)

    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise SystemExit(f"Duplicate normalized IDs found: {preview}")

    count = write_jsonl(args.output, normalized)
    if args.raw_copy and str(args.raw_copy).strip():
        args.raw_copy.parent.mkdir(parents=True, exist_ok=True)
        with args.raw_copy.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"normalized={count} output={args.output}")
    if args.raw_copy and str(args.raw_copy).strip():
        print(f"raw_copy={args.raw_copy}")


if __name__ == "__main__":
    main()
