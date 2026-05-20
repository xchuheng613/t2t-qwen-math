#!/usr/bin/env python3
"""Diff two benchmark prediction files into correctness transition groups."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from judger import Judger
from scripts.verify_public import (
    answer_values,
    extract_answer,
    load_predictions,
    load_public,
    resolve_prediction_path,
    score_response,
)


GROUP_LABELS = {
    (True, True): "base_correct_lora_correct",
    (True, False): "base_correct_lora_wrong",
    (False, True): "base_wrong_lora_correct",
    (False, False): "base_wrong_lora_wrong",
}


def by_id(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(record["id"]): record for record in records}


def selected_finish_reason(record: dict[str, Any]) -> str:
    finishes = record.get("finish_reasons")
    if isinstance(finishes, list) and finishes:
        idx = int(record.get("chosen_idx") or 0)
        if 0 <= idx < len(finishes):
            return "" if finishes[idx] is None else str(finishes[idx])
        return "" if finishes[0] is None else str(finishes[0])
    finish = record.get("finish_reason")
    return "" if finish is None else str(finish)


def response_text(record: dict[str, Any]) -> str:
    return str(record.get("response", ""))


def short_text(text: str, max_chars: int = 500) -> str:
    text = str(text).replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="Base result dir or JSONL/CSV")
    parser.add_argument("--lora", type=Path, required=True, help="LoRA result dir or JSONL/CSV")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/lora_diffs"))
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = resolve_prediction_path(args.base)
    lora_path = resolve_prediction_path(args.lora)
    name = args.name or f"{base_path.parent.name}__vs__{lora_path.parent.name}"

    public_rows = load_public(args.data_path)
    base_records = by_id(load_predictions(base_path))
    lora_records = by_id(load_predictions(lora_path))
    expected_ids = set(public_rows)
    missing_base = sorted(expected_ids - set(base_records))
    missing_lora = sorted(expected_ids - set(lora_records))
    if missing_base:
        raise SystemExit(f"Base predictions missing ids: {missing_base[:20]} count={len(missing_base)}")
    if missing_lora:
        raise SystemExit(f"LoRA predictions missing ids: {missing_lora[:20]} count={len(missing_lora)}")

    judger = Judger(strict_extract=False)
    groups: dict[str, list[dict[str, Any]]] = {label: [] for label in GROUP_LABELS.values()}

    for row_id in sorted(public_rows):
        row = public_rows[row_id]
        base_record = base_records[row_id]
        lora_record = lora_records[row_id]
        base_response = response_text(base_record)
        lora_response = response_text(lora_record)
        base_correct = score_response(judger, row, base_response)
        lora_correct = score_response(judger, row, lora_response)
        label = GROUP_LABELS[(base_correct, lora_correct)]

        try:
            base_extracted = extract_answer(judger, base_response)
        except Exception:
            base_extracted = ""
        try:
            lora_extracted = extract_answer(judger, lora_response)
        except Exception:
            lora_extracted = ""

        groups[label].append(
            {
                "id": row_id,
                "group": label,
                "base_correct": base_correct,
                "lora_correct": lora_correct,
                "gold": answer_values(row),
                "question": row.get("question", ""),
                "options": row.get("options"),
                "base_response": base_response,
                "lora_response": lora_response,
                "base_extracted": base_extracted,
                "lora_extracted": lora_extracted,
                "base_finish_reason": selected_finish_reason(base_record),
                "lora_finish_reason": selected_finish_reason(lora_record),
                "base_fallback_used": bool(base_record.get("fallback_used")),
                "lora_fallback_used": bool(lora_record.get("fallback_used")),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{name}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["group", "count"])
        writer.writeheader()
        for label in GROUP_LABELS.values():
            writer.writerow({"group": label, "count": len(groups[label])})

    for label, rows in groups.items():
        jsonl_path = args.out_dir / f"{name}__{label}.jsonl"
        csv_path = args.out_dir / f"{name}__{label}.csv"
        with jsonl_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            fields = [
                "id",
                "gold",
                "base_extracted",
                "lora_extracted",
                "base_finish_reason",
                "lora_finish_reason",
                "base_fallback_used",
                "lora_fallback_used",
                "question",
            ]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "id": row["id"],
                        "gold": json.dumps(row["gold"], ensure_ascii=False),
                        "base_extracted": short_text(row["base_extracted"], 240),
                        "lora_extracted": short_text(row["lora_extracted"], 240),
                        "base_finish_reason": row["base_finish_reason"],
                        "lora_finish_reason": row["lora_finish_reason"],
                        "base_fallback_used": int(row["base_fallback_used"]),
                        "lora_fallback_used": int(row["lora_fallback_used"]),
                        "question": short_text(row["question"], 500),
                    }
                )

    counts = Counter({label: len(rows) for label, rows in groups.items()})
    print(f"base={base_path}")
    print(f"lora={lora_path}")
    print(f"data={args.data_path}")
    for label in GROUP_LABELS.values():
        print(f"{label}: {counts[label]}")
    print(f"summary={summary_path}")
    print(f"groups_dir={args.out_dir}")


if __name__ == "__main__":
    main()
