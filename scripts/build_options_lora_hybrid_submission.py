#!/usr/bin/env python3
"""Merge base-model option/MCQ rows with LoRA free-response rows for submission."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from judger import Judger
from scripts.create_submission import extract_letter, is_problem_type_mcq_like


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def resolve_submission_jsonl(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "submission.jsonl"
        if candidate.exists():
            return candidate
    if path.exists():
        return path
    raise FileNotFoundError(f"Submission JSONL not found: {path}")


def load_records(paths: list[Path]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in paths:
        jsonl_path = resolve_submission_jsonl(path)
        for record in load_jsonl(jsonl_path):
            records[int(record["id"])] = record
    return records


def write_outputs(output_dir: Path, records: list[dict[str, Any]], submission_name: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / submission_name
    jsonl_path = csv_path.with_suffix(".jsonl")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "response"])
        writer.writeheader()
        for record in records:
            writer.writerow({"id": int(record["id"]), "response": record.get("response", "")})

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def inspect_records(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    judger = Judger(strict_extract=False)
    by_id = {int(row["id"]): row for row in rows}
    report = {
        "empty": 0,
        "options_no_letter": 0,
        "no_options_no_extract": 0,
        "no_options_count_mismatch": 0,
        "issues": [],
    }
    for record in records:
        row = by_id[int(record["id"])]
        response = str(record.get("response", ""))
        if not response.strip():
            report["empty"] += 1
            report["issues"].append((int(record["id"]), "empty_response", ""))
            continue
        if row.get("options"):
            if not extract_letter(response, row.get("options"), judger):
                report["options_no_letter"] += 1
                report["issues"].append((int(record["id"]), "options_no_letter", response[:160]))
            continue

        expected = max(1, str(row.get("question", "")).count("[ANS]"))
        try:
            extracted = judger.extract_ans(response) or ""
        except Exception:
            extracted = ""
        if not extracted:
            report["no_options_no_extract"] += 1
            report["issues"].append((int(record["id"]), "no_options_no_extract", response[:160]))
            continue
        try:
            parts = judger.split_by_comma(extracted)
        except Exception:
            parts = [extracted]
        if len(parts) != expected:
            report["no_options_count_mismatch"] += 1
            report["issues"].append((int(record["id"]), f"count_{len(parts)}_expected_{expected}", extracted[:160]))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/private.jsonl"))
    parser.add_argument("--base", type=Path, required=True, help="Base-model result dir or submission.jsonl.")
    parser.add_argument(
        "--lora",
        type=Path,
        action="append",
        required=True,
        help="LoRA result dir or submission.jsonl. May be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submission-name", default="submission.csv")
    parser.add_argument(
        "--route-mode",
        choices=["options", "mcq-like"],
        default="options",
        help=(
            "'options' sends only explicit options rows to base and all no-options rows to LoRA. "
            "'mcq-like' uses the older heuristic: explicit options or inline A/B options go to base."
        ),
    )
    parser.add_argument(
        "--allow-base-fallback",
        action="store_true",
        help="Use base records if a no-options row is missing from the LoRA records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.data_path)
    base_records = load_records([args.base])
    lora_records = load_records(args.lora)

    merged: list[dict[str, Any]] = []
    source_counts = {"base_routed": 0, "lora_routed": 0, "base_fallback": 0}
    missing: list[int] = []
    for row in rows:
        row_id = int(row["id"])
        route_to_base = bool(row.get("options")) if args.route_mode == "options" else is_problem_type_mcq_like(row)
        if route_to_base:
            source = dict(base_records[row_id])
            source["hybrid_source"] = "base_options" if args.route_mode == "options" else "base_mcq_like"
            source_counts["base_routed"] += 1
        elif row_id in lora_records:
            source = dict(lora_records[row_id])
            source["hybrid_source"] = "lora_no_options" if args.route_mode == "options" else "lora_free_like"
            source_counts["lora_routed"] += 1
        elif args.allow_base_fallback:
            source = dict(base_records[row_id])
            source["hybrid_source"] = "base_fallback"
            source_counts["base_fallback"] += 1
        else:
            missing.append(row_id)
            continue
        merged.append(source)

    if missing:
        raise SystemExit(f"Missing LoRA records for no-options rows: {missing[:20]} (count={len(missing)})")

    csv_path, jsonl_path = write_outputs(args.output_dir, merged, args.submission_name)
    report = inspect_records(rows, merged)

    print(f"rows={len(merged)} output={csv_path}")
    print(f"audit={jsonl_path}")
    print(f"source_counts={source_counts}")
    print(
        "diagnostics="
        f"empty:{report['empty']} "
        f"options_no_letter:{report['options_no_letter']} "
        f"no_options_no_extract:{report['no_options_no_extract']} "
        f"no_options_count_mismatch:{report['no_options_count_mismatch']}"
    )
    for issue in report["issues"][:20]:
        print(f"issue={issue}")


if __name__ == "__main__":
    main()
