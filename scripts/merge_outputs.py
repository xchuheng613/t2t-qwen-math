#!/usr/bin/env python3
"""Merge valid single-example worker outputs into the final reasoning JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lora_reasoning_common import (
    load_input_by_id,
    load_jsonl,
    read_json_object,
    task_path_for_id,
    validate_worker_record,
    validation_report_markdown,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, default=Path("data/normalized/public_normalized.jsonl"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_tasks"))
    parser.add_argument("--failed", type=Path, default=Path("output/failed_examples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/lora_training_data.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("output/validation_report.md"))
    return parser.parse_args()


def load_failure_by_id(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    failures: dict[str, dict] = {}
    for row in load_jsonl(path):
        row_id = str(row.get("id", ""))
        if row_id:
            failures[row_id] = row
    return failures


def main() -> None:
    args = parse_args()
    input_rows, input_by_id, duplicate_ids = load_input_by_id(args.normalized)
    failure_by_id = load_failure_by_id(args.failed)

    merged = []
    invalid_details: list[str] = []
    missing_count = 0
    final_failures = []

    for source in input_rows:
        row_id = str(source["id"])
        processed_path = task_path_for_id(args.processed_dir, row_id)
        has_failure = row_id in failure_by_id
        if processed_path.exists():
            try:
                record = read_json_object(processed_path)
            except ValueError as exc:
                invalid_details.append(f"{row_id}: {exc}")
                continue
            errors = validate_worker_record(record, source)
            if errors:
                invalid_details.append(f"{row_id}: {'; '.join(errors)}")
                continue
            if has_failure:
                invalid_details.append(f"{row_id}: has both a success output and a permanent failure record")
                continue
            merged.append(
                {
                    "id": record["id"],
                    "instruction": record["instruction"],
                    "reasoning": record["reasoning"].strip(),
                    "output": record["output"],
                    "warning": record["warning"],
                    "source": record["source"],
                }
            )
        elif has_failure:
            final_failures.append(failure_by_id[row_id])
        else:
            missing_count += 1

    output_ids = [str(row["id"]) for row in merged]
    if len(output_ids) != len(set(output_ids)):
        invalid_details.append("duplicate IDs in merged output")

    for path in args.processed_dir.glob("*.json"):
        try:
            record = read_json_object(path)
        except ValueError:
            continue
        row_id = str(record.get("id", ""))
        if row_id and row_id not in input_by_id:
            invalid_details.append(f"{row_id}: processed output has no matching input")

    write_jsonl(args.output, merged)
    if final_failures:
        write_jsonl(args.failed, final_failures)

    report = validation_report_markdown(
        input_count=len(input_rows),
        success_count=len(merged),
        failure_count=len(final_failures),
        missing_count=missing_count,
        invalid_count=len(invalid_details),
        duplicate_ids=duplicate_ids,
        invalid_details=invalid_details,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print(f"merged={len(merged)} output={args.output}")
    print(f"failures={len(final_failures)} missing={missing_count} invalid={len(invalid_details)}")
    print(f"report={args.report}")

    if duplicate_ids or invalid_details or missing_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
