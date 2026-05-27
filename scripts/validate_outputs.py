#!/usr/bin/env python3
"""Validate worker outputs against normalized input examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
    parser.add_argument("--report", type=Path, default=Path("output/validation_report.md"))
    parser.add_argument("--retry-queue-out", type=Path, default=Path("logs/retry_queue.jsonl"))
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def load_failed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("id", "")) for row in load_jsonl(path)}


def main() -> None:
    args = parse_args()
    input_rows, input_by_id, duplicate_ids = load_input_by_id(args.normalized)
    failed_ids = load_failed_ids(args.failed)

    success_count = 0
    invalid_details: list[str] = []
    invalid_retry_rows: list[dict[str, Any]] = []
    missing_retry_rows: list[dict[str, Any]] = []

    processed_ids: set[str] = set()
    duplicate_processed: list[str] = []
    for row in input_rows:
        row_id = str(row["id"])
        path = task_path_for_id(args.processed_dir, row_id)
        if not path.exists():
            if row_id not in failed_ids:
                missing_retry_rows.append(row)
            continue
        if row_id in processed_ids:
            duplicate_processed.append(row_id)
        processed_ids.add(row_id)
        try:
            record = read_json_object(path)
        except ValueError as exc:
            invalid_details.append(f"{row_id}: {exc}")
            invalid_retry_rows.append(row)
            continue
        errors = validate_worker_record(record, row)
        if errors:
            invalid_details.append(f"{row_id}: {'; '.join(errors)}")
            invalid_retry_rows.append(row)
        else:
            success_count += 1

    for row_id in duplicate_processed:
        invalid_details.append(f"{row_id}: duplicate processed output")

    unknown_processed = []
    for path in args.processed_dir.glob("*.json"):
        try:
            record = read_json_object(path)
        except ValueError:
            continue
        row_id = str(record.get("id", ""))
        if row_id and row_id not in input_by_id:
            unknown_processed.append(row_id)
            invalid_details.append(f"{row_id}: processed output has no matching input")

    retry_rows = invalid_retry_rows + missing_retry_rows
    write_jsonl(args.retry_queue_out, retry_rows)

    report = validation_report_markdown(
        input_count=len(input_rows),
        success_count=success_count,
        failure_count=len(failed_ids),
        missing_count=len(missing_retry_rows),
        invalid_count=len(invalid_details),
        duplicate_ids=duplicate_ids,
        invalid_details=invalid_details,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print(f"input={len(input_rows)} success={success_count} failed={len(failed_ids)}")
    print(f"missing={len(missing_retry_rows)} invalid={len(invalid_details)} retry_queue={args.retry_queue_out}")
    print(f"report={args.report}")

    if duplicate_ids or invalid_details or (missing_retry_rows and not args.allow_missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
