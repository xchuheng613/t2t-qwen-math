#!/usr/bin/env python3
"""Score a benchmark run and report accuracy plus formatting diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from judger import Judger
from scripts.verify_public import (
    answer_values,
    extract_letter,
    is_mcq_like,
    load_predictions,
    load_public,
    resolve_prediction_path,
    score_records,
)


def selected_finish_reason(record: dict[str, Any]) -> str:
    finishes = record.get("finish_reasons")
    if isinstance(finishes, list) and finishes:
        idx = int(record.get("chosen_idx") or 0)
        if 0 <= idx < len(finishes):
            return "" if finishes[idx] is None else str(finishes[idx])
        return "" if finishes[0] is None else str(finishes[0])
    finish = record.get("finish_reason")
    return "" if finish is None else str(finish)


def raw_text(record: dict[str, Any]) -> str:
    raw = record.get("raw_response")
    if raw is None:
        raw = record.get("response", "")
    return str(raw)


def response_text(record: dict[str, Any]) -> str:
    return str(record.get("response", ""))


def format_diagnostics(
    judger: Judger,
    public_rows: dict[int, dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    truncation_count = 0
    no_extraction_count = 0
    count_mismatch_count = 0
    mcq_format_error_count = 0
    free_total = 0
    mcq_total = 0
    fallback_used = 0
    stage0_repair_used = 0
    postprocess_used = 0
    response_lengths: list[int] = []
    raw_lengths: list[int] = []

    for record in records:
        row_id = int(record["id"])
        row = public_rows[row_id]
        finish = selected_finish_reason(record).lower()
        if finish and finish != "stop":
            truncation_count += 1
        fallback_used += int(bool(record.get("fallback_used")))
        stage0_repair_used += int(bool(record.get("stage0_repair_used")))
        postprocess_used += int(bool(record.get("postprocess_used")))

        response = response_text(record)
        response_lengths.append(len(response))
        raw_lengths.append(len(raw_text(record)))

        if is_mcq_like(row):
            mcq_total += 1
            if not extract_letter(response, row.get("options"), judger):
                mcq_format_error_count += 1
            continue

        free_total += 1
        gold = answer_values(row)
        try:
            extracted = judger.extract_ans(response) or ""
        except Exception:
            extracted = ""
        if not extracted:
            no_extraction_count += 1
            continue
        try:
            pred_parts = judger.split_by_comma(extracted)
        except Exception:
            pred_parts = [extracted]
        if len(pred_parts) != len(gold):
            count_mismatch_count += 1

    return {
        "truncation_count": truncation_count,
        "no_extraction_count": no_extraction_count,
        "count_mismatch_count": count_mismatch_count,
        "mcq_format_error_count": mcq_format_error_count,
        "format_error_count": no_extraction_count + count_mismatch_count + mcq_format_error_count,
        "fallback_used": fallback_used,
        "stage0_repair_used": stage0_repair_used,
        "postprocess_used": postprocess_used,
        "avg_response_chars": mean(response_lengths) if response_lengths else 0.0,
        "avg_raw_response_chars": mean(raw_lengths) if raw_lengths else 0.0,
        "free_total": free_total,
        "mcq_total": mcq_total,
    }


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path, help="Result directory or prediction JSONL/CSV")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--scored-jsonl", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_path = resolve_prediction_path(args.prediction)
    run_name = args.name or (
        prediction_path.parent.name if prediction_path.stem == "submission" else prediction_path.stem
    )
    public_rows = load_public(args.data_path)
    records = load_predictions(prediction_path)
    records = [record for record in records if int(record["id"]) in public_rows]

    scored, score_summary, _wrong_ids = score_records(public_rows, records, run_name)
    judger = Judger(strict_extract=False)
    diagnostics = format_diagnostics(judger, public_rows, records)

    row: dict[str, Any] = {
        "run_name": run_name,
        "prediction_path": str(prediction_path),
        "data_path": str(args.data_path),
        "total": score_summary.total,
        "correct": score_summary.correct,
        "accuracy": f"{score_summary.accuracy:.6f}",
        "mcq_correct": score_summary.mcq_correct,
        "mcq_total": score_summary.mcq_total,
        "mcq_accuracy": f"{score_summary.mcq_accuracy:.6f}" if score_summary.mcq_total else "",
        "free_correct": score_summary.free_correct,
        "free_total": score_summary.free_total,
        "free_accuracy": f"{score_summary.free_accuracy:.6f}" if score_summary.free_total else "",
        **diagnostics,
    }

    print(
        f"{run_name}: {score_summary.correct}/{score_summary.total} = "
        f"{score_summary.accuracy:.2%}"
    )
    if score_summary.mcq_total:
        print(f"  mcq:  {score_summary.mcq_correct}/{score_summary.mcq_total} = {score_summary.mcq_accuracy:.2%}")
    if score_summary.free_total:
        print(f"  free: {score_summary.free_correct}/{score_summary.free_total} = {score_summary.free_accuracy:.2%}")
    print(
        "  diagnostics: "
        f"truncated={diagnostics['truncation_count']} "
        f"format_errors={diagnostics['format_error_count']} "
        f"count_mismatch={diagnostics['count_mismatch_count']} "
        f"fallback={diagnostics['fallback_used']} "
        f"stage0_repair={diagnostics['stage0_repair_used']} "
        f"avg_response_chars={diagnostics['avg_response_chars']:.1f} "
        f"avg_raw_response_chars={diagnostics['avg_raw_response_chars']:.1f}"
    )

    if args.summary_csv:
        write_csv(args.summary_csv, row)
        print(f"summary_csv={args.summary_csv}")
    if args.scored_jsonl:
        args.scored_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.scored_jsonl.open("w", encoding="utf-8") as file:
            for record in scored:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"scored_jsonl={args.scored_jsonl}")


if __name__ == "__main__":
    main()
