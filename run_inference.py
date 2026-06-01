#!/usr/bin/env python3
"""Single entry point for the final Kaggle submission pipeline.

The final submission is a routed hybrid:

* MCQ-like rows use the base Qwen3-4B-Thinking model with the compact prompt
  and the same post-processing used by ``scripts/create_submission.py``.
* Free-response rows use the full-parameter GRPO checkpoint, loaded from
  HuggingFace Hub or another HuggingFace/vLLM-compatible model path.

Calling ``run_inference()`` writes a Kaggle-compatible ``id,response`` CSV plus
JSONL audit files for the intermediate and merged predictions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.create_submission import (  # noqa: E402
    DEFAULT_PROMPT_MODULE,
    load_private,
    run_legacy_submission,
    is_problem_type_mcq_like,
    write_outputs,
)


BASE_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_GRPO_MODEL_ID = "sengBJY/CSE151B_FinalProject"
GRPO_MODEL_ENV = "T2T_QWEN_GRPO_MODEL_ID"
DEFAULT_DATA_PATH = Path("data/private.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/final_run_inference")
DEFAULT_SUBMISSION_NAME = "submission.csv"


def _phase_args(
    *,
    model: str,
    data_path: Path,
    output_dir: Path,
    gpu_id: str,
    mcq_config: str,
    free_config: str,
    max_tokens: int,
    fallback_max_tokens: int,
    fallback_tail_tokens: int,
    high_budget_fallback: bool,
    high_budget_max_tokens: int,
    dynamic_free_continuation: bool,
    dynamic_free_max_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    enforce_eager: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_path=str(data_path),
        output_dir=str(output_dir),
        submission_name=DEFAULT_SUBMISSION_NAME,
        model=model,
        gpu_id=gpu_id,
        routing_mode="compact",
        config=free_config,
        prompt_module=DEFAULT_PROMPT_MODULE,
        mcq_prompt="compact",
        free_prompt="compact",
        mcq_config=mcq_config,
        free_config=free_config,
        max_tokens=max_tokens,
        fallback_max_tokens=fallback_max_tokens,
        fallback_tail_tokens=fallback_tail_tokens,
        high_budget_fallback=high_budget_fallback,
        high_budget_max_tokens=high_budget_max_tokens,
        dynamic_free_continuation=dynamic_free_continuation,
        dynamic_free_max_tokens=dynamic_free_max_tokens,
        context_safety_margin=512,
        no_fallback=False,
        no_repair=False,
        stage0_postprocess=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
    )


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _run_phase(
    *,
    label: str,
    rows: list[dict[str, Any]],
    original_ids: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    phase_dir = Path(args.output_dir)
    phase_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        return []

    print(f"[{label}] rows={len(rows)} model={args.model}", flush=True)
    records = run_legacy_submission(args, rows)
    write_outputs(phase_dir, phase_dir / DEFAULT_SUBMISSION_NAME, records, original_ids)
    _release_gpu_memory()
    return records


def _write_merge_report(
    *,
    path: Path,
    data_path: Path,
    output_csv: Path,
    rows: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    mcq_count: int,
    free_count: int,
    mcq_model_id: str,
    free_model_id: str,
) -> None:
    expected_ids = [int(row["id"]) for row in rows]
    seen_ids = [int(record["id"]) for record in final_records]
    missing = sorted(set(expected_ids) - set(seen_ids))
    extra = sorted(set(seen_ids) - set(expected_ids))
    duplicate_count = len(seen_ids) - len(set(seen_ids))
    empty = sum(1 for record in final_records if not str(record.get("response", "")).strip())
    report = {
        "source_data": str(data_path),
        "output_submission": str(output_csv),
        "expected_rows": len(expected_ids),
        "merged_rows": len(final_records),
        "mcq_like_rows_from_base": mcq_count,
        "free_rows_from_grpo": free_count,
        "mcq_model_id": mcq_model_id,
        "free_model_id": free_model_id,
        "missing": missing,
        "extra": extra,
        "duplicate_count": duplicate_count,
        "empty": empty,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_final_outputs(
    *,
    output_dir: Path,
    submission_name: str,
    records: list[dict[str, Any]],
    original_ids: list[int],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / submission_name
    jsonl_path = csv_path.with_suffix(".jsonl")
    by_id = {int(record["id"]): record for record in records}
    missing = [row_id for row_id in original_ids if row_id not in by_id]
    if missing:
        raise RuntimeError(f"Missing final predictions for ids: {missing[:10]} (count={len(missing)})")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "response"])
        writer.writeheader()
        for row_id in original_ids:
            writer.writerow({"id": row_id, "response": by_id[row_id]["response"]})

    with jsonl_path.open("w", encoding="utf-8") as file:
        for row_id in original_ids:
            file.write(json.dumps(by_id[row_id], ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def run_inference(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    submission_name: str = DEFAULT_SUBMISSION_NAME,
    mcq_model_id: str = BASE_MODEL_ID,
    free_model_id: str | None = None,
    gpu_id: str = "0",
    max_tokens: int = 16384,
    fallback_max_tokens: int = 8192,
    fallback_tail_tokens: int = 6000,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 32,
    max_num_batched_tokens: int = 16384,
    enforce_eager: bool = False,
    high_budget_fallback: bool = False,
    high_budget_max_tokens: int = 32768,
    dynamic_free_continuation: bool = False,
    dynamic_free_max_tokens: int = 32768,
) -> Path:
    """Run the full final pipeline and return the final submission CSV path.

    ``free_model_id`` must point to the uploaded full-parameter GRPO checkpoint
    on HuggingFace Hub, or to a local HuggingFace/vLLM-compatible model
    directory. If omitted, it is read from ``T2T_QWEN_GRPO_MODEL_ID`` or falls
    back to ``sengBJY/CSE151B_FinalProject``.
    """

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    free_model_id = free_model_id or os.environ.get(GRPO_MODEL_ENV) or DEFAULT_GRPO_MODEL_ID
    if not free_model_id:
        raise ValueError(
            f"Missing GRPO model id. Pass free_model_id=... or set {GRPO_MODEL_ENV} "
            "to the HuggingFace Hub path for the checkpoint-81 GRPO model."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    rows, original_ids = load_private(data_path)
    mcq_like_rows = [row for row in rows if is_problem_type_mcq_like(row)]
    free_rows = [row for row in rows if not is_problem_type_mcq_like(row)]

    mcq_args = _phase_args(
        model=mcq_model_id,
        data_path=data_path,
        output_dir=output_dir / "mcq_like_base",
        gpu_id=gpu_id,
        mcq_config="greedy_n1",
        # Some MCQ-like rows have inline A/B/C choices but no structured
        # ``options`` field, so the legacy runner treats them as free format.
        # Keep those in the base-model MCQ phase and run them greedily too.
        free_config="greedy_n1",
        max_tokens=max_tokens,
        fallback_max_tokens=fallback_max_tokens,
        fallback_tail_tokens=fallback_tail_tokens,
        high_budget_fallback=high_budget_fallback,
        high_budget_max_tokens=high_budget_max_tokens,
        dynamic_free_continuation=dynamic_free_continuation,
        dynamic_free_max_tokens=dynamic_free_max_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
    )
    free_args = _phase_args(
        model=free_model_id,
        data_path=data_path,
        output_dir=output_dir / "free_grpo",
        gpu_id=gpu_id,
        mcq_config="greedy_n1",
        free_config="sc_n3",
        max_tokens=max_tokens,
        fallback_max_tokens=fallback_max_tokens,
        fallback_tail_tokens=fallback_tail_tokens,
        high_budget_fallback=high_budget_fallback,
        high_budget_max_tokens=high_budget_max_tokens,
        dynamic_free_continuation=dynamic_free_continuation,
        dynamic_free_max_tokens=dynamic_free_max_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
    )

    mcq_records = _run_phase(
        label="mcq-like/base",
        rows=mcq_like_rows,
        original_ids=[int(row["id"]) for row in mcq_like_rows],
        args=mcq_args,
    )
    free_records = _run_phase(
        label="free-response/grpo",
        rows=free_rows,
        original_ids=[int(row["id"]) for row in free_rows],
        args=free_args,
    )

    final_records: list[dict[str, Any]] = []
    by_id: dict[int, dict[str, Any]] = {}
    for record in mcq_records:
        merged = dict(record)
        merged["hybrid_source"] = "base_mcq_like"
        merged["source_model_id"] = mcq_model_id
        by_id[int(record["id"])] = merged
    for record in free_records:
        merged = dict(record)
        merged["hybrid_source"] = "grpo_free_response"
        merged["source_model_id"] = free_model_id
        by_id[int(record["id"])] = merged
    for row_id in original_ids:
        final_records.append(by_id[row_id])

    csv_path, jsonl_path = _write_final_outputs(
        output_dir=output_dir,
        submission_name=submission_name,
        records=final_records,
        original_ids=original_ids,
    )
    _write_merge_report(
        path=output_dir / "merge_validation.json",
        data_path=data_path,
        output_csv=csv_path,
        rows=rows,
        final_records=final_records,
        mcq_count=len(mcq_records),
        free_count=len(free_records),
        mcq_model_id=mcq_model_id,
        free_model_id=free_model_id,
    )
    print(f"Final submission written to {csv_path}", flush=True)
    print(f"Final audit JSONL written to {jsonl_path}", flush=True)
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--submission-name", default=DEFAULT_SUBMISSION_NAME)
    parser.add_argument("--mcq-model-id", default=BASE_MODEL_ID)
    parser.add_argument("--free-model-id", default=None)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--fallback-max-tokens", type=int, default=8192)
    parser.add_argument("--fallback-tail-tokens", type=int, default=6000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--high-budget-fallback", action="store_true")
    parser.add_argument("--high-budget-max-tokens", type=int, default=32768)
    parser.add_argument("--dynamic-free-continuation", action="store_true")
    parser.add_argument("--dynamic-free-max-tokens", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        data_path=args.data_path,
        output_dir=args.output_dir,
        submission_name=args.submission_name,
        mcq_model_id=args.mcq_model_id,
        free_model_id=args.free_model_id,
        gpu_id=args.gpu_id,
        max_tokens=args.max_tokens,
        fallback_max_tokens=args.fallback_max_tokens,
        fallback_tail_tokens=args.fallback_tail_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        high_budget_fallback=args.high_budget_fallback,
        high_budget_max_tokens=args.high_budget_max_tokens,
        dynamic_free_continuation=args.dynamic_free_continuation,
        dynamic_free_max_tokens=args.dynamic_free_max_tokens,
    )


if __name__ == "__main__":
    main()
