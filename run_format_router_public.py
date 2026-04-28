#!/usr/bin/env python3
"""Run the full format router on public.jsonl and score immediately."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from create_submission import CONFIGS, build_format_routed_groups, generate_group
from format_router import (
    FORMAT_ALGORITHM_SEQUENCE,
    FORMAT_MCQ,
    FORMAT_MULTI_SELECT,
    answer_values,
    extract_choice_prediction,
    normalize_choice_letters,
    normalize_item,
    route_item,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_DATA_PATH = "data/public.jsonl"
DEFAULT_OUTPUT_DIR = "results/public_format_router"


def load_public(path: Path) -> list[dict[str, Any]]:
    return [normalize_item(json.loads(line)) for line in path.open()]


def select_items(rows: list[dict[str, Any]], num_examples: int, seed: int, shuffle: bool) -> list[dict[str, Any]]:
    pool = rows[:]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pool)
    return pool if num_examples < 0 else pool[:num_examples]


def score_record(item: dict[str, Any], record: dict[str, Any], judger: Any) -> tuple[bool, str]:
    route = route_item(item)
    response = str(record.get("response") or "")
    gold_values = answer_values(item)
    if not gold_values:
        return False, ""

    if route.format_type in {FORMAT_MCQ, FORMAT_ALGORITHM_SEQUENCE, FORMAT_MULTI_SELECT}:
        multi = route.format_type == FORMAT_MULTI_SELECT
        prediction = extract_choice_prediction(response, item, multi=multi, judger=None)
        if not prediction:
            prediction = str(record.get("vote_key") or "").strip().upper()

        if multi:
            gold = normalize_choice_letters(gold_values[0], len(item.get("options") or []))
            pred = normalize_choice_letters(prediction, len(item.get("options") or []))
            return bool(pred) and pred == gold, pred

        pred = prediction.strip().upper()
        return pred == gold_values[0].strip().upper(), pred

    try:
        correct = bool(
            judger.auto_judge(
                pred=response,
                gold=gold_values,
                options=[[]] * len(gold_values),
            )
        )
    except Exception:
        correct = False

    try:
        extracted = judger.extract_ans(response) or ""
    except Exception:
        extracted = ""
    return correct, str(extracted)


def write_records(path: Path, records: list[dict[str, Any]], original_ids: list[int]) -> None:
    by_id = {record["id"]: record for record in records}
    with path.open("w", encoding="utf-8") as file:
        for row_id in original_ids:
            file.write(json.dumps(by_id[row_id], ensure_ascii=False) + "\n")


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(label: str, subset: list[dict[str, Any]]) -> None:
        correct = sum(bool(record["correct"]) for record in subset)
        n = len(subset)
        rows.append(
            {
                "group": label,
                "n": n,
                "correct": correct,
                "acc": correct / n if n else 0.0,
                "fallback_used": sum(bool(record.get("fallback_used")) for record in subset),
            }
        )

    add_row("overall", records)
    for format_type in sorted({record["format_type"] for record in records}):
        add_row(f"format:{format_type}", [record for record in records if record["format_type"] == format_type])
    for prompt in sorted({record["prompt"] for record in records}):
        add_row(f"prompt:{prompt}", [record for record in records if record["prompt"] == prompt])
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["group", "n", "correct", "acc", "fallback_used"])
        writer.writeheader()
        writer.writerows(rows)


def make_judger() -> Any:
    try:
        from judger import Judger

        return Judger(strict_extract=False)
    except ImportError as exc:
        raise SystemExit(
            "Could not initialize the judger. Activate the notebook/CSE environment "
            "or install requirements.txt, including antlr4-python3-runtime."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="format_router_public")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--num-examples", type=int, default=100, help="Use -1 for the full public set.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--fallback-max-tokens", type=int, default=2048)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    sys.path.insert(0, ".")

    from transformers import AutoTokenizer
    from vllm import LLM

    all_rows = load_public(Path(args.data_path))
    rows = select_items(all_rows, args.num_examples, args.seed, args.shuffle)
    original_ids = [int(row["id"]) for row in rows]
    groups = build_format_routed_groups(rows)
    route_counts = Counter(route.format_type for group in groups.values() for _item, route in group)

    print(f"Public set: {len(all_rows)} total; evaluating {len(rows)}", flush=True)
    print(
        "Format counts: " + ", ".join(f"{name}={count}" for name, count in sorted(route_counts.items())),
        flush=True,
    )
    print("Generation groups:", flush=True)
    for index, ((prompt_name, config_name), group) in enumerate(sorted(groups.items()), start=1):
        prompt_counts = Counter(route.format_type for _item, route in group)
        prompt_summary = ", ".join(f"{name}={count}" for name, count in sorted(prompt_counts.items()))
        print(f"  {index}. {prompt_name} / {config_name}: {len(group)} ({prompt_summary})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=args.model,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
    )
    judger = make_judger()

    records: list[dict[str, Any]] = []
    for (prompt_name, config_name), group in sorted(groups.items()):
        records.extend(
            generate_group(
                llm,
                tokenizer,
                judger,
                prompt_name,
                CONFIGS[config_name],
                group,
                args.max_tokens,
                args.fallback_max_tokens,
                not args.no_fallback,
            )
        )

    by_id = {int(row["id"]): row for row in rows}
    for record in records:
        item = by_id[int(record["id"])]
        correct, extracted = score_record(item, record, judger)
        record["gold"] = item.get("answer")
        record["correct"] = correct
        record["extracted"] = extracted

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.run_name}.jsonl"
    summary_path = output_dir / f"{args.run_name}_summary.csv"
    write_records(result_path, records, original_ids)
    summary_rows = summarize(records)
    write_summary(summary_path, summary_rows)

    overall = summary_rows[0]
    print(f"Overall: {overall['correct']}/{overall['n']} = {overall['acc']:.3f}")
    for row in summary_rows[1:]:
        print(
            f"{row['group']}: {row['correct']}/{row['n']} = {row['acc']:.3f} "
            f"(fallback {row['fallback_used']})"
        )
    print(f"Results written to {result_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
