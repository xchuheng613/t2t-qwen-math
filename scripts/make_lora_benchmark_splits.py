#!/usr/bin/env python3
"""Create fixed public splits for LoRA training and benchmarking.

This script reserves stratified public dev/holdout sets first, then writes a
free-response KEEP-only training source from the remaining rows. That prevents
LoRA training leakage into the benchmark used for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OPTION_MARKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-J])[\.\)]\s+")
FORMULA_TOKENS = ("=", "*", "^", "/", "sqrt", "\\sqrt", "pi", "\\pi", "ln", "log", "exp")
STATS_WORDS = (
    "confidence interval",
    "standard deviation",
    "variance",
    "probability",
    "percent",
    "percentage",
    "round",
    "nearest",
    "decimal places",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_quality(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return {int(row["id"]): row for row in csv.DictReader(file)}


def answer_values(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer")
    values = answer if isinstance(answer, list) else [answer]
    return ["" if value is None else str(value).strip() for value in values]


def has_inline_options(question: str) -> bool:
    markers = [match.group(1) for match in OPTION_MARKER_RE.finditer(question)]
    return "A" in markers and "B" in markers


def is_all_letter_gold(row: dict[str, Any]) -> bool:
    values = answer_values(row)
    return bool(values) and all(re.fullmatch(r"[A-J]+", value.upper()) for value in values)


def inferred_format(row: dict[str, Any]) -> str:
    if row.get("options"):
        return "mcq_options"
    if is_all_letter_gold(row) and has_inline_options(str(row.get("question", ""))):
        return "mcq_inline"
    return "free"


def answer_style(row: dict[str, Any]) -> str:
    question = str(row.get("question", "")).lower()
    answers = answer_values(row)
    joined = " ".join(answers).lower()
    if any(word in question for word in STATS_WORDS):
        return "stats_precision"
    if any(token in joined for token in FORMULA_TOKENS):
        return "formula"
    if any(re.search(r"[a-zA-Z]", answer) for answer in answers):
        return "text_symbolic"
    return "numeric"


def blank_bucket(row: dict[str, Any]) -> str:
    question = str(row.get("question", ""))
    count = question.count("[ANS]")
    table_like = count >= 6 or "table" in question.lower() or "\\begin{array}" in question
    if count == 0:
        return "0_blank"
    if count == 1:
        return "1_blank"
    if count <= 5:
        return "2_5_blank"
    if table_like:
        return "6plus_table"
    return "6plus"


def stratum(row: dict[str, Any], quality: dict[int, dict[str, str]]) -> tuple[str, ...]:
    row_id = int(row["id"])
    fmt = quality.get(row_id, {}).get("format") or inferred_format(row)
    if fmt.startswith("mcq"):
        return ("mcq", fmt)
    return ("free", blank_bucket(row), answer_style(row))


def allocate_stratified(
    rows: list[dict[str, Any]],
    size: int,
    quality: dict[int, dict[str, str]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    if size > len(rows):
        raise ValueError(f"Requested split size {size} from only {len(rows)} eligible rows")

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row, quality)].append(row)

    for group_rows in groups.values():
        rng.shuffle(group_rows)

    total = len(rows)
    quota_items: list[tuple[float, tuple[str, ...], int]] = []
    selected_counts: dict[tuple[str, ...], int] = {}
    for key, group_rows in groups.items():
        raw = size * len(group_rows) / total
        base = int(raw)
        selected_counts[key] = min(base, len(group_rows))
        quota_items.append((raw - base, key, len(group_rows)))

    remaining = size - sum(selected_counts.values())
    for _frac, key, group_size in sorted(quota_items, reverse=True):
        if remaining <= 0:
            break
        if selected_counts[key] < group_size:
            selected_counts[key] += 1
            remaining -= 1

    selected: list[dict[str, Any]] = []
    for key, count in selected_counts.items():
        selected.extend(groups[key][:count])
    rng.shuffle(selected)
    return selected


def summarize(name: str, rows: list[dict[str, Any]], quality: dict[int, dict[str, str]]) -> None:
    statuses = Counter(quality.get(int(row["id"]), {}).get("status", "UNKNOWN") for row in rows)
    formats = Counter((quality.get(int(row["id"]), {}).get("format") or inferred_format(row)) for row in rows)
    strata = Counter(stratum(row, quality) for row in rows)
    print(f"{name}: n={len(rows)} status={dict(sorted(statuses.items()))} format={dict(sorted(formats.items()))}")
    print("  top_strata=" + ", ".join(f"{'/'.join(key)}={count}" for key, count in strata.most_common(8)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-path", type=Path, default=Path("data/public.jsonl"))
    parser.add_argument("--quality-csv", type=Path, default=Path("data/public_quality.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/lora_public_v1"))
    parser.add_argument("--dev-size", type=int, default=180)
    parser.add_argument("--holdout-size", type=int, default=180)
    parser.add_argument(
        "--eval-status",
        action="append",
        default=["KEEP"],
        help="Quality status eligible for dev/holdout. Repeat to include REVIEW.",
    )
    parser.add_argument("--seed", type=int, default=151)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.public_path)
    quality = load_quality(args.quality_csv)
    eval_statuses = set(args.eval_status)
    rng = random.Random(args.seed)

    eligible = [
        row
        for row in rows
        if quality.get(int(row["id"]), {}).get("status") in eval_statuses
    ]
    dev = allocate_stratified(eligible, args.dev_size, quality, rng)
    dev_ids = {int(row["id"]) for row in dev}

    remaining_eligible = [row for row in eligible if int(row["id"]) not in dev_ids]
    holdout = allocate_stratified(remaining_eligible, args.holdout_size, quality, rng)
    heldout_ids = dev_ids | {int(row["id"]) for row in holdout}

    train_pool = [row for row in rows if int(row["id"]) not in heldout_ids]
    train_free_keep = [
        row
        for row in train_pool
        if quality.get(int(row["id"]), {}).get("status") == "KEEP"
        and (quality.get(int(row["id"]), {}).get("format") or inferred_format(row)) == "free"
    ]
    dev_free = [
        row
        for row in dev
        if (quality.get(int(row["id"]), {}).get("format") or inferred_format(row)) == "free"
    ]
    holdout_free = [
        row
        for row in holdout
        if (quality.get(int(row["id"]), {}).get("format") or inferred_format(row)) == "free"
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "public_train_pool.jsonl", train_pool)
    write_jsonl(args.out_dir / "public_dev.jsonl", dev)
    write_jsonl(args.out_dir / "public_holdout.jsonl", holdout)
    write_jsonl(args.out_dir / "train_free_keep.jsonl", train_free_keep)
    write_jsonl(args.out_dir / "dev_free.jsonl", dev_free)
    write_jsonl(args.out_dir / "holdout_free.jsonl", holdout_free)

    print(f"source={args.public_path} quality={args.quality_csv} out={args.out_dir}")
    summarize("public_dev", dev, quality)
    summarize("public_holdout", holdout, quality)
    summarize("train_pool", train_pool, quality)
    summarize("train_free_keep", train_free_keep, quality)
    print(f"dev_free={len(dev_free)} holdout_free={len(holdout_free)}")
    print("wrote:")
    for name in [
        "public_train_pool.jsonl",
        "public_dev.jsonl",
        "public_holdout.jsonl",
        "train_free_keep.jsonl",
        "dev_free.jsonl",
        "holdout_free.jsonl",
    ]:
        print(f"  {args.out_dir / name}")


if __name__ == "__main__":
    main()
