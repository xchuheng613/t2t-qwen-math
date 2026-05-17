#!/usr/bin/env python3
"""Build compact-prompt SFT JSONL splits from public.jsonl.

The output uses a conversational ``messages`` field compatible with TRL's
SFTTrainer. Targets are final-answer blocks only, matching the compact
submission format.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompts.compact_prompt_pack import build_routed_prompt, rebuild_final_response


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def answer_values(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer")
    values = answer if isinstance(answer, list) else [answer]
    return [str(value).strip() for value in values]


def is_free_response(row: dict[str, Any]) -> bool:
    return not bool(row.get("options"))


def is_high_confidence(row: dict[str, Any]) -> bool:
    """Conservative label-format filter for the first SFT run."""
    answers = answer_values(row)
    question = str(row.get("question", ""))
    if not answers or any(answer == "" for answer in answers):
        return False
    if row.get("options"):
        return len(answers) == 1 and bool(re.fullmatch(r"[A-Z]+", answers[0]))

    expected = question.count("[ANS]")
    if expected > 0 and len(answers) != expected:
        return False
    if any(len(answer) > 240 or "\n" in answer for answer in answers):
        return False
    return True


def stratum(row: dict[str, Any]) -> tuple[str, int, str]:
    question = str(row.get("question", ""))
    n_ans = question.count("[ANS]")
    table_like = n_ans >= 6 or "table" in question.lower() or "\\begin{array}" in question
    return (
        "mcq" if row.get("options") else "free",
        min(n_ans, 6),
        "table" if table_like else "normal",
    )


def split_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)

    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for group_rows in groups.values():
        rng.shuffle(group_rows)
        n = len(group_rows)
        n_train = int(0.70 * n)
        n_dev = int(0.15 * n)
        train.extend(group_rows[:n_train])
        dev.extend(group_rows[n_train : n_train + n_dev])
        holdout.extend(group_rows[n_train + n_dev :])

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(holdout)
    return train, dev, holdout


def make_example(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row["question"])
    options = row.get("options") or None
    system, user = build_routed_prompt("compact", question, options)
    assistant = rebuild_final_response(answer_values(row), question)

    return {
        "id": int(row["id"]),
        "format": "mcq" if options else "free",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-path", type=Path, default=Path("data/public.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/sft_free_v1"))
    parser.add_argument("--free-only", action="store_true")
    parser.add_argument("--high-confidence-only", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.public_path)
    if args.free_only:
        rows = [row for row in rows if is_free_response(row)]
    if args.high_confidence_only:
        rows = [row for row in rows if is_high_confidence(row)]

    train, dev, holdout = split_rows(rows, seed=args.seed)
    write_jsonl(args.out_dir / "train.jsonl", [make_example(row) for row in train])
    write_jsonl(args.out_dir / "dev.jsonl", [make_example(row) for row in dev])
    write_jsonl(args.out_dir / "holdout.jsonl", [make_example(row) for row in holdout])

    print(f"source={args.public_path} out={args.out_dir}")
    print(f"train={len(train)} dev={len(dev)} holdout={len(holdout)}")


if __name__ == "__main__":
    main()
