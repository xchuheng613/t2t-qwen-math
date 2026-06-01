#!/usr/bin/env python3
"""Build free-response AutoResearch train/dev/holdout splits.

Inputs are chat-format SFT rows. Outputs include:
- train.jsonl and train_parts/*.jsonl in SFT messages format
- dev_sft.jsonl / holdout_sft.jsonl for trainer sanity eval
- dev_benchmark.jsonl / holdout_benchmark.jsonl for generated scoring
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from judger import Judger
from scripts.verify_public import score_response


BOXED_RE = re.compile(r"FINAL_ANSWERS:\s*\\boxed\{", re.S)
MCQ_RE = re.compile(
    r"(?i)\boptions\s*:|(?<![A-Za-z0-9])A[\.)]\s+.*(?<![A-Za-z0-9])B[\.)]\s+|\(\s*\)\s*A:",
    re.S,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def final_boxed_answer(text: str) -> str | None:
    marker = "FINAL_ANSWERS:"
    idx = text.rfind(marker)
    tail = text[idx + len(marker) :] if idx >= 0 else text
    box_idx = tail.rfind("\\boxed{")
    if box_idx < 0:
        return None
    start = box_idx + len("\\boxed{")
    depth = 1
    chars: list[str] = []
    for ch in tail[start:]:
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                answer = "".join(chars).strip()
                return answer or None
            chars.append(ch)
        else:
            chars.append(ch)
    return None


def reasoning_word_count(text: str) -> int:
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.S)
    reasoning = think_match.group(1) if think_match else text
    if "FINAL_ANSWERS:" in reasoning:
        reasoning = reasoning.split("FINAL_ANSWERS:", 1)[0]
    return len(re.findall(r"\S+", reasoning))


def is_usable_free_row(row: dict[str, Any], max_reasoning_words: int) -> bool:
    messages = row.get("messages") or []
    if len(messages) != 3:
        return False
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        return False
    user = str(messages[1].get("content", ""))
    assistant = str(messages[2].get("content", ""))
    if MCQ_RE.search(user):
        return False
    if not BOXED_RE.search(assistant):
        return False
    answer = final_boxed_answer(assistant)
    if not answer or re.fullmatch(r"[A-J]", answer.strip(), flags=re.I):
        return False
    if reasoning_word_count(assistant) > max_reasoning_words:
        return False
    if not is_judge_stable_answer(answer):
        return False
    return True


def is_judge_stable_answer(answer: str) -> bool:
    """Conservative filter for benchmark rows scored by the local judge."""
    answer = str(answer).strip()
    lowered = answer.lower()
    if any(ch in answer for ch in [",", "，", "$", ":", "：", "=", "%", "∃", "∈", "∉"]):
        return False
    if any(marker in lowered for marker in ["\\text", "\\mathrm", "\\mathbf", "\\mbox", "\\dfrac", "\\tfrac"]):
        return False
    if re.search(r"\b(inches?|feet|meters?|cm|hours?|minutes?|seconds?|dollars?|units?)\b", lowered):
        return False
    if re.search(r"(?<!\\)[A-Za-z]{2,}", answer):
        allowed = {"sqrt", "frac", "sin", "cos", "tan", "log", "ln", "pi", "arcsin", "arccos", "arctan"}
        words = set(re.findall(r"(?<!\\)([A-Za-z]{2,})", answer))
        if not words <= allowed:
            return False
    if re.fullmatch(r"\\?text\{?[A-Ja-j]\}?", answer):
        return False
    return True


def is_oracle_scoreable(row: dict[str, Any], judger: Judger, benchmark_id: int = 0) -> bool:
    messages = row["messages"]
    answer = final_boxed_answer(messages[2]["content"])
    if answer is None:
        return False
    benchmark_row = {
        "id": benchmark_id,
        "question": str(messages[1]["content"]).strip(),
        "answer": [answer],
    }
    response = f"FINAL_ANSWERS:\n\\boxed{{{answer}}}"
    try:
        return score_response(judger, benchmark_row, response)
    except Exception:
        return False


def to_benchmark_row(row: dict[str, Any], new_id: int, split: str) -> dict[str, Any]:
    messages = row["messages"]
    answer = final_boxed_answer(messages[2]["content"])
    if answer is None:
        raise ValueError(f"missing final answer in row {row.get('id')}")
    return {
        "id": new_id,
        "question": str(messages[1]["content"]).strip(),
        "answer": [answer],
        "source": row.get("source"),
        "source_id": row.get("id"),
        "split": split,
    }


def split_train_parts(train_path: Path, parts_dir: Path, part_size: int) -> list[Path]:
    parts_dir.mkdir(parents=True, exist_ok=True)
    for old_part in parts_dir.glob("*.jsonl"):
        old_part.unlink()
    part_paths: list[Path] = []
    part_index = 0
    current_count = 0
    current_file = None
    try:
        with train_path.open(encoding="utf-8") as src:
            for line in src:
                if current_file is None or current_count >= part_size:
                    if current_file is not None:
                        current_file.close()
                    part_path = parts_dir / f"train-{part_index:04d}.jsonl"
                    current_file = part_path.open("w", encoding="utf-8")
                    part_paths.append(part_path)
                    part_index += 1
                    current_count = 0
                current_file.write(line)
                current_count += 1
    finally:
        if current_file is not None:
            current_file.close()
    return part_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/hf_mixed_math_free_100k/train.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/autoresearch_free_v1"))
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--holdout-size", type=int, default=1000)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument(
        "--use-all-train",
        action="store_true",
        help="Use every remaining usable row for training after dev/holdout are removed.",
    )
    parser.add_argument("--max-reasoning-words", type=int, default=700)
    parser.add_argument("--no-oracle-score-filter", action="store_true")
    parser.add_argument("--part-size", type=int, default=25000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    usable = [row for row in rows if is_usable_free_row(row, args.max_reasoning_words)]
    if not args.no_oracle_score_filter:
        judger = Judger(strict_extract=False)
        usable = [row for idx, row in enumerate(usable) if is_oracle_scoreable(row, judger, idx)]
    needed = args.dev_size + args.holdout_size if args.use_all_train else args.train_size + args.dev_size + args.holdout_size
    if len(usable) <= needed:
        raise SystemExit(f"not enough usable rows: {len(usable)} <= {needed}")

    rng = random.Random(args.seed)
    rng.shuffle(usable)

    dev = usable[: args.dev_size]
    holdout = usable[args.dev_size : args.dev_size + args.holdout_size]
    train_pool = usable[args.dev_size + args.holdout_size :]
    train = train_pool if args.use_all_train else train_pool[: args.train_size]
    rng.shuffle(train)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "dev_sft.jsonl", dev)
    write_jsonl(args.out_dir / "holdout_sft.jsonl", holdout)
    write_jsonl(
        args.out_dir / "dev_benchmark.jsonl",
        [to_benchmark_row(row, idx, "dev") for idx, row in enumerate(dev)],
    )
    write_jsonl(
        args.out_dir / "holdout_benchmark.jsonl",
        [to_benchmark_row(row, idx, "holdout") for idx, row in enumerate(holdout)],
    )
    part_paths = split_train_parts(args.out_dir / "train.jsonl", args.out_dir / "train_parts", args.part_size)

    manifest = {
        "seed": args.seed,
        "input": str(args.input),
        "input_rows": len(rows),
        "usable_rows": len(usable),
        "oracle_score_filter": not args.no_oracle_score_filter,
        "max_reasoning_words": args.max_reasoning_words,
        "train_rows": len(train),
        "use_all_train": args.use_all_train,
        "unused_train_pool_rows": len(train_pool) - len(train),
        "dev_rows": len(dev),
        "holdout_rows": len(holdout),
        "train_parts": [str(path.relative_to(args.out_dir)) for path in part_paths],
        "notes": "Free-response only. Dev/holdout are random rows held out from training.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
