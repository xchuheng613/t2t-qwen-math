#!/usr/bin/env python3
"""Build a mixed Hugging Face math SFT dataset with reasoning traces.

Default output is 100k rows:
- 60k AI-MO/NuminaMath-CoT
- 12k EleutherAI/hendrycks_math
- 18k allenai/math_qa
- 10k microsoft/orca-math-word-problems-200k

Rows are written in the local chat ``messages`` JSONL format used by
scripts/train_lora_sft.py. Source metadata is preserved in each row.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download


SYSTEM_PROMPT = """Solve the math problem. Keep the reasoning clear and concise.

Final answer rules:
- End with exactly one final \\boxed{} answer when the answer is known.
- For multiple-choice problems, include the option letter in the final box.
- Do not put explanations inside the final box.

Output format:
<think>
brief reasoning
</think>

FINAL_ANSWERS:
\\boxed{answer}"""


DEFAULT_COUNTS = {
    "numina": 60000,
    "hendrycks_math": 12000,
    "math_qa": 18000,
    "orca": 10000,
}


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_parquet_rows(repo_id: str, files: Iterable[str]) -> Iterable[dict[str, Any]]:
    for filename in files:
        path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))
        parquet = pq.ParquetFile(path)
        for group_idx in range(parquet.num_row_groups):
            table = parquet.read_row_group(group_idx)
            df = table.to_pandas()
            for row in df.to_dict(orient="records"):
                yield row


def list_hf_files(repo_id: str, suffix: str = ".parquet") -> list[str]:
    info = HfApi().dataset_info(repo_id, files_metadata=False)
    return sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(suffix))


_MCQ_MARKER_RE = re.compile(
    r"(?i)\boptions\s*:|(?<![A-Za-z0-9])A[\.)]\s+.*(?<![A-Za-z0-9])B[\.)]\s+|\(\s*\)\s*A:",
    flags=re.S,
)


def is_mcq_like_text(text: str) -> bool:
    return bool(_MCQ_MARKER_RE.search(text))


def sample_rows(
    rows: Iterable[dict[str, Any]],
    count: int,
    seed: int,
    keep: Any | None = None,
) -> list[dict[str, Any]]:
    """Reservoir-sample count rows without assuming the iterable fits in memory."""
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for row in rows:
        if keep is not None and not keep(row):
            continue
        if seen < count:
            reservoir.append(row)
        else:
            replace_idx = rng.randint(0, seen)
            if replace_idx < count:
                reservoir[replace_idx] = row
        seen += 1
    if len(reservoir) < count:
        raise ValueError(f"requested {count} rows but only found {len(reservoir)}")
    rng.shuffle(reservoir)
    return reservoir


def extract_last_boxed(text: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    out: list[str] = []
    for char in text[start:]:
        if char == "{":
            depth += 1
            out.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(char)
        else:
            out.append(char)
    answer = "".join(out).strip()
    return answer or None


def extract_gsm_final(text: str) -> str | None:
    match = re.search(r"####\s*(.+?)\s*$", text, flags=re.S)
    return match.group(1).strip() if match else None


def extract_orca_final(text: str) -> str | None:
    patterns = [
        r"Therefore,?\s+.*?(?:is|=)\s*:?\s*([$\\\w.,/\-+() ]+)\.?\s*$",
        r"So,?\s+.*?(?:is|=)\s*:?\s*([$\\\w.,/\-+() ]+)\.?\s*$",
        r"answer is\s*:?\s*([$\\\w.,/\-+() ]+)\.?\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            answer = re.sub(r"\s+", " ", match.group(1)).strip()
            if 0 < len(answer) <= 80:
                return answer
    return None


def build_assistant(solution: str, final_answer: str | None) -> str:
    solution = clean_text(solution)
    if final_answer:
        return f"<think>\n{solution}\n</think>\n\nFINAL_ANSWERS:\n\\boxed{{{final_answer}}}"
    return solution


def make_message_row(
    row_id: str,
    source: str,
    user: str,
    assistant_solution: str,
    final_answer: str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "source": source,
        "metadata": metadata or {},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": clean_text(user)},
            {"role": "assistant", "content": build_assistant(assistant_solution, final_answer)},
        ],
    }


def build_numina(count: int, seed: int, free_only: bool = False) -> list[dict[str, Any]]:
    repo = "AI-MO/NuminaMath-CoT"
    files = [f for f in list_hf_files(repo) if f.startswith("data/train-")]
    keep = None
    if free_only:
        keep = lambda row: not is_mcq_like_text(clean_text(row.get("problem")))
    sampled = sample_rows(iter_parquet_rows(repo, files), count, seed, keep=keep)
    rows = []
    for i, row in enumerate(sampled):
        problem = clean_text(row.get("problem"))
        solution = clean_text(row.get("solution"))
        if not problem or not solution:
            continue
        final = extract_last_boxed(solution)
        rows.append(
            make_message_row(
                f"numina_{i}",
                "AI-MO/NuminaMath-CoT",
                problem,
                solution,
                final,
                {"source": row.get("source")},
            )
        )
    return rows[:count]


def build_hendrycks(count: int, seed: int, free_only: bool = False) -> list[dict[str, Any]]:
    repo = "EleutherAI/hendrycks_math"
    files = [f for f in list_hf_files(repo) if f.endswith(".parquet")]
    keep = None
    if free_only:
        keep = lambda row: not is_mcq_like_text(clean_text(row.get("problem")))
    sampled = sample_rows(iter_parquet_rows(repo, files), count, seed, keep=keep)
    rows = []
    for i, row in enumerate(sampled):
        problem = clean_text(row.get("problem"))
        solution = clean_text(row.get("solution"))
        if not problem or not solution:
            continue
        rows.append(
            make_message_row(
                f"hendrycks_math_{i}",
                "EleutherAI/hendrycks_math",
                problem,
                solution,
                extract_last_boxed(solution),
                {"level": row.get("level"), "type": row.get("type")},
            )
        )
    return rows[:count]


def build_math_qa(count: int, seed: int, free_only: bool = False) -> list[dict[str, Any]]:
    if free_only:
        return []
    dataset = load_dataset("allenai/math_qa", split="train", trust_remote_code=True)
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    rows = []
    for out_i, idx in enumerate(indices[:count]):
        row = dataset[idx]
        problem = clean_text(row.get("Problem"))
        options = clean_text(row.get("options"))
        rationale = clean_text(row.get("Rationale"))
        correct = clean_text(row.get("correct")).upper()
        user = f"{problem}\nOptions: {options}"
        rows.append(
            make_message_row(
                f"math_qa_{out_i}",
                "allenai/math_qa",
                user,
                rationale,
                correct,
                {"category": row.get("category"), "original_index": idx},
            )
        )
    return rows


def build_orca(count: int, seed: int, free_only: bool = False) -> list[dict[str, Any]]:
    repo = "microsoft/orca-math-word-problems-200k"
    files = ["data/train-00000-of-00001.parquet"]
    keep = None
    if free_only:
        keep = lambda row: not is_mcq_like_text(clean_text(row.get("question")))
    sampled = sample_rows(iter_parquet_rows(repo, files), count, seed, keep=keep)
    rows = []
    for i, row in enumerate(sampled):
        question = clean_text(row.get("question"))
        answer = clean_text(row.get("answer"))
        if not question or not answer:
            continue
        rows.append(
            make_message_row(
                f"orca_{i}",
                "microsoft/orca-math-word-problems-200k",
                question,
                answer,
                extract_last_boxed(answer) or extract_gsm_final(answer) or extract_orca_final(answer),
            )
        )
    return rows[:count]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/hf_mixed_math_100k"))
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--numina", type=int, default=DEFAULT_COUNTS["numina"])
    parser.add_argument("--hendrycks-math", type=int, default=DEFAULT_COUNTS["hendrycks_math"])
    parser.add_argument("--math-qa", type=int, default=DEFAULT_COUNTS["math_qa"])
    parser.add_argument("--orca", type=int, default=DEFAULT_COUNTS["orca"])
    parser.add_argument("--dev-size", type=int, default=0, help="Optional held-out dev rows taken after shuffling.")
    parser.add_argument("--free-only", action="store_true", help="Filter out MCQ-like prompts and skip MathQA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    build_plan = [
        ("numina", args.numina, build_numina),
        ("hendrycks_math", args.hendrycks_math, build_hendrycks),
        ("math_qa", args.math_qa, build_math_qa),
        ("orca", args.orca, build_orca),
    ]
    manifest = {
        "seed": args.seed,
        "requested_counts": {
            "AI-MO/NuminaMath-CoT": args.numina,
            "EleutherAI/hendrycks_math": args.hendrycks_math,
            "allenai/math_qa": args.math_qa,
            "microsoft/orca-math-word-problems-200k": args.orca,
        },
        "free_only": args.free_only,
    }
    for offset, (name, count, builder) in enumerate(build_plan):
        if count <= 0:
            continue
        if args.free_only and name == "math_qa":
            print("[info] skipping math_qa in free-only mode", flush=True)
            continue
        print(f"[info] building {name}: {count}", flush=True)
        source_rows = builder(count, args.seed + offset * 1009, free_only=args.free_only)
        if len(source_rows) != count:
            raise RuntimeError(f"{name}: expected {count}, got {len(source_rows)} after filtering")
        rows.extend(source_rows)

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    dev_rows: list[dict[str, Any]] = []
    if args.dev_size > 0:
        dev_rows = rows[: args.dev_size]
        rows = rows[args.dev_size :]

    write_jsonl(args.out_dir / "train.jsonl", rows)
    if dev_rows:
        write_jsonl(args.out_dir / "dev.jsonl", dev_rows)

    counts = pd.Series([row["source"] for row in rows]).value_counts().to_dict()
    dev_counts = pd.Series([row["source"] for row in dev_rows]).value_counts().to_dict() if dev_rows else {}
    manifest.update(
        {
            "train_rows": len(rows),
            "dev_rows": len(dev_rows),
            "train_source_counts": counts,
            "dev_source_counts": dev_counts,
            "format": "messages JSONL",
        }
    )
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] train={len(rows)} dev={len(dev_rows)} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
