#!/usr/bin/env python3
"""Build concise free-response HF math SFT datasets for AutoResearch.

The goal is a less verbose replacement for the previous mixed free-response
dataset. Sources are chosen by qualitative similarity to local public/private
examples and by short, judge-stable reasoning traces:

- AI-MO/NuminaMath-CoT: broad classroom/contest coverage, filtered to concise rows.
- EleutherAI/hendrycks_math: compact contest algebra/geometry/number theory.
- openai/gsm8k: concise arithmetic word problems.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from judger import Judger
from scripts.verify_public import score_response


SYSTEM_PROMPT = """Solve the math problem. Use only the steps needed.
Once the answer is known, stop reasoning and write FINAL_ANSWERS.
Do not re-check repeatedly.

Final answer rules:
- End with exactly one final \\boxed{} answer.
- Do not put labels, units, or explanation inside the box.

Output format:
<think>
concise reasoning
</think>

FINAL_ANSWERS:
\\boxed{answer}"""

SOURCE_COUNTS_FULL = {
    "numina": 33500,
    "hendrycks_math": 8000,
    "gsm8k": 8500,
}


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", clean_text(text)))


def list_hf_files(repo_id: str, suffix: str = ".parquet") -> list[str]:
    info = HfApi().dataset_info(repo_id, files_metadata=False)
    return sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(suffix))


def iter_parquet_rows(repo_id: str, filenames: Iterable[str]) -> Iterable[tuple[str, int, dict[str, Any]]]:
    for filename in filenames:
        path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))
        parquet = pq.ParquetFile(path)
        offset = 0
        for group_idx in range(parquet.num_row_groups):
            df = parquet.read_row_group(group_idx).to_pandas()
            for row_idx, row in enumerate(df.to_dict(orient="records")):
                yield filename, offset + row_idx, row
            offset += len(df)


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


def remove_box_commands(text: str) -> str:
    """Replace boxed expressions in reasoning with their contents."""
    result: list[str] = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx < 0:
            result.append(text[i:])
            break
        result.append(text[i:idx])
        start = idx + len("\\boxed{")
        depth = 1
        j = start
        chars: list[str] = []
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth:
                    chars.append(ch)
            else:
                chars.append(ch)
            j += 1
        result.append("".join(chars))
        i = j
    return "".join(result)


def reasoning_from_solution(solution: str, final_answer: str, source: str) -> str:
    solution = clean_text(solution)
    if source == "gsm8k":
        solution = re.sub(r"\s*####\s*.+?\s*$", "", solution, flags=re.S)
    solution = remove_box_commands(solution)
    solution = re.sub(r"\s+", " ", solution).strip()
    final_answer = re.escape(str(final_answer).strip())
    solution = re.sub(rf"(?:therefore|thus|so)[^.\n]*{final_answer}[^.\n]*\.?\s*$", "", solution, flags=re.I)
    return solution.strip()


def is_stable_answer(answer: str) -> bool:
    answer = clean_text(answer)
    lowered = answer.lower()
    if not answer or re.fullmatch(r"[A-Ja-j]", answer):
        return False
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
    return True


def is_oracle_scoreable(judger: Judger, question: str, answer: str) -> bool:
    response = f"FINAL_ANSWERS:\n\\boxed{{{answer}}}"
    try:
        return score_response(judger, {"id": 0, "question": question, "answer": [answer]}, response)
    except Exception:
        return False


def make_row(row_id: str, source: str, question: str, reasoning: str, answer: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row_id,
        "source": source,
        "metadata": metadata,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": clean_text(question)},
            {
                "role": "assistant",
                "content": f"<think>\n{clean_text(reasoning)}\n</think>\n\nFINAL_ANSWERS:\n\\boxed{{{clean_text(answer)}}}",
            },
        ],
    }


def load_eval_question_texts(paths: list[Path]) -> set[str]:
    texts: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "question" in row:
                    texts.add(clean_text(row["question"]))
                elif row.get("messages"):
                    texts.add(clean_text(row["messages"][1]["content"]))
    return texts


def candidate_rows(source: str, exclude_questions: set[str], max_reasoning_words: int, seed: int) -> Iterable[dict[str, Any]]:
    judger = Judger(strict_extract=False)
    if source == "gsm8k":
        repo = "openai/gsm8k"
        files = ["main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet"]
        random.Random(seed).shuffle(files)
        for filename, row_idx, row in iter_parquet_rows(repo, files):
            question = clean_text(row.get("question", ""))
            solution = clean_text(row.get("answer", ""))
            answer = extract_gsm_final(solution)
            if not question or question in exclude_questions or not answer:
                continue
            reasoning = reasoning_from_solution(solution, answer, source)
            if word_count(reasoning) > max_reasoning_words or not is_stable_answer(answer):
                continue
            if not is_oracle_scoreable(judger, question, answer):
                continue
            yield make_row(
                f"gsm8k::{filename}::{row_idx}",
                "openai/gsm8k",
                question,
                reasoning,
                answer,
                {"repo": repo, "file": filename, "row_idx": row_idx},
            )
    elif source == "hendrycks_math":
        repo = "EleutherAI/hendrycks_math"
        files = list_hf_files(repo)
        random.Random(seed).shuffle(files)
        for filename, row_idx, row in iter_parquet_rows(repo, files):
            question = clean_text(row.get("problem", ""))
            solution = clean_text(row.get("solution", ""))
            answer = extract_last_boxed(solution)
            if not question or question in exclude_questions or not answer:
                continue
            reasoning = reasoning_from_solution(solution, answer, source)
            if word_count(reasoning) > max_reasoning_words or not is_stable_answer(answer):
                continue
            if not is_oracle_scoreable(judger, question, answer):
                continue
            yield make_row(
                f"hendrycks_math::{filename}::{row_idx}",
                repo,
                question,
                reasoning,
                answer,
                {"repo": repo, "file": filename, "row_idx": row_idx, "level": row.get("level"), "type": row.get("type")},
            )
    elif source == "numina":
        repo = "AI-MO/NuminaMath-CoT"
        files = [f for f in list_hf_files(repo) if f.startswith("data/train")]
        random.Random(seed).shuffle(files)
        for filename, row_idx, row in iter_parquet_rows(repo, files):
            question = clean_text(row.get("problem", ""))
            solution = clean_text(row.get("solution", ""))
            answer = extract_last_boxed(solution)
            if not question or question in exclude_questions or not answer:
                continue
            reasoning = reasoning_from_solution(solution, answer, source)
            if word_count(reasoning) > max_reasoning_words or not is_stable_answer(answer):
                continue
            if not is_oracle_scoreable(judger, question, answer):
                continue
            yield make_row(
                f"numina::{filename}::{row_idx}",
                repo,
                question,
                reasoning,
                answer,
                {"repo": repo, "file": filename, "row_idx": row_idx, "source": row.get("source")},
            )
    else:
        raise ValueError(f"Unknown source: {source}")


def collect_rows(rows: Iterable[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for row in rows:
        sample.append(row)
        if len(sample) >= count:
            break
    if len(sample) < count:
        raise RuntimeError(f"Requested {count} rows but only found {len(sample)}")
    rng.shuffle(sample)
    return sample


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_parts(train_path: Path, part_size: int = 25000) -> list[str]:
    parts_dir = train_path.parent / "train_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for old in parts_dir.glob("*.jsonl"):
        old.unlink()
    part_paths: list[str] = []
    with train_path.open(encoding="utf-8") as src:
        part_idx = 0
        count = 0
        out = None
        try:
            for line in src:
                if out is None or count >= part_size:
                    if out is not None:
                        out.close()
                    path = parts_dir / f"train-{part_idx:04d}.jsonl"
                    out = path.open("w", encoding="utf-8")
                    part_paths.append(str(path.relative_to(train_path.parent)))
                    part_idx += 1
                    count = 0
                out.write(line)
                count += 1
        finally:
            if out is not None:
                out.close()
    return part_paths


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judger = Judger(strict_extract=False)
    bad_schema = 0
    oracle_fail = 0
    max_words = 0
    total_words = 0
    for row in rows:
        messages = row.get("messages") or []
        if len(messages) != 3 or [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            bad_schema += 1
            continue
        assistant = messages[2]["content"]
        answer = extract_last_boxed(assistant)
        reasoning = re.search(r"<think>(.*?)</think>", assistant, flags=re.S)
        wc = word_count(reasoning.group(1) if reasoning else assistant)
        max_words = max(max_words, wc)
        total_words += wc
        if not answer or not is_oracle_scoreable(judger, messages[1]["content"], answer):
            oracle_fail += 1
    return {
        "rows": len(rows),
        "bad_schema": bad_schema,
        "oracle_fail": oracle_fail,
        "avg_reasoning_words": round(total_words / len(rows), 2) if rows else 0,
        "max_reasoning_words": max_words,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/autoresearch_free_concise_v2"))
    parser.add_argument("--seed", type=int, default=529)
    parser.add_argument("--max-reasoning-words", type=int, default=400)
    parser.add_argument("--search-size", type=int, default=10000)
    parser.add_argument("--full-size", type=int, default=50000)
    parser.add_argument("--eval-file", action="append", type=Path, default=[
        Path("data/autoresearch_free_v1/dev_benchmark.jsonl"),
        Path("data/autoresearch_free_v1/holdout_benchmark.jsonl"),
    ])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exclude_questions = load_eval_question_texts(args.eval_file)
    full_counts = SOURCE_COUNTS_FULL.copy()
    if sum(full_counts.values()) != args.full_size:
        raise SystemExit(f"source counts sum to {sum(full_counts.values())}, expected {args.full_size}")

    full_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for offset, (source, count) in enumerate(full_counts.items()):
        print(f"[info] sampling {count} rows from {source}", flush=True)
        rows = collect_rows(
            candidate_rows(source, exclude_questions, args.max_reasoning_words, args.seed + 1009 * offset),
            count,
            args.seed + 1009 * offset,
        )
        source_counts[source] = len(rows)
        full_rows.extend(rows)

    rng = random.Random(args.seed)
    rng.shuffle(full_rows)
    search_rows = rng.sample(full_rows, args.search_size)

    for name, rows in [("full", full_rows), ("search", search_rows)]:
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "train.jsonl", rows)
        part_paths = split_train_parts(out / "train.jsonl")
        audit = audit_rows(rows)
        manifest = {
            "seed": args.seed,
            "source_counts": source_counts if name == "full" else pd.Series([row["source"] for row in rows]).value_counts().to_dict(),
            "max_reasoning_words_filter": args.max_reasoning_words,
            "excluded_eval_files": [str(path) for path in args.eval_file],
            "excluded_eval_question_count": len(exclude_questions),
            "train_rows": len(rows),
            "train_parts": part_paths,
            "audit": audit,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"[done] {name}: {json.dumps(audit)}", flush=True)

    report = {
        "local_sample_seed": 529,
        "decision": (
            "Use concise-filtered AI-MO/NuminaMath-CoT, EleutherAI/hendrycks_math, "
            "and openai/gsm8k. Exclude Orca because final answers are not reliably "
            "extractable from raw rows despite concise reasoning."
        ),
        "full_counts": full_counts,
        "reasoning_cap_words": args.max_reasoning_words,
    }
    (args.out_dir / "hf_search_summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
