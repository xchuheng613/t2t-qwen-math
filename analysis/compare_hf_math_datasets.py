#!/usr/bin/env python3
"""Find Hugging Face math datasets similar to the local public/private tasks.

This intentionally uses lightweight dependencies already present in the repo's
default Python environment: huggingface_hub, pandas, pyarrow, sklearn, numpy.
It does not require a Hugging Face token for public datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_CANDIDATES = [
    "openai/gsm8k",
    "allenai/math_qa",
    "qwedsacf/competition_math",
    "EleutherAI/hendrycks_math",
    "hendrycks/competition_math",
    "microsoft/orca-math-word-problems-200k",
    "AI-MO/NuminaMath-CoT",
    "di-zhang-fdu/DeepMind_Mathematics_QA",
    "LLMcompe-Team-Watanabe/math_AoPS-Instruct_preprocess_fixed",
]

QUESTION_COLUMN_HINTS = (
    "question",
    "problem",
    "query",
    "input",
    "prompt",
    "instruction",
)


@dataclass
class LocalQuestion:
    split: str
    local_id: int | str
    question: str


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.replace("\\frac", " frac ").replace("\\int", " int ")
    return text


def row_to_question(row: dict) -> str:
    text = normalize_text(row.get("question", ""))
    options = row.get("options")
    if options:
        if isinstance(options, list):
            opt_text = " ".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        else:
            opt_text = str(options)
        text = f"{text} Options: {normalize_text(opt_text)}"
    return text


def load_local_sample(path: Path, split: str, n: int, rng: random.Random) -> list[LocalQuestion]:
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            rows.append(LocalQuestion(split, row.get("id"), row_to_question(row)))
    return rng.sample(rows, n)


def question_columns(columns: Iterable[str]) -> list[str]:
    cols = list(columns)
    lowered = {c: c.lower() for c in cols}
    picked = [
        c
        for c in cols
        if any(hint in lowered[c] for hint in QUESTION_COLUMN_HINTS)
        and "answer" not in lowered[c]
        and "solution" not in lowered[c]
    ]
    return picked or cols


def combine_row_text(row: pd.Series, cols: list[str]) -> str:
    pieces = []
    for col in cols:
        value = row.get(col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            value = " ".join(map(str, value))
        pieces.append(str(value))
    return normalize_text(" ".join(pieces))


def load_parquet_texts(path: Path, limit: int) -> list[str]:
    pf = pq.ParquetFile(path)
    cols = question_columns(pf.schema.names)
    texts: list[str] = []
    for group_idx in range(pf.num_row_groups):
        if len(texts) >= limit:
            break
        table = pf.read_row_group(group_idx, columns=cols)
        df = table.to_pandas()
        for _, row in df.iterrows():
            text = combine_row_text(row, cols)
            if len(text) >= 20:
                texts.append(text)
            if len(texts) >= limit:
                break
    return texts


def load_json_texts(path: Path, limit: int) -> list[str]:
    if path.stat().st_size > 250_000_000:
        return []
    texts: list[str] = []
    with path.open() as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            iterator = data
        else:
            iterator = (json.loads(line) for line in f if line.strip())
        for item in iterator:
            if not isinstance(item, dict):
                continue
            cols = question_columns(item.keys())
            text = normalize_text(" ".join(str(item.get(c, "")) for c in cols))
            if len(text) >= 20:
                texts.append(text)
            if len(texts) >= limit:
                break
    return texts


def downloadable_data_files(repo_id: str) -> list[str]:
    info = HfApi().dataset_info(repo_id, files_metadata=False)
    files = [s.rfilename for s in info.siblings]
    preferred = [
        f
        for f in files
        if f.endswith((".parquet", ".jsonl", ".json", ".csv"))
        and not f.lower().endswith("readme.md")
    ]
    return sorted(preferred)


def load_hf_texts(repo_id: str, max_rows: int) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    used_files: list[str] = []
    for filename in downloadable_data_files(repo_id):
        if len(texts) >= max_rows:
            break
        try:
            local = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))
        except Exception as exc:
            print(f"[warn] download failed {repo_id}/{filename}: {exc}")
            continue
        remaining = max_rows - len(texts)
        try:
            if filename.endswith(".parquet"):
                new_texts = load_parquet_texts(local, remaining)
            elif filename.endswith((".json", ".jsonl")):
                new_texts = load_json_texts(local, remaining)
            elif filename.endswith(".csv"):
                df = pd.read_csv(local, nrows=remaining)
                cols = question_columns(df.columns)
                new_texts = [combine_row_text(row, cols) for _, row in df.iterrows()]
            else:
                new_texts = []
        except Exception as exc:
            print(f"[warn] parse failed {repo_id}/{filename}: {exc}")
            continue
        if new_texts:
            texts.extend(new_texts)
            used_files.append(filename)
    return texts[:max_rows], used_files


def load_with_datasets(repo_id: str, max_rows: int) -> tuple[list[str], list[str]]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except Exception as exc:
        raise RuntimeError(f"`datasets` fallback unavailable: {exc}") from exc

    texts: list[str] = []
    used_splits: list[str] = []
    try:
        configs = get_dataset_config_names(repo_id, trust_remote_code=True)
    except Exception:
        configs = [None]
    if not configs:
        configs = [None]

    for config in configs[:20]:
        if len(texts) >= max_rows:
            break
        try:
            dataset = load_dataset(repo_id, config, split=None, trust_remote_code=True)
        except Exception as exc:
            print(f"[warn] datasets load failed {repo_id}/{config}: {exc}")
            continue
        for split_name, split in dataset.items():
            if len(texts) >= max_rows:
                break
            cols = question_columns(split.column_names)
            for row in split:
                text = normalize_text(" ".join(str(row.get(c, "")) for c in cols))
                if len(text) >= 20:
                    texts.append(text)
                if len(texts) >= max_rows:
                    break
            used_splits.append(f"{config or 'default'}:{split_name}")
    return texts[:max_rows], used_splits


def score_dataset(queries: list[LocalQuestion], candidates: list[str], top_k: int) -> tuple[dict, list[dict]]:
    query_texts = [q.question for q in queries]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=True,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(query_texts + candidates)
    sims = cosine_similarity(matrix[: len(queries)], matrix[len(queries) :])
    best_idx = sims.argmax(axis=1)
    best_scores = sims[np.arange(len(queries)), best_idx]

    top_rows = []
    for q_i, q in enumerate(queries):
        idxs = np.argsort(-sims[q_i])[:top_k]
        for rank, idx in enumerate(idxs, start=1):
            top_rows.append(
                {
                    "split": q.split,
                    "local_id": q.local_id,
                    "rank": rank,
                    "similarity": float(sims[q_i, idx]),
                    "query": q.question,
                    "match": candidates[idx],
                }
            )

    public_scores = [s for q, s in zip(queries, best_scores) if q.split == "public"]
    private_scores = [s for q, s in zip(queries, best_scores) if q.split == "private"]
    metrics = {
        "rows_compared": len(candidates),
        "mean_best_similarity": float(np.mean(best_scores)),
        "median_best_similarity": float(np.median(best_scores)),
        "public_mean_best_similarity": float(np.mean(public_scores)),
        "private_mean_best_similarity": float(np.mean(private_scores)),
        "queries_with_match_ge_0.50": int(np.sum(best_scores >= 0.50)),
        "queries_with_match_ge_0.35": int(np.sum(best_scores >= 0.35)),
    }
    return metrics, top_rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, default=Path("data/public.jsonl"))
    parser.add_argument("--private", type=Path, default=Path("data/private.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/hf_dataset_similarity"))
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--max-rows-per-dataset", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate", action="append", default=[])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    queries = load_local_sample(args.public, "public", args.sample_size, rng)
    queries += load_local_sample(args.private, "private", args.sample_size, rng)

    write_jsonl(
        args.out_dir / "sampled_questions.jsonl",
        [
            {"split": q.split, "id": q.local_id, "question": q.question}
            for q in queries
        ],
    )

    repo_ids = args.candidate or DEFAULT_CANDIDATES
    summary_rows = []
    all_top_rows = []
    skipped = []
    for repo_id in repo_ids:
        print(f"[info] loading {repo_id}")
        try:
            texts, used_files = load_hf_texts(repo_id, args.max_rows_per_dataset)
        except Exception as exc:
            print(f"[warn] direct load failed {repo_id}: {exc}")
            try:
                texts, used_files = load_with_datasets(repo_id, args.max_rows_per_dataset)
            except Exception as fallback_exc:
                skipped.append({"repo_id": repo_id, "reason": f"{exc}; fallback: {fallback_exc}"})
                print(f"[warn] skipped {repo_id}: {fallback_exc}")
                continue
        if not texts:
            try:
                texts, used_files = load_with_datasets(repo_id, args.max_rows_per_dataset)
            except Exception as fallback_exc:
                skipped.append({"repo_id": repo_id, "reason": f"no directly readable question text found; fallback: {fallback_exc}"})
                print(f"[warn] skipped {repo_id}: no texts")
                continue
            if not texts:
                skipped.append({"repo_id": repo_id, "reason": "no question text found"})
                print(f"[warn] skipped {repo_id}: no texts")
                continue
        metrics, top_rows = score_dataset(queries, texts, args.top_k)
        metrics.update({"repo_id": repo_id, "files": ";".join(used_files)})
        summary_rows.append(metrics)
        for row in top_rows:
            row["repo_id"] = repo_id
        all_top_rows.extend(top_rows)

    summary_rows.sort(key=lambda r: r["mean_best_similarity"], reverse=True)
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "dataset_similarity_summary.csv", index=False)
    pd.DataFrame(all_top_rows).to_csv(args.out_dir / "top_matches.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_jsonl(args.out_dir / "skipped_datasets.jsonl", skipped)

    lines = ["# Hugging Face Dataset Similarity Report", ""]
    lines.append(f"Seed: `{args.seed}`. Local sample: {args.sample_size} public + {args.sample_size} private questions.")
    lines.append(f"Max HF rows per dataset: `{args.max_rows_per_dataset}`.")
    lines.append("")
    lines.append("| Rank | Dataset | Mean best | Public mean | Private mean | >=0.50 | Rows |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(summary_rows, start=1):
        lines.append(
            f"| {rank} | `{row['repo_id']}` | {row['mean_best_similarity']:.3f} | "
            f"{row['public_mean_best_similarity']:.3f} | {row['private_mean_best_similarity']:.3f} | "
            f"{row['queries_with_match_ge_0.50']} | {row['rows_compared']} |"
        )
    if skipped:
        lines.extend(["", "Skipped datasets:"])
        for row in skipped:
            lines.append(f"- `{row['repo_id']}`: {row['reason']}")
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")

    print(f"[done] wrote {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
