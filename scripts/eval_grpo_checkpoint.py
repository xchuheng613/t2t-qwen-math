#!/usr/bin/env python3
"""Generate and score free-response rows from a GRPO/full-model checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from judger import Judger
from prompts.compact_prompt_pack import build_routed_prompt, rebuild_final_response


DEFAULT_MODEL = "checkpoints/full_grpo_free/qwen3_4b"
DEFAULT_TOKENIZER = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_DATA_FILE = "data/public_dev.jsonl"
DEFAULT_OUTPUT_DIR = "results/full_grpo_free_eval"


@dataclass(frozen=True)
class EvalSummary:
    total: int
    correct: int
    accuracy: float
    no_answer: int
    count_mismatch: int
    avg_response_chars: float


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def answer_values(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer")
    values = answer if isinstance(answer, list) else [answer]
    return ["" if value is None else str(value).strip() for value in values]


def make_prompts(tokenizer: Any, rows: list[dict[str, Any]]) -> list[str]:
    prompts: list[str] = []
    for row in rows:
        system, user = build_routed_prompt("compact", str(row["question"]), row.get("options") or None)
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def selected_text(output: Any) -> tuple[str, str]:
    item = output.outputs[0]
    return item.text, str(getattr(item, "finish_reason", ""))


def postprocess_response(judger: Judger, row: dict[str, Any], response: str) -> tuple[str, bool]:
    question = str(row.get("question", ""))
    expected = max(1, question.count("[ANS]"))
    try:
        answers = judger.extract_all_boxed(response) or []
    except Exception:
        answers = []
    if len(answers) == 1 and expected > 1:
        try:
            split_answers = judger.split_by_comma(answers[0])
        except Exception:
            split_answers = answers
        if len(split_answers) == expected:
            answers = split_answers
    if not answers:
        try:
            extracted = judger.extract_ans(response)
        except Exception:
            extracted = ""
        if extracted:
            try:
                parts = judger.split_by_comma(extracted)
            except Exception:
                parts = [extracted]
            answers = parts
    if not answers:
        return response, False
    if expected > 1 and len(answers) != expected:
        return response, False
    if expected == 1 and len(answers) != 1:
        return response, False
    return rebuild_final_response(answers, question), True


def score_response(judger: Judger, row: dict[str, Any], response: str) -> bool:
    gold = answer_values(row)
    try:
        return bool(judger.auto_judge(pred=response, gold=gold, options=[[]] * len(gold)))
    except Exception:
        return False


def score_records(judger: Judger, rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> EvalSummary:
    correct = 0
    no_answer = 0
    count_mismatch = 0
    lengths: list[int] = []

    for row, record in zip(rows, records):
        response = str(record["response"])
        lengths.append(len(response))
        correct += int(score_response(judger, row, response))
        try:
            extracted = judger.extract_ans(response) or ""
        except Exception:
            extracted = ""
        if not extracted:
            no_answer += 1
            continue
        try:
            parts = judger.split_by_comma(extracted)
        except Exception:
            parts = [extracted]
        if len(parts) != len(answer_values(row)):
            count_mismatch += 1

    total = len(records)
    return EvalSummary(
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        no_answer=no_answer,
        count_mismatch=count_mismatch,
        avg_response_chars=mean(lengths) if lengths else 0.0,
    )


def write_outputs(output_dir: Path, records: list[dict[str, Any]], summary: EvalSummary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "submission.jsonl"
    csv_path = output_dir / "submission.csv"
    summary_path = output_dir / "score_summary.csv"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "response"])
        writer.writeheader()
        for record in records:
            writer.writerow({"id": record["id"], "response": record["response"]})

    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.__dict__))
        writer.writeheader()
        writer.writerow(summary.__dict__)

    print(f"wrote {jsonl_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "Missing eval dependency. Install project requirements on the cloud machine:\n\n"
            "  pip install -r requirements.txt\n"
        ) from exc

    rows = load_rows(Path(args.data_file), limit=args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    prompts = make_prompts(tokenizer, rows)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompts, sampling)
    judger = Judger(strict_extract=False)

    records: list[dict[str, Any]] = []
    for row, output in zip(rows, outputs):
        raw, finish = selected_text(output)
        response = raw.strip()
        postprocess_used = False
        if args.postprocess:
            response, postprocess_used = postprocess_response(judger, row, response)
        records.append(
            {
                "id": int(row["id"]),
                "response": response,
                "raw_response": raw,
                "finish_reason": finish,
                "postprocess_used": postprocess_used,
            }
        )

    summary = score_records(judger, rows, records)
    print(f"accuracy={summary.correct}/{summary.total} = {summary.accuracy:.4f}")
    print(f"no_answer={summary.no_answer} count_mismatch={summary.count_mismatch}")
    print(f"avg_response_chars={summary.avg_response_chars:.1f}")
    write_outputs(Path(args.output_dir), records, summary)


if __name__ == "__main__":
    main()
