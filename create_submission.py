#!/usr/bin/env python3
"""Generate a private-set submission CSV.

The CSV has exactly the competition-required columns:

    id,response

When self-consistency is enabled, `response` is the chosen raw model output
after majority vote. A JSONL audit file with all samples is also written next
to the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_DATA_PATH = "data/private.jsonl"
DEFAULT_OUTPUT_DIR = "results/private_submission"
DEFAULT_MCQ_PROMPT = "match_back_few_shot"
DEFAULT_FREE_PROMPT = "multi_answer"


@dataclass(frozen=True)
class SamplingConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int
    n: int
    vote: bool = False


CONFIGS = {
    "greedy_n1": SamplingConfig("greedy_n1", temperature=0.0, top_p=1.0, top_k=-1, n=1),
    "sc_n1": SamplingConfig("sc_n1", temperature=0.7, top_p=0.95, top_k=20, n=1),
    "sc_n3": SamplingConfig("sc_n3", temperature=0.7, top_p=0.95, top_k=20, n=3, vote=True),
}


def load_private(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    from prompt_sweep import normalize_item

    rows = [normalize_item(json.loads(line)) for line in path.open()]
    mcq = [row for row in rows if row.get("options")]
    free = [row for row in rows if not row.get("options")]
    return mcq, free, [int(row["id"]) for row in rows]


def render_prompts(tokenizer: Any, qtype: str, prompt_name: str, items: list[dict[str, Any]]) -> list[str]:
    from prompt_variants import build_free_prompt, build_mcq_prompt

    prompts = []
    for item in items:
        if qtype == "mcq":
            system, user = build_mcq_prompt(prompt_name, item["question"], item["options"])
        else:
            system, user = build_free_prompt(prompt_name, item["question"])
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def vote_key(response: str, item: dict[str, Any], qtype: str, judger: Any) -> str:
    from prompt_sweep import extract_letter

    if qtype == "mcq":
        return extract_letter(response, item.get("options"), judger)
    try:
        answer = judger.extract_ans(response)
        return judger.norm_ans_str(answer) if answer else ""
    except Exception:
        return ""


def choose_majority(
    responses: list[str],
    item: dict[str, Any],
    qtype: str,
    judger: Any,
) -> tuple[str, int, str]:
    keys = [vote_key(response, item, qtype, judger) for response in responses]
    counts = Counter(key for key in keys if key)
    if not counts:
        return responses[0], 0, ""
    winning = counts.most_common(1)[0][0]
    for idx, (key, response) in enumerate(zip(keys, responses)):
        if key == winning:
            return response, idx, winning
    return responses[0], 0, ""


def generate_group(
    llm: Any,
    tokenizer: Any,
    judger: Any,
    qtype: str,
    prompt_name: str,
    config: SamplingConfig,
    items: list[dict[str, Any]],
    max_tokens: int,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    if not items:
        return []

    prompts = render_prompts(tokenizer, qtype, prompt_name, items)
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        n=config.n,
    )

    print(f"[{qtype} / {prompt_name} / {config.name}] generating {len(items)} prompts x n={config.n} ...")
    outputs = llm.generate(prompts, sampling_params=sampling)

    records = []
    for item, output in zip(items, outputs):
        responses = [choice.text.strip() for choice in output.outputs]
        finishes = [getattr(choice, "finish_reason", None) for choice in output.outputs]
        if config.vote and len(responses) > 1:
            chosen, chosen_idx, winning_key = choose_majority(responses, item, qtype, judger)
        else:
            chosen, chosen_idx, winning_key = responses[0], 0, vote_key(responses[0], item, qtype, judger)
        records.append(
            {
                "id": int(item["id"]),
                "qtype": qtype,
                "response": chosen,
                "all_samples": responses if config.n > 1 else None,
                "finish_reasons": finishes,
                "chosen_idx": chosen_idx,
                "vote_key": winning_key,
            }
        )
    return records


def write_outputs(
    output_dir: Path,
    submission_path: Path,
    records: list[dict[str, Any]],
    original_ids: list[int],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {record["id"]: record for record in records}
    missing = [row_id for row_id in original_ids if row_id not in by_id]
    if missing:
        raise RuntimeError(f"Missing predictions for ids: {missing[:10]} (count={len(missing)})")

    with submission_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "response"])
        writer.writeheader()
        for row_id in original_ids:
            writer.writerow({"id": row_id, "response": by_id[row_id]["response"]})

    audit_path = submission_path.with_suffix(".jsonl")
    with audit_path.open("w", encoding="utf-8") as file:
        for row_id in original_ids:
            file.write(json.dumps(by_id[row_id], ensure_ascii=False) + "\n")

    return submission_path, audit_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--submission-name", default="submission_sc_n3.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--mcq-prompt", default=DEFAULT_MCQ_PROMPT)
    parser.add_argument("--free-prompt", default=DEFAULT_FREE_PROMPT)
    parser.add_argument("--mcq-config", default="sc_n3", choices=sorted(CONFIGS))
    parser.add_argument("--free-config", default="sc_n3", choices=sorted(CONFIGS))
    parser.add_argument("--max-tokens", type=int, default=16384)
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

    from judger import Judger
    from transformers import AutoTokenizer
    from vllm import LLM

    output_dir = Path(args.output_dir)
    submission_path = output_dir / args.submission_name
    mcq_config = CONFIGS[args.mcq_config]
    free_config = CONFIGS[args.free_config]
    mcq_items, free_items, original_ids = load_private(Path(args.data_path))

    print(f"Private set: {len(mcq_items)} MCQ + {len(free_items)} free-form = {len(original_ids)} total")
    print(f"MCQ : {args.mcq_prompt} / {mcq_config.name}")
    print(f"FREE: {args.free_prompt} / {free_config.name}")

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
    judger = Judger(strict_extract=False)

    records = []
    records.extend(
        generate_group(
            llm,
            tokenizer,
            judger,
            "mcq",
            args.mcq_prompt,
            mcq_config,
            mcq_items,
            args.max_tokens,
        )
    )
    records.extend(
        generate_group(
            llm,
            tokenizer,
            judger,
            "free",
            args.free_prompt,
            free_config,
            free_items,
            args.max_tokens,
        )
    )

    csv_path, audit_path = write_outputs(output_dir, submission_path, records, original_ids)
    print(f"Submission written to {csv_path}")
    print(f"Audit JSONL written to {audit_path}")


if __name__ == "__main__":
    main()
