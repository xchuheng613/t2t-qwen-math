#!/usr/bin/env python3
"""Generate a private-set submission CSV.

The CSV has exactly the competition-required columns:

    id,response

By default, rows are routed by answer format first, then generated in prompt
family/config groups. The submitted `response` is a cleaned FINAL_ANSWERS block
derived from the chosen model output. A JSONL audit file with raw samples,
routes, and fallback details is also written next to the CSV.
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
DEFAULT_MCQ_PROMPT = "general_mcq_eliminate"
DEFAULT_FREE_PROMPT = "baseline"


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


def load_private(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    from format_router import normalize_item

    rows = [normalize_item(json.loads(line)) for line in path.open()]
    return rows, [int(row["id"]) for row in rows]


def render_prompts(
    tokenizer: Any,
    prompt_name: str,
    items: list[dict[str, Any]],
    fallback: bool = False,
) -> list[str]:
    from prompt_variants import build_routed_prompt

    prompts = []
    for item in items:
        system, user = build_routed_prompt(
            prompt_name,
            item["question"],
            item.get("options") or None,
            fallback=fallback,
        )
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def vote_key(response: str, item: dict[str, Any], route: Any, judger: Any) -> str:
    from format_router import (
        FORMAT_ALGORITHM_SEQUENCE,
        FORMAT_MCQ,
        FORMAT_MULTI_SELECT,
        extract_choice_prediction,
    )

    if route.format_type in {FORMAT_MCQ, FORMAT_ALGORITHM_SEQUENCE, FORMAT_MULTI_SELECT}:
        return extract_choice_prediction(
            response,
            item,
            multi=route.format_type == FORMAT_MULTI_SELECT,
            judger=None if route.format_type == FORMAT_ALGORITHM_SEQUENCE else judger,
        )
    try:
        answer = judger.extract_ans(response)
        return judger.norm_ans_str(answer) if answer else ""
    except Exception:
        return ""


def choose_majority(
    responses: list[str],
    item: dict[str, Any],
    route: Any,
    judger: Any,
) -> tuple[str, int, str]:
    keys = [vote_key(response, item, route, judger) for response in responses]
    counts = Counter(key for key in keys if key)
    if not counts:
        return responses[0], 0, ""
    winning = counts.most_common(1)[0][0]
    for idx, (key, response) in enumerate(zip(keys, responses)):
        if key == winning:
            return response, idx, winning
    return responses[0], 0, ""


def _fallback_sampling_params(max_tokens: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        n=1,
    )


def generate_group(
    llm: Any,
    tokenizer: Any,
    judger: Any,
    prompt_name: str,
    config: SamplingConfig,
    routed_items: list[tuple[dict[str, Any], Any]],
    max_tokens: int,
    fallback_max_tokens: int,
    enable_fallback: bool,
) -> list[dict[str, Any]]:
    from format_router import clean_final_response
    from vllm import SamplingParams

    if not routed_items:
        return []

    items = [item for item, _route in routed_items]
    routes = [route for _item, route in routed_items]
    prompts = render_prompts(tokenizer, prompt_name, items)
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        n=config.n,
    )

    route_names = Counter(route.format_type for route in routes)
    route_summary = ", ".join(f"{name}={count}" for name, count in sorted(route_names.items()))
    print(
        f"[{prompt_name} / {config.name}] generating {len(items)} prompts x n={config.n} ({route_summary}) ...",
        flush=True,
    )
    outputs = llm.generate(prompts, sampling_params=sampling)
    print(f"[{prompt_name} / {config.name}] generation finished; post-processing ...", flush=True)

    records = []
    fallback_indices: list[int] = []
    for idx, (item, route, output) in enumerate(zip(items, routes, outputs)):
        responses = [choice.text.strip() for choice in output.outputs]
        finishes = [getattr(choice, "finish_reason", None) for choice in output.outputs]
        if config.vote and len(responses) > 1:
            chosen, chosen_idx, winning_key = choose_majority(responses, item, route, judger)
        else:
            chosen, chosen_idx = responses[0], 0
            winning_key = vote_key(responses[0], item, route, judger)

        clean_response = clean_final_response(chosen, item, route, judger)
        finish_reason = finishes[chosen_idx] if chosen_idx < len(finishes) else None
        record = {
            "id": int(item["id"]),
            "format_type": route.format_type,
            "prompt": prompt_name,
            "config": config.name,
            "route": route.to_dict(),
            "response": clean_response or chosen,
            "raw_response": chosen,
            "all_samples": responses if config.n > 1 else None,
            "finish_reasons": finishes,
            "chosen_idx": chosen_idx,
            "vote_key": winning_key,
            "fallback_used": False,
        }
        if enable_fallback and (finish_reason == "length" or not clean_response):
            fallback_indices.append(idx)
        records.append(record)

    if fallback_indices:
        fallback_items = [items[idx] for idx in fallback_indices]
        fallback_routes = [routes[idx] for idx in fallback_indices]
        fallback_prompts = render_prompts(tokenizer, prompt_name, fallback_items, fallback=True)
        print(
            f"  fallback: retrying {len(fallback_items)} truncated/no-answer prompts with short no-reasoning prompt ...",
            flush=True,
        )
        fallback_outputs = llm.generate(
            fallback_prompts,
            sampling_params=_fallback_sampling_params(fallback_max_tokens),
        )
        for record_idx, item, route, output in zip(fallback_indices, fallback_items, fallback_routes, fallback_outputs):
            fallback_raw = output.outputs[0].text.strip()
            fallback_finish = getattr(output.outputs[0], "finish_reason", None)
            fallback_clean = clean_final_response(fallback_raw, item, route, judger)
            record = records[record_idx]
            record["fallback_used"] = True
            record["fallback_raw_response"] = fallback_raw
            record["fallback_finish_reason"] = fallback_finish
            record["raw_response_before_fallback"] = record["raw_response"]
            if fallback_clean:
                record["raw_response"] = fallback_raw
                record["response"] = fallback_clean
                record["vote_key"] = vote_key(fallback_raw, item, route, judger)
    print(f"[{prompt_name} / {config.name}] done.", flush=True)
    return records


def build_format_routed_groups(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[tuple[dict[str, Any], Any]]]:
    from format_router import route_item

    groups: dict[tuple[str, str], list[tuple[dict[str, Any], Any]]] = {}
    for item in rows:
        route = route_item(item)
        groups.setdefault((route.prompt_family, route.config_name), []).append((item, route))
    return groups


def build_legacy_groups(
    rows: list[dict[str, Any]],
    mcq_prompt: str,
    free_prompt: str,
    mcq_config: str,
    free_config: str,
) -> dict[tuple[str, str], list[tuple[dict[str, Any], Any]]]:
    from format_router import FORMAT_FREE_RESPONSE, FORMAT_MCQ, Route

    groups: dict[tuple[str, str], list[tuple[dict[str, Any], Any]]] = {}
    for item in rows:
        if item.get("options"):
            route = Route(
                format_type=FORMAT_MCQ,
                prompt_family=mcq_prompt,
                config_name=mcq_config,
                expected_answers=1,
                has_options=True,
                notes=("legacy",),
            )
        else:
            route = Route(
                format_type=FORMAT_FREE_RESPONSE,
                prompt_family=free_prompt,
                config_name=free_config,
                expected_answers=max(1, str(item.get("question", "")).count("[ANS]")),
                notes=("legacy",),
            )
        groups.setdefault((route.prompt_family, route.config_name), []).append((item, route))
    return groups


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
    parser.add_argument("--submission-name", default="submission_format_router.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--routing-mode", choices=["format", "legacy"], default="format")
    parser.add_argument("--mcq-prompt", default=DEFAULT_MCQ_PROMPT)
    parser.add_argument("--free-prompt", default=DEFAULT_FREE_PROMPT)
    parser.add_argument("--mcq-config", default="greedy_n1", choices=sorted(CONFIGS))
    parser.add_argument("--free-config", default="sc_n3", choices=sorted(CONFIGS))
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

    from judger import Judger
    from transformers import AutoTokenizer
    from vllm import LLM

    output_dir = Path(args.output_dir)
    submission_path = output_dir / args.submission_name
    rows, original_ids = load_private(Path(args.data_path))
    if args.routing_mode == "format":
        groups = build_format_routed_groups(rows)
    else:
        groups = build_legacy_groups(
            rows,
            args.mcq_prompt,
            args.free_prompt,
            args.mcq_config,
            args.free_config,
        )

    route_counts = Counter(route.format_type for group in groups.values() for _item, route in group)
    print(f"Private set: {len(original_ids)} total")
    print("Format counts: " + ", ".join(f"{name}={count}" for name, count in sorted(route_counts.items())))
    print(f"Routing mode: {args.routing_mode}")

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

    csv_path, audit_path = write_outputs(output_dir, submission_path, records, original_ids)
    print(f"Submission written to {csv_path}")
    print(f"Audit JSONL written to {audit_path}")


if __name__ == "__main__":
    main()
