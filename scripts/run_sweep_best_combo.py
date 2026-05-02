#!/usr/bin/env python3
"""Run the best sweep MCQ/free prompt combination on a larger subset.

This is intentionally separate from prompt_sweep.py so the original
results/sweep/summary.csv is not overwritten.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_DATA_PATH = "data/public.jsonl"
DEFAULT_OUTPUT_DIR = "results/sweep_best100"
DEFAULT_MCQ_PROMPT = "eliminate"
DEFAULT_FREE_PROMPT = "baseline"
DEFAULT_PROMPT_MODULE = "prompts.legacy_prompts"
DEFAULT_SEED = 42


@dataclass(frozen=True)
class SamplingConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int
    n: int
    vote: bool = False


GREEDY_N1 = SamplingConfig(name="greedy_n1", temperature=0.0, top_p=1.0, top_k=-1, n=1)
SC_N3 = SamplingConfig(name="sc_n3", temperature=0.7, top_p=0.95, top_k=20, n=3, vote=True)
CONFIGS = {config.name: config for config in (GREEDY_N1, SC_N3)}

_LETTER_RE = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")
_LETTER_PHRASE_RE = re.compile(
    r"(?:option|choice|answer\s+is)\s*[:\s]*\(?([A-Z])\)?\b",
    re.IGNORECASE,
)


def load_stratified_subset(path: Path, k: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    data = [json.loads(line) for line in path.open()]
    mcq = [row for row in data if row.get("options")]
    free = [row for row in data if not row.get("options")]
    rng.shuffle(mcq)
    rng.shuffle(free)

    half = k // 2
    n_mcq = min(half, len(mcq))
    n_free = min(k - n_mcq, len(free))
    if n_mcq + n_free < k:
        n_mcq = min(k - n_free, len(mcq))
    return mcq[:n_mcq], free[:n_free]


def load_prompt_module(module_name: str) -> Any:
    return importlib.import_module(module_name)


def render_prompts(
    tokenizer: Any,
    qtype: str,
    prompt_name: str,
    items: list[dict[str, Any]],
    prompt_module: Any,
) -> list[str]:
    build_free_prompt = prompt_module.build_free_prompt
    build_mcq_prompt = prompt_module.build_mcq_prompt

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


def extract_letter(text: str, options: list[str] | None, judger: Any) -> str:
    think_end = text.rfind("</think>")
    tail = text[think_end + len("</think>") :] if think_end >= 0 else text

    match = _LETTER_RE.search(tail) or _LETTER_RE.search(text)
    if match:
        return match.group(1).upper()

    if options:
        try:
            boxed_contents = judger.extract_all_boxed(tail) or judger.extract_all_boxed(text)
        except Exception:
            boxed_contents = []
        if boxed_contents:
            candidate = boxed_contents[-1]
            try:
                candidate_norm = judger.norm_ans_str(candidate)
            except Exception:
                candidate_norm = candidate
            for idx, option in enumerate(options):
                option_text = str(option).strip()
                if candidate.strip() == option_text or candidate_norm == option_text:
                    return chr(65 + idx)
                try:
                    if judger.is_equal(candidate_norm, judger.norm_ans_str(option_text)):
                        return chr(65 + idx)
                except Exception:
                    pass

    phrase_matches = list(_LETTER_PHRASE_RE.finditer(tail))
    if phrase_matches:
        return phrase_matches[-1].group(1).upper()

    valid = {chr(65 + idx) for idx in range(len(options or []))}
    matches = [letter for letter in re.findall(r"\b([A-Z])\b", tail.upper()) if letter in valid]
    return matches[-1] if matches else ""


def score_response(judger: Any, item: dict[str, Any], qtype: str, response: str) -> bool:
    if qtype == "mcq":
        return extract_letter(response, item.get("options"), judger) == str(item["answer"]).strip().upper()

    gold = item["answer"]
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return bool(judger.auto_judge(pred=response, gold=gold_list, options=[[]] * len(gold_list)))
    except Exception:
        return False


def vote_key(response: str, item: dict[str, Any], qtype: str, judger: Any) -> str:
    if qtype == "mcq":
        return extract_letter(response, item.get("options"), judger)
    try:
        answer = judger.extract_ans(response)
        return judger.norm_ans_str(answer) if answer else ""
    except Exception:
        return ""


def choose_majority(responses: list[str], item: dict[str, Any], qtype: str, judger: Any) -> str:
    keys = [vote_key(response, item, qtype, judger) for response in responses]
    counts = Counter(key for key in keys if key)
    if not counts:
        return responses[0]
    winning = counts.most_common(1)[0][0]
    for key, response in zip(keys, responses):
        if key == winning:
            return response
    return responses[0]


def run_one(
    llm: Any,
    tokenizer: Any,
    judger: Any,
    output_dir: Path,
    qtype: str,
    prompt_name: str,
    config: SamplingConfig,
    items: list[dict[str, Any]],
    max_tokens: int,
    prompt_module: Any,
) -> dict[str, Any]:
    from vllm import SamplingParams

    prompts = render_prompts(tokenizer, qtype, prompt_name, items, prompt_module)
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        n=config.n,
    )
    print(f"[{qtype} / {prompt_name} / {config.name}] generating {len(prompts)} prompts x n={config.n} ...")
    outputs = llm.generate(prompts, sampling_params=sampling)

    records = []
    for item, output in zip(items, outputs):
        responses = [choice.text.strip() for choice in output.outputs]
        chosen = choose_majority(responses, item, qtype, judger) if config.vote and len(responses) > 1 else responses[0]
        records.append(
            {
                "id": item.get("id"),
                "is_mcq": qtype == "mcq",
                "gold": item["answer"],
                "response": chosen,
                "all_samples": responses if config.n > 1 else None,
                "correct": score_response(judger, item, qtype, chosen),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{qtype}__{prompt_name}__{config.name}.jsonl"
    with out_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    correct = sum(bool(record["correct"]) for record in records)
    summary = {
        "type": qtype,
        "prompt": prompt_name,
        "config": config.name,
        "n": len(records),
        "correct": correct,
        "acc": correct / len(records) if records else 0.0,
        "path": str(out_path),
    }
    print(f"  -> {correct}/{len(records)} = {summary['acc']:.3f} [{out_path}]")
    return summary


def write_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    total_n = sum(row["n"] for row in summaries)
    total_correct = sum(row["correct"] for row in summaries)
    combined = {
        "type": "combined",
        "prompt": "__".join(f"{row['type']}_{row['prompt']}" for row in summaries),
        "config": "__".join(f"{row['type']}_{row['config']}" for row in summaries),
        "n": total_n,
        "correct": total_correct,
        "acc": total_correct / total_n if total_n else 0.0,
        "path": "",
    }
    rows = [combined, *summaries]
    path = output_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["type", "prompt", "config", "n", "correct", "acc", "path"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-module",
        default=DEFAULT_PROMPT_MODULE,
        help=(
            "Prompt module to use. Defaults to prompts.legacy_prompts; use "
            "prompts.legacy_prompts_v2 to opt into the new routed free prompts."
        ),
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mcq-prompt", default=DEFAULT_MCQ_PROMPT)
    parser.add_argument("--free-prompt", default=DEFAULT_FREE_PROMPT)
    parser.add_argument("--mcq-config", default=GREEDY_N1.name, choices=sorted(CONFIGS))
    parser.add_argument("--free-config", default=SC_N3.name, choices=sorted(CONFIGS))
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from judger import Judger
    from transformers import AutoTokenizer
    from vllm import LLM

    output_dir = Path(args.output_dir)
    prompt_module = load_prompt_module(args.prompt_module)
    mcq_names = {name for name, *_ in prompt_module.MCQ_PROMPTS}
    free_names = {name for name, *_ in prompt_module.FREE_PROMPTS}
    if args.mcq_prompt not in mcq_names:
        raise SystemExit(f"Unknown MCQ prompt for {args.prompt_module}: {args.mcq_prompt}")
    if args.free_prompt not in free_names:
        raise SystemExit(f"Unknown free prompt for {args.prompt_module}: {args.free_prompt}")

    mcq_config = CONFIGS[args.mcq_config]
    free_config = CONFIGS[args.free_config]
    mcq_items, free_items = load_stratified_subset(Path(args.data_path), args.num_examples, args.seed)
    print(f"Subset: {len(mcq_items)} MCQ + {len(free_items)} free-form = {len(mcq_items) + len(free_items)} total")
    print(f"MCQ:  {args.mcq_prompt} / {mcq_config.name}")
    print(f"FREE: {args.free_prompt} / {free_config.name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=args.model,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=16384,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
    )
    judger = Judger(strict_extract=False)

    summaries = [
        run_one(
            llm,
            tokenizer,
            judger,
            output_dir,
            "mcq",
            args.mcq_prompt,
            mcq_config,
            mcq_items,
            args.max_tokens,
            prompt_module,
        ),
        run_one(
            llm,
            tokenizer,
            judger,
            output_dir,
            "free",
            args.free_prompt,
            free_config,
            free_items,
            args.max_tokens,
            prompt_module,
        ),
    ]
    summary_path = write_summary(output_dir, summaries)
    total_correct = sum(row["correct"] for row in summaries)
    total_n = sum(row["n"] for row in summaries)
    print(f"Combined: {total_correct}/{total_n} = {total_correct / total_n:.3f}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
