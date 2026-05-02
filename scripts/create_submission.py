#!/usr/bin/env python3
"""Generate a private-set submission CSV.

The default submission path uses the current problem-type-separated prompt
package in :mod:`prompts.math_reasoning_prompts`, matching the pipeline used
for the full-public 32GB balanced run. The older MCQ/free split is still
available with ``--routing-mode legacy`` and defaults to ``prompts.legacy_prompts``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_DATA_PATH = "data/private.jsonl"
DEFAULT_OUTPUT_DIR = "results/private_submission"
DEFAULT_SUBMISSION_NAME = "submission.csv"
DEFAULT_MCQ_PROMPT = "general_mcq_eliminate"
DEFAULT_FREE_PROMPT = "baseline"
DEFAULT_PROMPT_MODULE = "prompts.legacy_prompts"

FORMAT_MCQ = "mcq"
FORMAT_FREE_RESPONSE = "free_response"

_LETTER_RE = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")
_LETTER_PHRASE_RE = re.compile(
    r"(?:option|choice|answer\s+is)\s*[:\s]*\(?([A-Z])\)?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SamplingConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int
    n: int
    vote: bool = False


@dataclass(frozen=True)
class Route:
    format_type: str
    prompt_family: str
    config_name: str
    expected_answers: int
    has_options: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_type": self.format_type,
            "prompt_family": self.prompt_family,
            "config_name": self.config_name,
            "expected_answers": self.expected_answers,
            "has_options": self.has_options,
            "notes": list(self.notes),
        }


CONFIGS = {
    "greedy_n1": SamplingConfig("greedy_n1", temperature=0.0, top_p=1.0, top_k=-1, n=1),
    "sc_n1": SamplingConfig("sc_n1", temperature=0.7, top_p=0.95, top_k=20, n=1),
    "sc_n3": SamplingConfig("sc_n3", temperature=0.7, top_p=0.95, top_k=20, n=3, vote=True),
}


def load_private(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return rows, [int(row["id"]) for row in rows]


def load_prompt_module(module_name: str) -> Any:
    return importlib.import_module(module_name)


def is_problem_type_mcq_like(row: dict[str, Any]) -> bool:
    """Mirror run_math_prompts' MCQ routing for the hybrid submission path."""
    if row.get("options"):
        return True
    question = str(row.get("question", ""))
    has_a = "A." in question or "A)" in question or "(A)" in question
    has_b = "B." in question or "B)" in question or "(B)" in question
    return has_a and has_b


def render_legacy_prompts(
    tokenizer: Any,
    prompt_name: str,
    items: list[dict[str, Any]],
    prompt_module: Any,
    fallback: bool = False,
) -> list[str]:
    prompts = []
    for item in items:
        system, user = prompt_module.build_routed_prompt(
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


def _tail_after_think(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>") :] if think_end >= 0 else text


def extract_letter(text: str, options: list[str] | None, judger: Any) -> str:
    tail = _tail_after_think(text)

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
    matches = [letter for letter in re.findall(r"\b([A-Z])\b", tail.upper()) if not valid or letter in valid]
    return matches[-1] if matches else ""


def _clean_answer_text(prompt_module: Any, answer: str) -> str:
    cleaner = getattr(prompt_module, "clean_answer_text", None)
    if cleaner:
        return cleaner(answer)
    return re.sub(r"\s+", " ", str(answer)).strip()


def _rebuild_final_response(prompt_module: Any, answers: list[str]) -> str:
    rebuilder = getattr(prompt_module, "rebuild_final_response", None)
    if rebuilder:
        return rebuilder(answers)
    boxes = "\n".join(f"\\boxed{{{_clean_answer_text(prompt_module, answer)}}}" for answer in answers)
    return "FINAL_ANSWERS:\n" + boxes


def clean_legacy_final_response(
    response: str,
    item: dict[str, Any],
    route: Route,
    judger: Any,
    prompt_module: Any,
) -> str:
    if route.format_type == FORMAT_MCQ:
        letter = extract_letter(response, item.get("options"), judger)
        return _rebuild_final_response(prompt_module, [letter]) if letter else ""

    tail = _tail_after_think(response)
    try:
        answers = judger.extract_all_boxed(tail) or judger.extract_all_boxed(response)
    except Exception:
        answers = []

    if len(answers) == 1 and route.expected_answers > 1:
        try:
            split_answers = judger.split_by_comma(answers[0])
        except Exception:
            split_answers = answers
        if len(split_answers) == route.expected_answers:
            answers = split_answers

    if not answers:
        try:
            extracted = judger.extract_ans(response)
        except Exception:
            extracted = ""
        if extracted:
            answers = [extracted]

    return _rebuild_final_response(prompt_module, answers) if answers else ""


def vote_key(response: str, item: dict[str, Any], route: Route, judger: Any) -> str:
    if route.format_type == FORMAT_MCQ:
        return extract_letter(response, item.get("options"), judger)
    try:
        answer = judger.extract_ans(response)
        return judger.norm_ans_str(answer) if answer else ""
    except Exception:
        return ""


def choose_majority(
    responses: list[str],
    item: dict[str, Any],
    route: Route,
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


def generate_legacy_group(
    llm: Any,
    tokenizer: Any,
    judger: Any,
    prompt_name: str,
    config: SamplingConfig,
    routed_items: list[tuple[dict[str, Any], Route]],
    max_tokens: int,
    fallback_max_tokens: int,
    enable_fallback: bool,
    prompt_module: Any,
) -> list[dict[str, Any]]:
    from tqdm import tqdm
    from vllm import SamplingParams

    if not routed_items:
        return []

    items = [item for item, _route in routed_items]
    routes = [route for _item, route in routed_items]
    prompts = render_legacy_prompts(tokenizer, prompt_name, items, prompt_module)
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
        f"[legacy {prompt_name} / {config.name}] generating {len(items)} prompts x n={config.n} "
        f"({route_summary}) ...",
        flush=True,
    )
    outputs = llm.generate(prompts, sampling_params=sampling)
    print(f"[legacy {prompt_name} / {config.name}] generation finished; post-processing ...", flush=True)

    records = []
    fallback_indices: list[int] = []
    for idx, (item, route, output) in enumerate(
        tqdm(
            zip(items, routes, outputs),
            total=len(items),
            desc=f"Post-processing {prompt_name}",
        )
    ):
        responses = [choice.text.strip() for choice in output.outputs]
        finishes = [getattr(choice, "finish_reason", None) for choice in output.outputs]
        if config.vote and len(responses) > 1:
            chosen, chosen_idx, winning_key = choose_majority(responses, item, route, judger)
        else:
            chosen, chosen_idx = responses[0], 0
            winning_key = vote_key(responses[0], item, route, judger)

        clean_response = clean_legacy_final_response(chosen, item, route, judger, prompt_module)
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
        fallback_prompts = render_legacy_prompts(
            tokenizer,
            prompt_name,
            fallback_items,
            prompt_module,
            fallback=True,
        )
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
            fallback_clean = clean_legacy_final_response(fallback_raw, item, route, judger, prompt_module)
            record = records[record_idx]
            record["fallback_used"] = True
            record["fallback_raw_response"] = fallback_raw
            record["fallback_finish_reason"] = fallback_finish
            record["raw_response_before_fallback"] = record["raw_response"]
            if fallback_clean:
                record["raw_response"] = fallback_raw
                record["response"] = fallback_clean
                record["vote_key"] = vote_key(fallback_raw, item, route, judger)
    print(f"[legacy {prompt_name} / {config.name}] done.", flush=True)
    return records


def build_legacy_groups(
    rows: list[dict[str, Any]],
    mcq_prompt: str,
    free_prompt: str,
    mcq_config: str,
    free_config: str,
) -> dict[tuple[str, str], list[tuple[dict[str, Any], Route]]]:
    groups: dict[tuple[str, str], list[tuple[dict[str, Any], Route]]] = {}
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
    by_id = {int(record["id"]): record for record in records}
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


def make_llm(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

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
    return llm, tokenizer


def run_problem_type_submission(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from run_math_prompts import run_submission

    llm, tokenizer = make_llm(args)
    config = CONFIGS[args.config]
    print(
        f"[problem_type / {config.name}] generating {len(rows)} prompts with "
        "prompts.math_reasoning_prompts ...",
        flush=True,
    )
    return run_submission(
        rows,
        llm,
        tokenizer,
        config,
        args.max_tokens,
        include_few_shot=args.include_few_shot,
        math_type=args.math_type,
        enable_repair=not args.no_repair and not args.no_fallback,
        normalize_free_final_answers=args.normalize_free_final_answers,
        rank_free_samples=args.rank_free_samples,
    )


def run_hybrid_submission(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use run_math_prompts for MCQ-like rows and updated legacy prompts for free rows."""
    from judger import Judger
    from run_math_prompts import run_submission

    prompt_module = load_prompt_module(args.prompt_module)
    free_names = {name for name, *_ in prompt_module.FREE_PROMPTS}
    if args.free_prompt not in free_names:
        raise SystemExit(f"Unknown free prompt for {args.prompt_module}: {args.free_prompt}")

    mcq_rows = [row for row in rows if is_problem_type_mcq_like(row)]
    free_rows = [row for row in rows if not is_problem_type_mcq_like(row)]
    print(
        f"Hybrid counts: run_math_prompts_mcq_like={len(mcq_rows)}  "
        f"updated_free_response={len(free_rows)}",
        flush=True,
    )

    llm, tokenizer = make_llm(args)
    records: list[dict[str, Any]] = []

    if mcq_rows:
        config = CONFIGS[args.config]
        print(
            f"[hybrid mcq / run_math_prompts / {config.name}] generating {len(mcq_rows)} prompts ...",
            flush=True,
        )
        mcq_records = run_submission(
            mcq_rows,
            llm,
            tokenizer,
            config,
            args.max_tokens,
            include_few_shot=args.include_few_shot,
            math_type=args.math_type,
            enable_repair=not args.no_repair and not args.no_fallback,
            normalize_free_final_answers=args.normalize_free_final_answers,
            rank_free_samples=args.rank_free_samples,
        )
        for record in mcq_records:
            record.update(
                routing_mode="hybrid",
                hybrid_source="run_math_prompts",
                prompt="prompts.math_reasoning_prompts",
                config=config.name,
            )
        records.extend(mcq_records)

    if free_rows:
        judger = Judger(strict_extract=False)
        free_route_items = [
            (
                item,
                Route(
                    format_type=FORMAT_FREE_RESPONSE,
                    prompt_family=args.free_prompt,
                    config_name=args.free_config,
                    expected_answers=max(1, str(item.get("question", "")).count("[ANS]")),
                    notes=("hybrid_updated_free", args.prompt_module),
                ),
            )
            for item in free_rows
        ]
        free_records = generate_legacy_group(
            llm,
            tokenizer,
            judger,
            args.free_prompt,
            CONFIGS[args.free_config],
            free_route_items,
            args.max_tokens,
            args.fallback_max_tokens,
            not args.no_fallback,
            prompt_module,
        )
        for record in free_records:
            record.update(
                routing_mode="hybrid",
                hybrid_source="updated_free_prompt",
                prompt_module=args.prompt_module,
            )
        records.extend(free_records)

    return records


def run_legacy_submission(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from judger import Judger

    prompt_module = load_prompt_module(args.prompt_module)
    mcq_names = {name for name, *_ in prompt_module.MCQ_PROMPTS}
    free_names = {name for name, *_ in prompt_module.FREE_PROMPTS}
    if args.mcq_prompt not in mcq_names:
        raise SystemExit(f"Unknown MCQ prompt for {args.prompt_module}: {args.mcq_prompt}")
    if args.free_prompt not in free_names:
        raise SystemExit(f"Unknown free prompt for {args.prompt_module}: {args.free_prompt}")

    groups = build_legacy_groups(
        rows,
        args.mcq_prompt,
        args.free_prompt,
        args.mcq_config,
        args.free_config,
    )
    route_counts = Counter(route.format_type for group in groups.values() for _item, route in group)
    print("Legacy format counts: " + ", ".join(f"{name}={count}" for name, count in sorted(route_counts.items())))

    llm, tokenizer = make_llm(args)
    judger = Judger(strict_extract=False)

    records = []
    for (prompt_name, config_name), group in sorted(groups.items()):
        records.extend(
            generate_legacy_group(
                llm,
                tokenizer,
                judger,
                prompt_name,
                CONFIGS[config_name],
                group,
                args.max_tokens,
                args.fallback_max_tokens,
                not args.no_fallback,
                prompt_module,
            )
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--submission-name", default=DEFAULT_SUBMISSION_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--routing-mode",
        choices=["problem_type", "legacy", "format", "hybrid"],
        default="problem_type",
        help=(
            "'problem_type' uses the latest separated prompt package. "
            "'hybrid' uses run_math_prompts for MCQ-like rows and --prompt-module/--free-prompt for free rows. "
            "'format' is a deprecated alias."
        ),
    )
    parser.add_argument("--config", default="sc_n3", choices=sorted(CONFIGS))
    parser.add_argument("--prompt-module", default=DEFAULT_PROMPT_MODULE)
    parser.add_argument("--mcq-prompt", default=DEFAULT_MCQ_PROMPT)
    parser.add_argument("--free-prompt", default=DEFAULT_FREE_PROMPT)
    parser.add_argument("--mcq-config", default="greedy_n1", choices=sorted(CONFIGS))
    parser.add_argument("--free-config", default="sc_n3", choices=sorted(CONFIGS))
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--fallback-max-tokens", type=int, default=2048)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--math-type", default=None)
    parser.add_argument("--include-few-shot", action="store_true")
    parser.add_argument(
        "--normalize-free-final-answers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize non-option final boxed answers to plain ASCII in problem_type mode.",
    )
    parser.add_argument(
        "--rank-free-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For non-option n>1 problem_type runs, choose the best formatted/precision sample.",
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    output_dir = Path(args.output_dir)
    submission_path = output_dir / args.submission_name
    rows, original_ids = load_private(Path(args.data_path))
    routing_mode = "problem_type" if args.routing_mode == "format" else args.routing_mode

    print(f"Private set: {len(original_ids)} total")
    print(f"Routing mode: {routing_mode}")
    if routing_mode == "problem_type":
        print("Prompt pipeline: prompts.math_reasoning_prompts submission_response_mode")
        records = run_problem_type_submission(args, rows)
    elif routing_mode == "hybrid":
        print(
            "Prompt pipeline: hybrid run_math_prompts MCQ + "
            f"{args.prompt_module}.{args.free_prompt} free response"
        )
        records = run_hybrid_submission(args, rows)
    else:
        print(f"Prompt pipeline: legacy split via {args.prompt_module}")
        records = run_legacy_submission(args, rows)

    csv_path, audit_path = write_outputs(output_dir, submission_path, records, original_ids)
    print(f"Submission written to {csv_path}")
    print(f"Audit JSONL written to {audit_path}")


if __name__ == "__main__":
    main()
