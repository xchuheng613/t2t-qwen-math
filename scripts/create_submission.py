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
    r"(?:correct\s+option\s+is|correct\s+answer\s+is|answer\s+is|"
    r"therefore|thus|hence|i\s+choose|choose|selected|option|choice)"
    r"\s*[:\s]*\(?([A-Z])\)?\b",
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


def render_legacy_fallback_prompts(
    tokenizer: Any,
    prompt_name: str,
    items: list[dict[str, Any]],
    routes: list[Route],
    raw_responses: list[str],
    prompt_module: Any,
    stage: str,
    tail_tokens: int,
) -> list[str]:
    prompts = []
    for item, route, raw_response in zip(items, routes, raw_responses):
        previous_tail = _tail_by_tokens(tokenizer, raw_response, tail_tokens)
        system, user = build_stage_fallback_prompt(
            prompt_module,
            stage,
            prompt_name,
            item["question"],
            item.get("options") or None,
            raw_response=raw_response,
            previous_tail=previous_tail,
            required_answers=route.expected_answers,
        )
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def _tail_by_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text.strip()
        return tokenizer.decode(token_ids[-max_tokens:], skip_special_tokens=True).strip()
    except Exception:
        return text[-max_tokens * 4 :].strip()


def _repair_sqrt_artifacts(text: str) -> str:
    return str(text).replace("sqrt{(}", "sqrt(")


def _options_block(options: list[str] | None) -> str:
    if not options:
        return ""
    return "\n\nOptions:\n" + "\n".join(
        f"{chr(65 + idx)}. {str(option).strip()}"
        for idx, option in enumerate(options)
    )


def _generic_stage_fallback_prompt(
    stage: str,
    question: str,
    options: list[str] | None,
    *,
    previous_tail: str,
    required_answers: int,
) -> tuple[str, str]:
    if stage == "continuation":
        system = (
            "Continue from the previous reasoning. Do not restart the solution. "
            "Use the previous work to finish the answer.\n\n"
            "Output only:\n"
            "FINAL_ANSWERS:\n"
            "\\boxed{answer}\n\n"
            "No explanation. For MCQ, box only uppercase option letter(s). "
            f"For free-response, return exactly {required_answers} answer(s) in order; "
            "if multiple answers are needed, put them in one box separated by commas."
        )
        user = (
            f"Problem:\n{question.strip()}{_options_block(options)}\n\n"
            f"Previous response tail:\n{previous_tail.strip()}"
        )
        return system, user

    if stage == "bounded":
        system = (
            "Solve concisely. Do not verify repeatedly. Use at most 8 short "
            "reasoning steps, then finish the answer.\n\n"
            "Output only:\n"
            "FINAL_ANSWERS:\n"
            "\\boxed{answer}\n\n"
            "No explanation after the final box. For MCQ, box only uppercase "
            f"option letter(s). For free-response, return exactly {required_answers} "
            "answer(s) in order; if multiple answers are needed, put them in one "
            "box separated by commas."
        )
        user = f"Problem:\n{question.strip()}{_options_block(options)}"
        return system, user

    raise ValueError(f"Unknown fallback stage: {stage}")


def build_stage_fallback_prompt(
    prompt_module: Any,
    stage: str,
    prompt_name: str,
    question: str,
    options: list[str] | None,
    *,
    raw_response: str,
    previous_tail: str,
    required_answers: int,
) -> tuple[str, str]:
    custom_name = f"build_{stage}_fallback_prompt"
    custom_builder = getattr(prompt_module, custom_name, None)
    if custom_builder:
        return custom_builder(
            prompt_name,
            question,
            options,
            raw_response=raw_response,
            previous_tail=previous_tail,
            required_answers=required_answers,
        )

    if stage not in {"continuation", "bounded"}:
        custom_builder = getattr(prompt_module, "build_fallback_prompt", None)
        if custom_builder:
            return custom_builder(
                prompt_name,
                question,
                options,
                raw_response=raw_response,
                required_answers=required_answers,
            )
        else:
            return prompt_module.build_routed_prompt(prompt_name, question, options, fallback=True)

    return _generic_stage_fallback_prompt(
        stage,
        question,
        options,
        previous_tail=previous_tail,
        required_answers=required_answers,
    )


def extract_raw_final_boxed_group(text: str) -> list[str]:
    """Extract final boxed contents without judge normalization."""
    search_text = _tail_after_think(text)
    entries: list[tuple[int, int, str]] = []
    start = 0
    while True:
        idx = search_text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        i = brace_start
        while i < len(search_text) and depth > 0:
            if search_text[i] == "{":
                depth += 1
            elif search_text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            content = search_text[brace_start : i - 1].strip()
            if content:
                entries.append((idx, i, content))
        start = i

    if not entries:
        return []

    last_group = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = search_text[entries[j][1] : entries[j + 1][0]]
        if re.match(r"^[\s,\$\.\;\:\-\&\\]*$", gap):
            last_group.insert(0, entries[j])
        else:
            break
    return [_repair_sqrt_artifacts(entry[2]) for entry in last_group]


def _extract_raw_boxed_anywhere(text: str) -> list[str]:
    entries: list[str] = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            content = text[brace_start : i - 1].strip()
            if content:
                entries.append(_repair_sqrt_artifacts(content))
        start = i
    return entries


def _final_answer_text(response: str) -> str:
    answers = extract_raw_final_boxed_group(response)
    return ", ".join(answers)


def answer_looks_like_reasoning(answer: str) -> bool:
    bad_markers = [
        "let's",
        "wait",
        "case",
        "suppose",
        "strategy",
        "turn",
        "therefore",
        "because",
        "we need",
        "the problem",
    ]
    answer = str(answer)
    lowered = answer.lower()
    return (
        len(answer) > 250
        or "\n" in answer
        or any(marker in lowered for marker in bad_markers)
    )


def _answers_match_expected_count(answers: list[str], expected: int, judger: Any) -> list[str]:
    answers = [_repair_sqrt_artifacts(answer) for answer in answers]
    if expected <= 1:
        return answers[-1:]
    if len(answers) == expected:
        return answers
    if len(answers) == 1:
        try:
            split_answers = judger.split_by_comma(answers[0])
        except Exception:
            split_answers = answers
        if len(split_answers) == expected:
            return split_answers
    if len(answers) > expected:
        return answers[-expected:]
    return []


def _prompt_token_count(tokenizer: Any, prompt_text: str) -> int:
    try:
        return len(tokenizer.encode(prompt_text, add_special_tokens=False))
    except Exception:
        return max(1, len(prompt_text) // 4)


def _high_budget_sampling_params(max_tokens: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        n=1,
    )


def _apply_fallback_result(
    record: dict[str, Any],
    item: dict[str, Any],
    route: Route,
    judger: Any,
    prompt_module: Any,
    *,
    stage: str,
    raw_response: str,
    finish_reason: str | None,
    max_tokens: int,
) -> bool:
    clean_response = clean_legacy_final_response(raw_response, item, route, judger, prompt_module)
    reasoning_like_answer = (
        str(finish_reason).lower() == "length"
        and answer_looks_like_reasoning(_final_answer_text(clean_response))
    )
    attempt = {
        "stage": stage,
        "finish_reason": finish_reason,
        "max_tokens": max_tokens,
        "cleaned": bool(clean_response) and not reasoning_like_answer,
        "rejected_reasoning_like_answer": reasoning_like_answer,
        "raw_response": raw_response,
    }
    record.setdefault("fallback_attempts", []).append(attempt)
    if not clean_response or reasoning_like_answer:
        return False

    record["fallback_used"] = True
    record["fallback_stage"] = stage
    record["fallback_raw_response"] = raw_response
    record["fallback_finish_reason"] = finish_reason
    record.setdefault("raw_response_before_fallback", record["raw_response"])
    record["raw_response"] = raw_response
    record["response"] = clean_response
    record["vote_key"] = vote_key(raw_response, item, route, judger)
    return True


def _run_fallback_stage(
    llm: Any,
    tokenizer: Any,
    judger: Any,
    prompt_name: str,
    records: list[dict[str, Any]],
    items: list[dict[str, Any]],
    routes: list[Route],
    unresolved_indices: list[int],
    prompt_module: Any,
    *,
    stage: str,
    max_tokens: int,
    tail_tokens: int,
) -> list[int]:
    if not unresolved_indices:
        return []

    stage_items = [items[idx] for idx in unresolved_indices]
    stage_routes = [routes[idx] for idx in unresolved_indices]
    stage_raws = [records[idx]["raw_response"] for idx in unresolved_indices]
    fallback_prompts = render_legacy_fallback_prompts(
        tokenizer,
        prompt_name,
        stage_items,
        stage_routes,
        stage_raws,
        prompt_module,
        stage=stage,
        tail_tokens=tail_tokens,
    )
    print(
        f"  fallback stage={stage}: retrying {len(stage_items)} rows with max_tokens={max_tokens} ...",
        flush=True,
    )
    fallback_outputs = llm.generate(
        fallback_prompts,
        sampling_params=_fallback_sampling_params(max_tokens),
    )

    still_unresolved: list[int] = []
    for record_idx, item, route, output in zip(unresolved_indices, stage_items, stage_routes, fallback_outputs):
        fallback_raw = output.outputs[0].text.strip()
        fallback_finish = getattr(output.outputs[0], "finish_reason", None)
        ok = _apply_fallback_result(
            records[record_idx],
            item,
            route,
            judger,
            prompt_module,
            stage=stage,
            raw_response=fallback_raw,
            finish_reason=fallback_finish,
            max_tokens=max_tokens,
        )
        if not ok:
            still_unresolved.append(record_idx)
    return still_unresolved


def _run_high_budget_fallback(
    llm: Any,
    tokenizer: Any,
    judger: Any,
    prompt_name: str,
    records: list[dict[str, Any]],
    items: list[dict[str, Any]],
    routes: list[Route],
    unresolved_indices: list[int],
    prompt_module: Any,
    *,
    high_budget_max_tokens: int,
    max_model_len: int,
    context_safety_margin: int,
) -> list[int]:
    if not unresolved_indices:
        return []

    stage_items = [items[idx] for idx in unresolved_indices]
    stage_routes = [routes[idx] for idx in unresolved_indices]
    prompts = render_legacy_prompts(tokenizer, prompt_name, stage_items, prompt_module)
    max_prompt_tokens = max(_prompt_token_count(tokenizer, prompt) for prompt in prompts)
    effective_max_tokens = min(
        high_budget_max_tokens,
        max_model_len - max_prompt_tokens - context_safety_margin,
    )
    if effective_max_tokens <= 0:
        print(
            "  fallback stage=high_budget: skipped because prompts do not fit "
            f"max_model_len={max_model_len} with safety={context_safety_margin}.",
            flush=True,
        )
        for idx in unresolved_indices:
            records[idx].setdefault("fallback_attempts", []).append(
                {
                    "stage": "high_budget",
                    "skipped": True,
                    "reason": "insufficient_context_window",
                    "max_prompt_tokens": max_prompt_tokens,
                    "max_model_len": max_model_len,
                }
            )
        return unresolved_indices

    print(
        f"  fallback stage=high_budget: retrying {len(stage_items)} rows with "
        f"max_tokens={effective_max_tokens} (max_prompt_tokens={max_prompt_tokens}) ...",
        flush=True,
    )
    outputs = llm.generate(
        prompts,
        sampling_params=_high_budget_sampling_params(effective_max_tokens),
    )

    still_unresolved: list[int] = []
    for record_idx, item, route, output in zip(unresolved_indices, stage_items, stage_routes, outputs):
        fallback_raw = output.outputs[0].text.strip()
        fallback_finish = getattr(output.outputs[0], "finish_reason", None)
        ok = _apply_fallback_result(
            records[record_idx],
            item,
            route,
            judger,
            prompt_module,
            stage="high_budget",
            raw_response=fallback_raw,
            finish_reason=fallback_finish,
            max_tokens=effective_max_tokens,
        )
        if not ok:
            still_unresolved.append(record_idx)
    return still_unresolved


def _tail_after_think(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>") :] if think_end >= 0 else text


def extract_letter(text: str, options: list[str] | None, judger: Any) -> str:
    tail = _tail_after_think(text)

    match = _LETTER_RE.search(tail) or _LETTER_RE.search(text)
    if match:
        return match.group(1).upper()

    if options:
        boxed_contents = extract_raw_final_boxed_group(tail) or extract_raw_final_boxed_group(text)
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


def _clean_answer_text(prompt_module: Any, answer: str, question: str = "") -> str:
    cleaner = getattr(prompt_module, "clean_answer_text", None)
    if cleaner:
        try:
            return cleaner(answer, question=question)
        except TypeError:
            return cleaner(answer)
    return re.sub(r"\s+", " ", str(answer)).strip()


def _rebuild_final_response(prompt_module: Any, answers: list[str], question: str = "") -> str:
    rebuilder = getattr(prompt_module, "rebuild_final_response", None)
    if rebuilder:
        try:
            return rebuilder(answers, question=question)
        except TypeError:
            return rebuilder(answers)
    boxes = "\n".join(
        f"\\boxed{{{_clean_answer_text(prompt_module, answer, question)}}}"
        for answer in answers
    )
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
        return _rebuild_final_response(prompt_module, [letter], item.get("question", "")) if letter else ""

    tail = _tail_after_think(response)
    answers = extract_raw_final_boxed_group(tail) or extract_raw_final_boxed_group(response)
    answers = _answers_match_expected_count(answers, route.expected_answers, judger)
    if not answers:
        all_boxed = _extract_raw_boxed_anywhere(tail) or _extract_raw_boxed_anywhere(response)
        answers = _answers_match_expected_count(all_boxed, route.expected_answers, judger)

    if not answers:
        try:
            extracted = judger.extract_ans(response)
        except Exception:
            extracted = ""
        if extracted:
            answers = [extracted]
    answers = _answers_match_expected_count(answers, route.expected_answers, judger)

    return _rebuild_final_response(prompt_module, answers, item.get("question", "")) if answers else ""


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
    fallback_tail_tokens: int = 6000,
    enable_high_budget_fallback: bool = False,
    high_budget_max_tokens: int = 32768,
    max_model_len: int = 32768,
    context_safety_margin: int = 512,
    enable_stage0_postprocess: bool = True,
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

        finish_reason = finishes[chosen_idx] if chosen_idx < len(finishes) else None
        clean_response = (
            clean_legacy_final_response(chosen, item, route, judger, prompt_module)
            if enable_stage0_postprocess
            else ""
        )
        rejected_reasoning_like_answer = (
            enable_stage0_postprocess
            and bool(clean_response)
            and
            str(finish_reason).lower() == "length"
            and answer_looks_like_reasoning(_final_answer_text(clean_response))
        )
        if rejected_reasoning_like_answer:
            clean_response = ""
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
            "fallback_stage": "",
            "fallback_attempts": [],
            "stage0_postprocess_enabled": enable_stage0_postprocess,
            "stage0_repair_used": bool(enable_stage0_postprocess and clean_response and clean_response != chosen),
            "stage0_rejected_reasoning_like_answer": rejected_reasoning_like_answer,
        }
        # Stage 0 is the cleanup above: if a valid final block can be rebuilt
        # from the raw/truncated text, do not spend another inference call.
        if enable_fallback and enable_stage0_postprocess and not clean_response:
            fallback_indices.append(idx)
        records.append(record)

    if fallback_indices:
        unresolved = _run_fallback_stage(
            llm,
            tokenizer,
            judger,
            prompt_name,
            records,
            items,
            routes,
            fallback_indices,
            prompt_module,
            stage="continuation",
            max_tokens=fallback_max_tokens,
            tail_tokens=fallback_tail_tokens,
        )
        unresolved = _run_fallback_stage(
            llm,
            tokenizer,
            judger,
            prompt_name,
            records,
            items,
            routes,
            unresolved,
            prompt_module,
            stage="bounded",
            max_tokens=fallback_max_tokens,
            tail_tokens=fallback_tail_tokens,
        )
        if enable_high_budget_fallback:
            unresolved = _run_high_budget_fallback(
                llm,
                tokenizer,
                judger,
                prompt_name,
                records,
                items,
                routes,
                unresolved,
                prompt_module,
                high_budget_max_tokens=high_budget_max_tokens,
                max_model_len=max_model_len,
                context_safety_margin=context_safety_margin,
            )
        if unresolved:
            print(f"  fallback: {len(unresolved)} rows still unresolved after staged fallback.", flush=True)
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
            fallback_tail_tokens=args.fallback_tail_tokens,
            enable_high_budget_fallback=args.high_budget_fallback,
            high_budget_max_tokens=args.high_budget_max_tokens,
            max_model_len=args.max_model_len,
            context_safety_margin=args.context_safety_margin,
            enable_stage0_postprocess=args.stage0_postprocess,
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
                fallback_tail_tokens=args.fallback_tail_tokens,
                enable_high_budget_fallback=args.high_budget_fallback,
                high_budget_max_tokens=args.high_budget_max_tokens,
                max_model_len=args.max_model_len,
                context_safety_margin=args.context_safety_margin,
                enable_stage0_postprocess=args.stage0_postprocess,
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
    parser.add_argument(
        "--fallback-max-tokens",
        type=int,
        default=8192,
        help="Token budget for continuation and bounded-solving fallback stages.",
    )
    parser.add_argument(
        "--fallback-tail-tokens",
        type=int,
        default=6000,
        help="Approximate number of previous-response tokens to include in continuation fallback.",
    )
    parser.add_argument(
        "--high-budget-fallback",
        action="store_true",
        help="After staged fallback fails, rerun unresolved rows with a larger dynamic generation budget.",
    )
    parser.add_argument("--high-budget-max-tokens", type=int, default=32768)
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument(
        "--stage0-postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For legacy/hybrid free paths, rebuild clean final-answer blocks before fallback.",
    )
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
