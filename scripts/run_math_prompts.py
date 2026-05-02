#!/usr/bin/env python3
"""Runner for the new prompt package in :mod:`prompts.math_reasoning_prompts`.

This script is the executable counterpart to the prompt package. It loads a
JSONL dataset, builds prompts via the package's builder functions in either
``internal_answer_json_mode`` or ``submission_response_mode``, runs them
through vLLM, parses the output, and writes:

  - submission mode -> ``submission.csv`` with the ``id,response`` columns
                       (the format expected by ``data/sample_submission.csv``)
                       plus an audit JSONL beside it.
  - internal mode   -> ``internal.jsonl`` with one record per row containing
                       the parsed JSON answer, validity flags, and the raw
                       generation.

A one-shot repair pass is automatically attempted on rows that fail
validation, using the matching repair prompt from the package.

Examples
--------

    # Submission CSV from the private set, format-routed prompts.
    python run_math_prompts.py \\
        --mode submission \\
        --data-path data/private.jsonl \\
        --output-dir results/private_new_prompts

    # Internal JSON answers on the public set (for offline scoring).
    python run_math_prompts.py \\
        --mode internal \\
        --data-path data/public.jsonl \\
        --output-dir results/public_new_prompts \\
        --max-tokens 1024

    # Quick dry-run: just print the (system, user) prompts without a model.
    python run_math_prompts.py --mode internal --dry-run --limit 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Ensure the project root is importable (script lives in scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts.math_reasoning_prompts import (
    Mode,
    build_internal_prompt,
    build_submission_prompt,
    build_repair_internal_prompt,
    build_repair_submission_prompt,
)
from utils import last_boxed_only_string, remove_boxed


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


# ════════════════════════════════════════════════════════════════════════════
# Sampling configs (mirrors create_submission / prompt_sweep)
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SamplingConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int
    n: int


CONFIGS: dict[str, SamplingConfig] = {
    "greedy_n1": SamplingConfig("greedy_n1", 0.0, 1.0, -1, 1),
    "sc_n1": SamplingConfig("sc_n1", 0.7, 0.95, 20, 1),
    "sc_n3": SamplingConfig("sc_n3", 0.7, 0.95, 20, 3),
}


# ════════════════════════════════════════════════════════════════════════════
# Data IO
# ════════════════════════════════════════════════════════════════════════════

def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def n_blanks(question: str) -> int:
    return question.count("[ANS]")


# ════════════════════════════════════════════════════════════════════════════
# Output parsing
# ════════════════════════════════════════════════════════════════════════════

_THINK_RE = re.compile(r"<think\b.*?</think>", re.DOTALL | re.IGNORECASE)
_FINAL_MARKER_RE = re.compile(
    r"Final\s+answers?(?:\s*,\s*in\s+order)?\s*:\s*\\boxed\{",
    re.IGNORECASE,
)
_LETTER_RE = re.compile(r"^[A-Z]+$")
_LATEX_ANSWER_CMD_RE = re.compile(
    r"\\(?:d?frac|tfrac|infty|sin|cos|tan|cot|sec|csc|sqrt|pi|theta)\b"
)
_ROUNDING_HINT_RE = re.compile(
    r"\b(round|rounded|nearest|decimal places?|significant figures?|"
    r"to \d+ places?|to the nearest)\b",
    re.IGNORECASE,
)
_EXACT_HINT_RE = re.compile(
    r"\b(exact|formula|expression|function|model|in terms of|symbolic|"
    r"simplify|closed form)\b",
    re.IGNORECASE,
)
_TOP_LEVEL_BRACKETS = {"(": ")", "[": "]", "{": "}", "<": ">"}


def strip_think_tags(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def is_free_response_row(row: dict[str, Any]) -> bool:
    """Rows without an explicit options field benefit from free-answer cleanup."""
    return not bool(row.get("options"))


def expected_answer_count(row: dict[str, Any]) -> int | None:
    blanks = n_blanks(str(row.get("question", "")))
    return blanks if blanks >= 1 else None


def split_top_level_commas(expr: str) -> list[str]:
    """Split on commas outside brackets/braces, preserving intervals/pairs."""
    parts: list[str] = []
    stack: list[str] = []
    start = 0
    i = 0
    while i < len(expr):
        char = expr[i]
        if char == "\\" and i + 1 < len(expr):
            i += 2
            continue
        if char in _TOP_LEVEL_BRACKETS:
            stack.append(_TOP_LEVEL_BRACKETS[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == "," and not stack:
            parts.append(expr[start:i].strip())
            start = i + 1
        i += 1
    tail = expr[start:].strip()
    if tail:
        parts.append(tail)
    return parts or [expr.strip()]


def preferred_theta_name(row: dict[str, Any]) -> str:
    question = str(row.get("question", ""))
    if re.search(r"(?<![A-Za-z])t(?![A-Za-z])", question):
        return "t"
    return "theta"


def _compound_ascii(expr: str) -> bool:
    return bool(re.search(r"[+\-*/^]", expr.strip()))


def _wrap_if_compound(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return expr
    if (expr.startswith("(") and expr.endswith(")")) or (
        expr.startswith("[") and expr.endswith("]")
    ):
        return expr
    return f"({expr})" if _compound_ascii(expr) else expr


def _replace_simple_latex_fraction(text: str) -> str:
    frac_re = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")

    def repl(match: re.Match[str]) -> str:
        num = _wrap_if_compound(match.group(1))
        den = _wrap_if_compound(match.group(2))
        return f"{num}/{den}"

    previous = None
    current = text
    while previous != current:
        previous = current
        current = frac_re.sub(repl, current)
    return current


def _replace_latex_functions(text: str) -> str:
    # Braced functions: \sqrt{2}, \ln{x}, \cos{t}
    for name in ("sqrt", "ln", "log", "sin", "cos", "tan", "cot", "sec", "csc"):
        text = re.sub(rf"\\{name}\s*\{{([^{{}}]+)\}}", rf"{name}(\1)", text)
    # Space functions: \cos t, \sin theta
    for name in ("ln", "log", "sin", "cos", "tan", "cot", "sec", "csc"):
        text = re.sub(rf"\\{name}\s+([A-Za-z][A-Za-z0-9_]*)", rf"{name}(\1)", text)
    # Bare commands that may already have parenthesized arguments.
    for name in ("ln", "log", "sin", "cos", "tan", "cot", "sec", "csc"):
        text = text.replace(f"\\{name}", name)
    return text


def normalize_answer_atom(answer: str, row: dict[str, Any]) -> str:
    """Convert common LaTeX final-answer syntax to plain ASCII."""
    text = answer.strip().strip("$")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\;", "").replace("\\:", "")
    text = text.replace("\\!", "").replace("\\ ", " ")
    text = text.replace("\\{", "{").replace("\\}", "}")
    text = text.replace("\\langle", "<").replace("\\rangle", ">")
    text = text.replace("\\infty", "infinity").replace("∞", "infinity")
    text = text.replace("\\pi", "pi")
    text = text.replace("\\theta", preferred_theta_name(row))
    text = re.sub(r"\\(?=\d)", "", text)
    text = _replace_latex_functions(text)
    text = _replace_simple_latex_fraction(text)
    text = re.sub(r"e\^\{([^{}]+)\}", r"e^(\1)", text)
    text = re.sub(r"([A-Za-z0-9_)])\^\{([^{}]+)\}", r"\1^(\2)", text)
    text = re.sub(r"\s*\*\s*", "*", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*\^\s*", "^", text)
    text = re.sub(r"\s*\+\s*", "+", text)
    text = re.sub(r"\s+-\s*", "-", text)
    text = re.sub(r"(?<=\d)(?=(?:sin|cos|tan|cot|sec|csc|ln|log|sqrt)\()", "*", text)
    text = re.sub(r"(?<=\d)\s+(?=(?:sin|cos|tan|cot|sec|csc|ln|log|sqrt)\()", "*", text)
    text = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=[A-Za-z]\()", "*", text)
    text = re.sub(r"\s*,\s*", ",", text) if text[:1] in "([{" and text[-1:] in ")]}" else text
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_answer_inner(inner: str, row: dict[str, Any]) -> str:
    parts = split_top_level_commas(inner)
    normalized = [normalize_answer_atom(part, row) for part in parts]
    return ", ".join(part for part in normalized if part)


def normalize_submission_final_answer(text: str, row: dict[str, Any]) -> tuple[str, bool]:
    boxed = last_boxed_only_string(text)
    inner = remove_boxed(boxed) if boxed else None
    if inner is None:
        return text, False
    normalized_inner = normalize_answer_inner(inner, row)
    if normalized_inner == inner:
        return text, False
    idx = text.rfind(boxed)
    if idx < 0:
        return text, False
    normalized_boxed = f"\\boxed{{{normalized_inner}}}"
    return text[:idx] + normalized_boxed + text[idx + len(boxed) :], True


def final_answer_inner(text: str) -> str:
    boxed = last_boxed_only_string(text)
    return remove_boxed(boxed) or ""


def free_submission_candidate_score(text: str, row: dict[str, Any], valid: bool) -> int:
    inner = final_answer_inner(text)
    parts = split_top_level_commas(inner) if inner else []
    question = str(row.get("question", ""))
    expected = expected_answer_count(row)
    score = 1000 if valid else 0

    if expected is not None:
        score += 120 if len(parts) == expected else -160 * abs(len(parts) - expected)
    if inner:
        score -= 30 * len(_LATEX_ANSWER_CMD_RE.findall(inner))
        score -= 8 * inner.count("\\")
        if ",\\" in inner:
            score -= 80
    if not _ROUNDING_HINT_RE.search(question):
        decimal_digits = [len(match.group(1)) for match in re.finditer(r"\d+\.(\d+)", inner)]
        score += min(sum(decimal_digits), 60)
        score -= 20 * sum(1 for digits in decimal_digits if digits and digits < 3)
    if _EXACT_HINT_RE.search(question) and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", inner.strip()):
        score -= 40
    # Prefer concise valid responses after answer-quality features tie.
    score -= min(len(text) // 1000, 40)
    return score


# ── Internal mode parsing ──────────────────────────────────────────────────

def parse_internal_json(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Try hard to parse a JSON object out of `raw`. Returns (parsed, error)."""
    text = strip_think_tags(raw)
    # Direct attempt.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "answer" in obj:
            return obj, ""
    except json.JSONDecodeError:
        pass
    # Strip ```json fences.
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        obj = json.loads(fenced)
        if isinstance(obj, dict) and "answer" in obj:
            return obj, ""
    except json.JSONDecodeError:
        pass
    # Find the first balanced {...} block.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start : i + 1]
                    try:
                        obj = json.loads(blob)
                        if isinstance(obj, dict) and "answer" in obj:
                            return obj, ""
                    except json.JSONDecodeError:
                        break
                    break
        start = text.find("{", start + 1)
    return None, "could not parse JSON object with `answer` key"


def validate_internal(
    obj: dict[str, Any] | None,
    row: dict[str, Any],
) -> tuple[bool, str]:
    if obj is None:
        return False, "no JSON parsed"
    if set(obj.keys()) != {"answer"}:
        return False, f"unexpected top-level keys: {sorted(obj.keys())}"
    answer = obj["answer"]
    has_options = bool(row.get("options"))
    blanks = n_blanks(str(row.get("question", "")))

    if has_options:
        if not isinstance(answer, str) or not _LETTER_RE.match(answer):
            return False, f"multiple choice expects uppercase letter string, got {answer!r}"
        n_opts = len(row["options"])
        for letter in answer:
            if ord(letter) - 65 >= n_opts:
                return False, f"letter {letter} out of option range (n={n_opts})"
        return True, ""

    if blanks >= 1:
        if not isinstance(answer, list) or not all(isinstance(x, str) for x in answer):
            return False, "fill-in-the-blank expects array of strings"
        if blanks >= 2 and len(answer) != blanks:
            return False, f"expected {blanks} answers, got {len(answer)}"
        return True, ""

    # Free response: string or array.
    if not isinstance(answer, (str, list)):
        return False, "free response expects string or array"
    if isinstance(answer, list) and not all(isinstance(x, str) for x in answer):
        return False, "free-response array must contain only strings"
    return True, ""


# ── Submission mode parsing ────────────────────────────────────────────────

def validate_submission(text: str, row: dict[str, Any]) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty response"
    cleaned = strip_think_tags(text)
    if "<think>" in cleaned.lower():
        return False, "contains hidden <think> tag"
    if not _FINAL_MARKER_RE.search(cleaned):
        return False, "missing `Final answer(s): \\boxed{...}` marker"
    boxed = last_boxed_only_string(cleaned)
    if boxed is None:
        return False, "no \\boxed{...} extractable"
    inner = remove_boxed(boxed) or ""
    if row.get("options"):
        candidate = inner.strip().upper()
        if not _LETTER_RE.match(candidate):
            return False, f"multiple choice expects boxed letter, got {inner!r}"
    return True, ""


def clean_submission_text(text: str) -> str:
    """Strip <think> tags and trim. The boxed final answer must remain."""
    return strip_think_tags(text)


# ════════════════════════════════════════════════════════════════════════════
# vLLM glue
# ════════════════════════════════════════════════════════════════════════════

def render_chat(tokenizer: Any, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def vllm_generate(
    llm: Any,
    tokenizer: Any,
    pairs: list[tuple[str, str]],
    config: SamplingConfig,
    max_tokens: int,
) -> list[list[str]]:
    from vllm import SamplingParams

    if not pairs:
        return []
    prompts = [render_chat(tokenizer, s, u) for s, u in pairs]
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        n=config.n,
    )
    outputs = llm.generate(prompts, sampling_params=sampling)
    return [[choice.text.strip() for choice in out.outputs] for out in outputs]


# ════════════════════════════════════════════════════════════════════════════
# Pipeline
# ════════════════════════════════════════════════════════════════════════════

def build_pairs(
    rows: list[dict[str, Any]],
    mode: Mode,
    include_few_shot: bool,
    math_type: str | None,
) -> list[tuple[str, str]]:
    builder = build_internal_prompt if mode is Mode.INTERNAL else build_submission_prompt
    return [
        builder(row, math_type=math_type, include_few_shot=include_few_shot)
        for row in rows
    ]


def pick_best_internal(
    samples: list[str], row: dict[str, Any]
) -> tuple[str, dict[str, Any] | None, bool, str]:
    """Pick the first valid sample; if none valid, return the first sample."""
    for sample in samples:
        obj, err = parse_internal_json(sample)
        if obj is None:
            continue
        ok, vmsg = validate_internal(obj, row)
        if ok:
            return sample, obj, True, ""
    # Nothing valid — use the first non-empty sample for repair.
    raw = next((s for s in samples if s.strip()), samples[0] if samples else "")
    obj, err = parse_internal_json(raw)
    ok, vmsg = (False, err) if obj is None else validate_internal(obj, row)
    return raw, obj, ok, vmsg or err


def pick_best_submission(
    samples: list[str],
    row: dict[str, Any],
    *,
    normalize_free_final_answers: bool = False,
    rank_free_samples: bool = False,
) -> tuple[str, bool, str, int, bool, list[dict[str, Any]]]:
    """Pick a submission sample.

    Default behavior is the original "first valid sample" policy. When the
    balanced preset enables free-response post-processing, non-option rows get
    final-box ASCII normalization and n>1 samples are ranked by formatting /
    precision signals.
    """
    candidates: list[dict[str, Any]] = []
    free_row = is_free_response_row(row)

    for idx, sample in enumerate(samples):
        cleaned = clean_submission_text(sample)
        normalized = cleaned
        normalized_used = False
        if free_row and normalize_free_final_answers:
            normalized, normalized_used = normalize_submission_final_answer(cleaned, row)
        ok, msg = validate_submission(normalized, row)
        candidates.append(
            {
                "idx": idx,
                "text": normalized,
                "valid": ok,
                "error": "" if ok else msg,
                "normalized": normalized_used,
                "score": free_submission_candidate_score(normalized, row, ok)
                if free_row and rank_free_samples
                else (1000 if ok else 0),
            }
        )

    if not candidates:
        raw = ""
        ok, msg = validate_submission(raw, row)
        return raw, ok, msg, 0, False, []

    if free_row and rank_free_samples:
        best = max(candidates, key=lambda c: (c["score"], c["valid"], -c["idx"]))
        return (
            str(best["text"]),
            bool(best["valid"]),
            "" if best["valid"] else str(best["error"]),
            int(best["idx"]),
            bool(best["normalized"]),
            candidates,
        )

    for candidate in candidates:
        if candidate["valid"]:
            return (
                str(candidate["text"]),
                True,
                "",
                int(candidate["idx"]),
                bool(candidate["normalized"]),
                candidates,
            )

    first = candidates[0]
    return (
        str(first["text"]),
        bool(first["valid"]),
        str(first["error"]),
        int(first["idx"]),
        bool(first["normalized"]),
        candidates,
    )


def run_internal(
    rows: list[dict[str, Any]],
    llm: Any,
    tokenizer: Any,
    config: SamplingConfig,
    max_tokens: int,
    include_few_shot: bool,
    math_type: str | None,
    enable_repair: bool,
) -> list[dict[str, Any]]:
    pairs = build_pairs(rows, Mode.INTERNAL, include_few_shot, math_type)
    print(f"[internal] generating {len(pairs)} prompts (n={config.n}) ...", flush=True)
    all_samples = vllm_generate(llm, tokenizer, pairs, config, max_tokens)

    records: list[dict[str, Any]] = []
    repair_indices: list[int] = []
    for idx, (row, samples) in enumerate(zip(rows, all_samples)):
        raw, obj, valid, err = pick_best_internal(samples, row)
        record = {
            "id": int(row["id"]),
            "raw_response": raw,
            "all_samples": samples if config.n > 1 else None,
            "parsed": obj,
            "valid": valid,
            "validation_error": err,
            "repair_used": False,
        }
        records.append(record)
        if not valid and enable_repair:
            repair_indices.append(idx)

    if repair_indices and enable_repair:
        repair_pairs = [
            build_repair_internal_prompt(rows[i], records[i]["raw_response"])
            for i in repair_indices
        ]
        print(f"[internal] repairing {len(repair_pairs)} invalid rows ...", flush=True)
        repair_samples = vllm_generate(
            llm, tokenizer, repair_pairs,
            CONFIGS["greedy_n1"], max_tokens,
        )
        for idx, samples in zip(repair_indices, repair_samples):
            raw = samples[0] if samples else ""
            obj, err = parse_internal_json(raw)
            ok, msg = (False, err) if obj is None else validate_internal(obj, rows[idx])
            records[idx].update(
                repair_used=True,
                repair_raw=raw,
                parsed=obj,
                valid=ok,
                validation_error="" if ok else (msg or err),
            )
    return records


def run_submission(
    rows: list[dict[str, Any]],
    llm: Any,
    tokenizer: Any,
    config: SamplingConfig,
    max_tokens: int,
    include_few_shot: bool,
    math_type: str | None,
    enable_repair: bool,
    normalize_free_final_answers: bool,
    rank_free_samples: bool,
) -> list[dict[str, Any]]:
    pairs = build_pairs(rows, Mode.SUBMISSION, include_few_shot, math_type)
    print(f"[submission] generating {len(pairs)} prompts (n={config.n}) ...", flush=True)
    all_samples = vllm_generate(llm, tokenizer, pairs, config, max_tokens)

    records: list[dict[str, Any]] = []
    repair_indices: list[int] = []
    for idx, (row, samples) in enumerate(zip(rows, all_samples)):
        chosen, valid, err, chosen_idx, normalized, candidates = pick_best_submission(
            samples,
            row,
            normalize_free_final_answers=normalize_free_final_answers,
            rank_free_samples=rank_free_samples,
        )
        record = {
            "id": int(row["id"]),
            "response": chosen,
            "all_samples": samples if config.n > 1 else None,
            "chosen_idx": chosen_idx if config.n > 1 else None,
            "free_postprocess_used": normalized,
            "candidate_scores": [
                {
                    "idx": c["idx"],
                    "valid": c["valid"],
                    "score": c["score"],
                    "normalized": c["normalized"],
                    "error": c["error"],
                }
                for c in candidates
            ] if config.n > 1 and rank_free_samples and is_free_response_row(row) else None,
            "valid": valid,
            "validation_error": err,
            "repair_used": False,
        }
        records.append(record)
        if not valid and enable_repair:
            repair_indices.append(idx)

    if repair_indices and enable_repair:
        repair_pairs = [
            build_repair_submission_prompt(rows[i], records[i]["response"])
            for i in repair_indices
        ]
        print(f"[submission] repairing {len(repair_pairs)} invalid rows ...", flush=True)
        repair_samples = vllm_generate(
            llm, tokenizer, repair_pairs,
            CONFIGS["greedy_n1"], max_tokens,
        )
        for idx, samples in zip(repair_indices, repair_samples):
            raw = clean_submission_text(samples[0] if samples else "")
            normalized = False
            if is_free_response_row(rows[idx]) and normalize_free_final_answers:
                raw, normalized = normalize_submission_final_answer(raw, rows[idx])
            ok, msg = validate_submission(raw, rows[idx])
            records[idx].update(
                repair_used=True,
                repair_raw=raw,
                repair_free_postprocess_used=normalized,
                response=raw if ok else records[idx]["response"],
                valid=ok,
                validation_error="" if ok else msg,
            )
    return records


# ════════════════════════════════════════════════════════════════════════════
# Output writers
# ════════════════════════════════════════════════════════════════════════════

def write_internal_outputs(out_dir: Path, records: list[dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "internal.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_valid = sum(1 for r in records if r["valid"])
    print(f"  -> {path}  ({n_valid}/{len(records)} valid)", flush=True)
    return path


def write_submission_outputs(out_dir: Path, records: list[dict[str, Any]]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "submission.csv"
    audit_path = out_dir / "submission.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        for r in records:
            writer.writerow({"id": r["id"], "response": r["response"]})

    with audit_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_valid = sum(1 for r in records if r["valid"])
    print(f"  -> {csv_path}", flush=True)
    print(f"  -> {audit_path}  ({n_valid}/{len(records)} valid)", flush=True)
    return csv_path, audit_path


# ════════════════════════════════════════════════════════════════════════════
# Dry-run mode (no model needed)
# ════════════════════════════════════════════════════════════════════════════

def dry_run(rows: list[dict[str, Any]], mode: Mode, include_few_shot: bool, math_type: str | None) -> None:
    pairs = build_pairs(rows, mode, include_few_shot, math_type)
    for row, (system, user) in zip(rows, pairs):
        print("=" * 78)
        print(f"id={row.get('id')}  mode={mode.value}")
        print("─ SYSTEM ─")
        print(system)
        print("─ USER ─")
        print(user)
    print("=" * 78)
    print(f"Built {len(pairs)} prompt pairs (dry-run, no model called).")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=[m.value for m in Mode], required=True,
                        help="Which output mode to run.")
    parser.add_argument("--data-path", default="data/public.jsonl",
                        help="Path to the input JSONL.")
    parser.add_argument("--output-dir", default="results/new_prompts",
                        help="Directory to write outputs into.")
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, only run on the first N rows.")
    parser.add_argument("--config", choices=sorted(CONFIGS), default="greedy_n1",
                        help="Sampling config.")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="max_tokens for generation. Default: 1024 internal / 16384 submission.")
    parser.add_argument("--math-type", default=None,
                        help="Optional math-domain hint (e.g. unit_conversion).")
    parser.add_argument("--include-few-shot", action="store_true",
                        help="Prepend few-shot examples to each prompt.")
    parser.add_argument("--no-repair", action="store_true",
                        help="Disable the one-shot repair pass on invalid rows.")
    parser.add_argument("--normalize-free-final-answers", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Normalize non-option final boxed answers to plain ASCII.")
    parser.add_argument("--rank-free-samples", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="For non-option n>1 runs, choose the best formatted/precision sample.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print built prompts and exit without loading a model.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model id for vLLM.")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = Mode(args.mode)
    rows = load_rows(Path(args.data_path), limit=args.limit)
    if not rows:
        sys.exit(f"No rows loaded from {args.data_path}")
    print(f"Loaded {len(rows)} rows from {args.data_path}", flush=True)

    if args.dry_run:
        dry_run(rows[: min(len(rows), args.limit or 5)], mode, args.include_few_shot, args.math_type)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
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

    config = CONFIGS[args.config]
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = 1024 if mode is Mode.INTERNAL else 16384

    out_dir = Path(args.output_dir)
    if mode is Mode.INTERNAL:
        records = run_internal(
            rows, llm, tokenizer, config, max_tokens,
            args.include_few_shot, args.math_type,
            enable_repair=not args.no_repair,
        )
        write_internal_outputs(out_dir, records)
    else:
        records = run_submission(
            rows, llm, tokenizer, config, max_tokens,
            args.include_few_shot, args.math_type,
            enable_repair=not args.no_repair,
            normalize_free_final_answers=args.normalize_free_final_answers,
            rank_free_samples=args.rank_free_samples,
        )
        write_submission_outputs(out_dir, records)


if __name__ == "__main__":
    main()
