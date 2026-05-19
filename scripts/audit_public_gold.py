#!/usr/bin/env python3
"""Audit public gold answers before using them for LoRA SFT.

The audit is intentionally conservative. It assigns each public row to:

* KEEP: clean enough for the first public-only SFT run
* REVIEW: suspicious; save for teacher verification
* DROP: likely format-noisy or contradictory; exclude from first SFT

It also writes deterministic free-response-only KEEP train/dev/holdout splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import signal
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from judger import Judger
from prompts.compact_prompt_pack import rebuild_final_response


LETTER_RE = re.compile(r"^[A-J]+$")
OPTION_MARKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-J])[\.\)]\s+")
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
ORDINAL_PLACES = {
    "whole": 0,
    "whole number": 0,
    "integer": 0,
    "tenth": 1,
    "tenths": 1,
    "hundredth": 2,
    "hundredths": 2,
    "thousandth": 3,
    "thousandths": 3,
    "ten-thousandth": 4,
    "ten-thousandths": 4,
    "millionth": 6,
    "millionths": 6,
}
DROP_FLAGS = {
    "answer_count_mismatch",
    "rounding_conflict",
    "exact_decimal_conflict",
    "decimal_exact_conflict",
    "mcq_answer_invalid",
    "empty_answer",
    "judge_count_mismatch",
}


@dataclass
class AuditResult:
    row: dict[str, Any]
    status: str
    flags: list[str]
    notes: list[str]
    row_format: str
    ans_count: int
    expected_count: int
    gold_count: int
    extracted_count: int | None
    judge_self_ok: bool | None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_values(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer")
    values = answer if isinstance(answer, list) else [answer]
    return ["" if value is None else str(value).strip() for value in values]


def has_inline_options(question: str) -> bool:
    markers = [match.group(1) for match in OPTION_MARKER_RE.finditer(question)]
    return "A" in markers and "B" in markers


def is_all_letter_gold(row: dict[str, Any]) -> bool:
    values = answer_values(row)
    return bool(values) and all(bool(LETTER_RE.fullmatch(value.upper())) for value in values)


def is_mcq_like(row: dict[str, Any]) -> bool:
    return bool(row.get("options")) or (
        is_all_letter_gold(row) and has_inline_options(str(row.get("question", "")))
    )


def row_format(row: dict[str, Any]) -> str:
    if row.get("options"):
        return "mcq_options"
    if is_mcq_like(row):
        return "mcq_inline"
    return "free"


def expected_answer_count(row: dict[str, Any]) -> int:
    return max(1, str(row.get("question", "")).count("[ANS]"))


def context_has_at_least(text: str, start: int) -> bool:
    return "at least" in text[max(0, start - 35) : start + 5]


def number_word_to_int(text: str) -> int | None:
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    return NUMBER_WORDS.get(text)


def rounding_requirements(question: str) -> list[tuple[int, str]]:
    """Return exact requested decimal places from explicit round/nearest wording."""
    q = question.lower()
    requirements: list[tuple[int, str]] = []

    decimal_phrase = r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
    for pattern in [
        rf"round(?:\s+your\s+answer)?\s+to\s+{decimal_phrase}\s+decimal\s+places?",
        rf"rounded\s+to\s+{decimal_phrase}\s+decimal\s+places?",
        rf"correct\s+to\s+{decimal_phrase}\s+decimal\s+places?",
    ]:
        for match in re.finditer(pattern, q):
            if context_has_at_least(q, match.start()):
                continue
            places = number_word_to_int(match.group(1))
            if places is not None:
                requirements.append((places, match.group(0)))

    nearest_patterns = [
        r"nearest\s+(whole\s+number|integer|tenth|tenths|hundredth|hundredths|thousandth|thousandths|ten-thousandth|ten-thousandths|millionth|millionths)",
        r"nearest\s+(cent)",
        r"nearest\s+(0\.\d+)",
    ]
    for pattern in nearest_patterns:
        for match in re.finditer(pattern, q):
            if context_has_at_least(q, match.start()):
                continue
            unit = match.group(1)
            if unit == "cent":
                places = 2
            elif unit.startswith("0."):
                places = len(unit.split(".", 1)[1].rstrip("0"))
            else:
                places = ORDINAL_PLACES[unit]
            requirements.append((places, match.group(0)))

    return requirements


def decimal_places_if_plain_number(answer: str) -> tuple[int, bool] | None:
    text = answer.strip().replace(",", "")
    text = text.strip("$%")
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:e[-+]?\d+)?", text, flags=re.I):
        mantissa = re.split(r"e", text, flags=re.I)[0]
        decimals = mantissa.split(".", 1)[1]
        return len(decimals), bool(decimals.rstrip("0"))
    if re.fullmatch(r"[-+]?\d+(?:e[-+]?\d+)?", text, flags=re.I):
        return 0, False
    return None


def answer_violates_rounding(answer: str, places: int) -> bool:
    parsed = decimal_places_if_plain_number(answer)
    if parsed is None:
        return False
    decimals, has_nonzero_fraction = parsed
    if places == 0:
        return decimals > 0 and has_nonzero_fraction
    if decimals <= places:
        return False
    extra = answer.strip().replace(",", "").strip("$%").split(".", 1)[1]
    extra = re.split(r"e", extra, flags=re.I)[0][places:]
    return bool(extra.rstrip("0"))


def is_plain_decimal(answer: str) -> bool:
    parsed = decimal_places_if_plain_number(answer)
    return parsed is not None and "." in answer


def looks_symbolic(answer: str) -> bool:
    text = answer.strip()
    if not text:
        return False
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text.replace(",", "")):
        return False
    symbolic_tokens = ["\\frac", "sqrt", "\\sqrt", "pi", "\\pi", "ln", "log", "exp", "^", "/"]
    if any(token in text for token in symbolic_tokens):
        return True
    return bool(re.search(r"[a-zA-Z]", text)) and not LETTER_RE.fullmatch(text.upper())


def exact_requested(question: str) -> bool:
    q = question.lower()
    phrases = [
        "do not approximate",
        "don't approximate",
        "not approximate",
        "exact value",
        "exact answer",
        "exact solution",
        "exact form",
        "give exact",
        "decimal answers are not allowed",
        "decimals are not allowed",
        "no decimals",
        "may contain no decimals",
        "must contain no decimals",
    ]
    return any(phrase in q for phrase in phrases)


def decimal_required(question: str) -> bool:
    q = question.lower()
    if "decimal or fraction" in q or "fraction or decimal" in q:
        return False
    phrases = [
        "answer must be a decimal",
        "must be a decimal",
        "as a decimal",
        "in decimal form",
        "write as a decimal",
        "express as a decimal",
    ]
    return any(phrase in q for phrase in phrases)


def suspicious_answer_flags(answers: list[str]) -> list[str]:
    flags: list[str] = []
    unit_pattern = re.compile(
        r"[-+]?\d+(?:\.\d+)?\s*(?:degrees?|radians?|feet|foot|ft|inches?|"
        r"hours?|hrs?|minutes?|mins?|seconds?|secs?|cm|mm|km|meters?|liters?|"
        r"miles?|dollars?|fahrenheit|celsius|kelvin|percent)\b",
        re.I,
    )
    for answer in answers:
        low = answer.lower()
        if answer == "":
            flags.append("empty_answer")
        if len(answer) > 240 or "\n" in answer:
            flags.append("very_long_answer")
        if any(word in low for word in ["because", "therefore", "hence", " so "]):
            flags.append("sentence_answer")
        if re.search(r"<[^>]+>", answer) or "\\begin" in answer or "\\end" in answer:
            flags.append("latex_or_html_garbage")
        if unit_pattern.search(answer) or "\\text{" in answer:
            flags.append("unit_in_answer")
        if answer.count("$") >= 2:
            flags.append("latex_or_html_garbage")
    return sorted(set(flags))


def mcq_flags(row: dict[str, Any], answers: list[str]) -> list[str]:
    if not is_mcq_like(row):
        return []
    options = row.get("options") or []
    max_letter = chr(ord("A") + len(options) - 1) if options else "J"
    valid = set(chr(code) for code in range(ord("A"), ord(max_letter) + 1))
    for answer in answers:
        text = answer.strip().upper().replace(" ", "")
        if not text or not LETTER_RE.fullmatch(text) or any(letter not in valid for letter in text):
            return ["mcq_answer_invalid"]
    return []


def format_conflict_flags(row: dict[str, Any], answers: list[str]) -> list[str]:
    question = str(row.get("question", ""))
    flags: list[str] = []

    for places, phrase in rounding_requirements(question):
        offenders = [answer for answer in answers if answer_violates_rounding(answer, places)]
        if offenders:
            flags.append("rounding_conflict")
            break

    if exact_requested(question) and any(is_plain_decimal(answer) for answer in answers):
        flags.append("exact_decimal_conflict")

    if decimal_required(question):
        symbolic = [answer for answer in answers if looks_symbolic(answer)]
        if len(answers) == 1 and symbolic:
            flags.append("decimal_exact_conflict")
        elif symbolic:
            flags.append("decimal_exact_conflict_possible")

    return flags


def judge_self_check(
    judger: Judger,
    row: dict[str, Any],
    answers: list[str],
    timeout_seconds: int,
) -> tuple[list[str], list[str], int | None, bool | None]:
    flags: list[str] = []
    notes: list[str] = []
    response = rebuild_final_response(answers, str(row.get("question", "")))
    expected = len(answers)

    try:
        extracted = judger.extract_ans(response)
        split = judger.split_by_comma(extracted) if extracted else []
        extracted_count = len(split)
    except Exception as exc:
        flags.append("judge_extract_exception")
        notes.append(f"judge_extract_exception={type(exc).__name__}")
        return flags, notes, None, None

    if extracted_count != expected:
        flags.append("judge_count_mismatch")
        notes.append(f"judge_extracted_count={extracted_count}")
        return flags, notes, extracted_count, False

    def handler(signum: int, frame: Any) -> None:
        raise TimeoutError("judge self-check timed out")

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_seconds)
        ok = bool(judger.auto_judge(pred=response, gold=answers, options=[[]] * len(answers)))
    except Exception as exc:
        ok = False
        notes.append(f"judge_self_exception={type(exc).__name__}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if not ok:
        flags.append("judge_self_check_failed")
    return flags, notes, extracted_count, ok


def audit_row(
    judger: Judger | None,
    row: dict[str, Any],
    timeout_seconds: int,
    use_judge_self_check: bool = True,
) -> AuditResult:
    answers = answer_values(row)
    question = str(row.get("question", ""))
    expected = expected_answer_count(row)
    flags: list[str] = []
    notes: list[str] = []
    fmt = row_format(row)

    if not is_mcq_like(row) and len(answers) != expected:
        flags.append("answer_count_mismatch")
        notes.append(f"expected={expected}")

    flags.extend(mcq_flags(row, answers))
    flags.extend(suspicious_answer_flags(answers))
    flags.extend(format_conflict_flags(row, answers))
    extracted_count: int | None = None
    judge_ok: bool | None = None
    if use_judge_self_check:
        if judger is None:
            raise RuntimeError("judge self-check requested but Judger is unavailable")
        judge_flags, judge_notes, extracted_count, judge_ok = judge_self_check(
            judger, row, answers, timeout_seconds
        )
        flags.extend(judge_flags)
        notes.extend(judge_notes)

    flags = sorted(set(flags))
    if any(flag in DROP_FLAGS for flag in flags):
        status = "DROP"
    elif flags:
        status = "REVIEW"
    else:
        status = "KEEP"

    return AuditResult(
        row=row,
        status=status,
        flags=flags,
        notes=notes,
        row_format=fmt,
        ans_count=question.count("[ANS]"),
        expected_count=expected,
        gold_count=len(answers),
        extracted_count=extracted_count,
        judge_self_ok=judge_ok,
    )


def stratum(row: dict[str, Any]) -> tuple[int, str]:
    question = str(row.get("question", ""))
    n_ans = min(question.count("[ANS]"), 8)
    table_like = n_ans >= 6 or "table" in question.lower() or "\\begin{array}" in question
    return n_ans, "table" if table_like else "normal"


def split_rows(
    rows: list[dict[str, Any]],
    seed: int,
    train_frac: float,
    dev_frac: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)

    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for group_rows in groups.values():
        rng.shuffle(group_rows)
        n = len(group_rows)
        n_train = int(train_frac * n)
        n_dev = int(dev_frac * n)
        train.extend(group_rows[:n_train])
        dev.extend(group_rows[n_train : n_train + n_dev])
        holdout.extend(group_rows[n_train + n_dev :])

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(holdout)
    return train, dev, holdout


def write_quality_csv(path: Path, results: list[AuditResult]) -> None:
    fields = [
        "id",
        "status",
        "format",
        "flags",
        "notes",
        "ans_count",
        "expected_count",
        "gold_count",
        "judge_extracted_count",
        "judge_self_ok",
        "answer",
        "question",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: int(item.row["id"])):
            writer.writerow(
                {
                    "id": result.row.get("id"),
                    "status": result.status,
                    "format": result.row_format,
                    "flags": "|".join(result.flags),
                    "notes": "|".join(result.notes),
                    "ans_count": result.ans_count,
                    "expected_count": result.expected_count,
                    "gold_count": result.gold_count,
                    "judge_extracted_count": (
                        "" if result.extracted_count is None else result.extracted_count
                    ),
                    "judge_self_ok": (
                        "" if result.judge_self_ok is None else int(result.judge_self_ok)
                    ),
                    "answer": json.dumps(result.row.get("answer"), ensure_ascii=False),
                    "question": str(result.row.get("question", "")).replace("\n", "\\n"),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-path", type=Path, default=Path("data/public.jsonl"))
    parser.add_argument("--quality-csv", type=Path, default=Path("data/public_quality.csv"))
    parser.add_argument("--keep-path", type=Path, default=Path("data/public_train_keep.jsonl"))
    parser.add_argument("--review-path", type=Path, default=Path("data/public_train_review.jsonl"))
    parser.add_argument("--drop-path", type=Path, default=Path("data/public_train_drop.jsonl"))
    parser.add_argument(
        "--free-split-prefix",
        type=Path,
        default=Path("data/public"),
        help="Prefix for raw free-response KEEP splits: _train/dev/holdout.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    parser.add_argument("--judge-timeout-seconds", type=int, default=5)
    parser.add_argument(
        "--skip-judge-self-check",
        action="store_true",
        help="Only use regex/count filters. Intended for quick debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.public_path)
    judger: Judger | None = None
    if not args.skip_judge_self_check:
        try:
            judger = Judger(strict_extract=False)
        except ImportError as exc:
            raise SystemExit(
                "Could not initialize Judger. Install the repo requirements, especially "
                "antlr4-python3-runtime==4.11.1, or rerun with --skip-judge-self-check."
            ) from exc

    results: list[AuditResult] = []
    for row in rows:
        result = audit_row(
            judger,
            row,
            timeout_seconds=args.judge_timeout_seconds,
            use_judge_self_check=not args.skip_judge_self_check,
        )
        results.append(result)

    keep = [result.row for result in results if result.status == "KEEP"]
    review = [result.row for result in results if result.status == "REVIEW"]
    drop = [result.row for result in results if result.status == "DROP"]

    write_quality_csv(args.quality_csv, results)
    write_jsonl(args.keep_path, keep)
    write_jsonl(args.review_path, review)
    write_jsonl(args.drop_path, drop)

    free_keep = [result.row for result in results if result.status == "KEEP" and result.row_format == "free"]
    train, dev, holdout = split_rows(
        free_keep,
        seed=args.seed,
        train_frac=args.train_frac,
        dev_frac=args.dev_frac,
    )
    write_jsonl(args.free_split_prefix.with_name(args.free_split_prefix.name + "_train.jsonl"), train)
    write_jsonl(args.free_split_prefix.with_name(args.free_split_prefix.name + "_dev.jsonl"), dev)
    write_jsonl(args.free_split_prefix.with_name(args.free_split_prefix.name + "_holdout.jsonl"), holdout)

    status_counts = Counter(result.status for result in results)
    format_counts = Counter(result.row_format for result in results)
    flag_counts = Counter(flag for result in results for flag in result.flags)
    print(f"source={args.public_path} rows={len(rows)}")
    print(
        "status "
        + " ".join(f"{name.lower()}={status_counts.get(name, 0)}" for name in ["KEEP", "REVIEW", "DROP"])
    )
    print("format " + " ".join(f"{key}={value}" for key, value in sorted(format_counts.items())))
    print(f"free_keep_split train={len(train)} dev={len(dev)} holdout={len(holdout)}")
    if flag_counts:
        top_flags = ", ".join(f"{flag}={count}" for flag, count in flag_counts.most_common(12))
        print(f"top_flags {top_flags}")
    print(f"quality_csv={args.quality_csv}")
    print(f"keep={args.keep_path} review={args.review_path} drop={args.drop_path}")


if __name__ == "__main__":
    main()
