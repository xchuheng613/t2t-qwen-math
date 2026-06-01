"""Full-GRPO prompt and reward helpers for the math benchmark.

This module keeps the active compact-prompt behavior, but makes the rollout
prompt shorter and the reward dense enough for GRPO.  The reward functions are
compatible with TRL-style GRPOTrainer call signatures:

    reward(completions, question=[...], answer=[...], options=[...], **kwargs)

They also work with a single example when called directly.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from judger import Judger
from prompts.compact_prompt_pack import clean_answer_text, rebuild_final_response
from utils import last_boxed_only_string, remove_boxed


OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

GRPO_BASE_SYSTEM = r"""
You are solving a math benchmark problem. Solve efficiently. Do not box
anything before the final answer block.
""".strip()

GRPO_MCQ_SYSTEM = r"""
Multiple-choice rules:
- Compare the computed answer with the listed options.
- The final box must contain only the option letter or letters.
- For select-all/check-all, concatenate letters in option order, e.g. BCE.
- Never put option text, units, or explanation inside the final box.

Output exactly:
FINAL_ANSWERS:
\boxed{A}

Stop immediately after the final box.
""".strip()

GRPO_FREE_SYSTEM = r"""
Free-response rules:
- Use only the reasoning needed to get the answer.
- If there are multiple [ANS] blanks, answer every blank in order inside one
  final box, separated by comma + space.
- If one answer is an interval, point, set, list, solution list, or confidence
  interval, keep it grouped with parentheses/brackets.
- No labels, units, [ANS], or explanations inside the box.
- Do not round unless the problem explicitly asks for rounding.
- Preserve requested symbolic form. Use explicit multiplication and powers.

Output exactly:
FINAL_ANSWERS:
\boxed{answer}

Stop immediately after the final box.
""".strip()

REWARD_WEIGHTS = {
    "correctness": 2.0,
    "format": 0.4,
    "length": 0.2,
}

_OPTION_MARKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z])[\.\)]\s+")
_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*([A-Z]+)\s*\}", re.IGNORECASE)
_LETTER_PHRASE_RE = re.compile(
    r"(?:correct\s+option\s+is|correct\s+answer\s+is|answer\s+is|"
    r"therefore|thus|hence|choose|selected|option|choice)"
    r"\s*[:\s]*\(?([A-Z]+)\)?\b",
    re.IGNORECASE,
)
_FINAL_MARKER_RE = re.compile(r"FINAL_ANSWERS\s*:", re.IGNORECASE)
_TOP_LEVEL_BRACKETS = {"(": ")", "[": "]", "{": "}", "<": ">"}

_JUDGER: Judger | None = None


def _judger() -> Judger:
    global _JUDGER
    if _JUDGER is None:
        _JUDGER = Judger(strict_extract=False)
    return _JUDGER


def answer_values(answer: Any) -> list[str]:
    if answer is None:
        return []
    values = answer if isinstance(answer, (list, tuple)) else [answer]
    return ["" if value is None else str(value).strip() for value in values]


def expected_answer_count(question: str, answer: Any = None) -> int:
    blanks = str(question or "").count("[ANS]")
    if blanks > 0:
        return blanks
    values = [value for value in answer_values(answer) if value != ""]
    return max(1, len(values))


def has_inline_options(question: str) -> bool:
    markers = [match.group(1) for match in _OPTION_MARKER_RE.finditer(str(question or ""))]
    return "A" in markers and "B" in markers


def is_mcq_like(question: str, options: Any = None, answer: Any = None) -> bool:
    if bool(options):
        return True
    values = answer_values(answer)
    all_letters = bool(values) and all(re.fullmatch(r"[A-Z]+", value.upper()) for value in values)
    return all_letters and has_inline_options(question)


def _format_options(options: Any) -> str:
    if not options:
        return ""
    if isinstance(options, Mapping):
        lines = [f"{key}. {str(options[key]).strip()}" for key in sorted(options)]
        return "\n".join(lines)
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        return "\n".join(
            f"{OPTION_LETTERS[idx] if idx < len(OPTION_LETTERS) else idx + 1}. {str(option).strip()}"
            for idx, option in enumerate(options)
        )
    return str(options).strip()


def _answer_count_line(question: str) -> str:
    count = str(question or "").count("[ANS]")
    if count == 0:
        return "This problem has no [ANS] marker. Return exactly one final answer."
    if count == 1:
        return "This problem has 1 [ANS] blank. Return exactly 1 answer."
    return f"This problem has {count} [ANS] blanks. Return exactly {count} answers."


def build_grpo_prompt(question: str, options: Any = None) -> tuple[str, str]:
    """Return a compact ``(system, user)`` pair for GRPO rollouts."""
    if options:
        system = "\n\n".join([GRPO_BASE_SYSTEM, GRPO_MCQ_SYSTEM])
    else:
        system = "\n\n".join([GRPO_BASE_SYSTEM, _answer_count_line(question), GRPO_FREE_SYSTEM])

    user = str(question).strip()
    option_block = _format_options(options)
    if option_block:
        user = f"{user}\n\nOptions:\n{option_block}"
    return system, user


def build_grpo_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build chat messages for a dataset row with ``question`` and optional ``options``."""
    question = str(row.get("question", row.get("prompt", row.get("instruction", "")))).strip()
    if not question:
        raise ValueError("row is missing question/prompt/instruction")
    system, user = build_grpo_prompt(question, row.get("options") or None)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def make_grpo_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a benchmark row into a GRPOTrainer-friendly row."""
    question = str(row.get("question", row.get("prompt", row.get("instruction", "")))).strip()
    answer = row.get("answer", row.get("gold", row.get("output")))
    options = row.get("options") or None
    return {
        "id": row.get("id"),
        "prompt": build_grpo_messages(row),
        "question": question,
        "options": options,
        "answer": answer_values(answer),
        "expected_answer_count": expected_answer_count(question, answer),
        "is_mcq": is_mcq_like(question, options, answer),
    }


def make_grpo_dataset_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [make_grpo_row(row) for row in rows]


def completion_to_text(completion: Any) -> str:
    """Handle plain strings and chat-message completions from TRL."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        content = completion.get("content")
        return "" if content is None else str(content)
    if isinstance(completion, Sequence) and not isinstance(completion, (str, bytes)):
        messages = [item for item in completion if isinstance(item, Mapping)]
        if messages:
            for message in reversed(messages):
                if message.get("role") == "assistant" and message.get("content") is not None:
                    return str(message["content"])
            content = messages[-1].get("content")
            return "" if content is None else str(content)
        return "".join(str(item) for item in completion)
    return str(completion)


def split_top_level_commas(expr: str) -> list[str]:
    parts: list[str] = []
    stack: list[str] = []
    start = 0
    i = 0
    text = str(expr)
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text):
            i += 2
            continue
        if char in _TOP_LEVEL_BRACKETS:
            stack.append(_TOP_LEVEL_BRACKETS[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == "," and not stack:
            parts.append(text[start:i].strip())
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts or ([text.strip()] if text.strip() else [])


def boxed_contents(text: str) -> list[str]:
    contents: list[str] = []
    source = str(text)
    start = 0
    while True:
        idx = source.find(r"\boxed{", start)
        if idx < 0:
            return contents
        pos = idx + len(r"\boxed{")
        depth = 1
        chars: list[str] = []
        while pos < len(source) and depth > 0:
            char = source[pos]
            if char == "{":
                depth += 1
                chars.append(char)
            elif char == "}":
                depth -= 1
                if depth > 0:
                    chars.append(char)
            else:
                chars.append(char)
            pos += 1
        if depth == 0:
            contents.append("".join(chars).strip())
            start = pos
        else:
            return contents


def final_answer_inner(text: str) -> str:
    boxed = last_boxed_only_string(str(text))
    return (remove_boxed(boxed) or "").strip() if boxed else ""


def _tail_after_think(text: str) -> str:
    idx = str(text).rfind("</think>")
    return str(text)[idx + len("</think>") :] if idx >= 0 else str(text)


def _valid_letters(options: Any = None) -> set[str]:
    if isinstance(options, Mapping):
        return {str(key).strip().upper() for key in options}
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        return set(OPTION_LETTERS[: len(options)])
    return set(OPTION_LETTERS)


def extract_mcq_letter(text: str, options: Any = None) -> str:
    """Extract final MCQ letters. Also maps a boxed option value when possible."""
    tail = _tail_after_think(text)
    valid = _valid_letters(options)
    inner = final_answer_inner(tail) or final_answer_inner(text)
    compact_inner = re.sub(r"\s+", "", inner).upper()
    if compact_inner and re.fullmatch(r"[A-Z]+", compact_inner):
        if set(compact_inner) <= valid:
            return compact_inner

    match = _BOXED_LETTER_RE.search(tail) or _BOXED_LETTER_RE.search(text)
    if match:
        letters = match.group(1).upper()
        if set(letters) <= valid:
            return letters

    if inner and options:
        judger = _judger()
        try:
            candidate_norm = judger.norm_ans_str(inner)
        except Exception:
            candidate_norm = inner
        option_items = list(options.values()) if isinstance(options, Mapping) else list(options or [])
        for idx, option in enumerate(option_items):
            option_text = str(option).strip()
            try:
                option_norm = judger.norm_ans_str(option_text)
                if candidate_norm == option_norm or judger.is_equal(candidate_norm, option_norm):
                    return OPTION_LETTERS[idx]
            except Exception:
                if inner.strip() == option_text:
                    return OPTION_LETTERS[idx]

    phrase_matches = list(_LETTER_PHRASE_RE.finditer(tail))
    if phrase_matches:
        letters = phrase_matches[-1].group(1).upper()
        if set(letters) <= valid:
            return letters
    return ""


def _normalized_response_for_judge(text: str, question: str) -> str:
    inner = final_answer_inner(text)
    if not inner:
        return text
    answers = split_top_level_commas(inner)
    if not answers:
        return text
    return rebuild_final_response(answers, question)


def _shape_ok(text: str, question: str, options: Any, answer: Any, expected_count: int | None) -> bool:
    inner = final_answer_inner(text)
    if not inner:
        return False
    if is_mcq_like(question, options, answer):
        letters = re.sub(r"\s+", "", inner).upper()
        return bool(re.fullmatch(r"[A-Z]+", letters)) and set(letters) <= _valid_letters(options)
    count = expected_count or expected_answer_count(question, answer)
    return len(split_top_level_commas(inner)) == count


def _format_score(text: str, question: str, options: Any, answer: Any, expected_count: int | None) -> float:
    source = str(text).strip()
    if not source:
        return 0.0

    marker_matches = list(_FINAL_MARKER_RE.finditer(source))
    marker_start = marker_matches[-1].start() if marker_matches else -1
    final_region = source[marker_start:] if marker_start >= 0 else source
    pre_region = source[:marker_start] if marker_start >= 0 else ""
    final_boxes = boxed_contents(final_region)
    all_boxes = boxed_contents(source)

    score = 0.0
    if marker_matches:
        score += 0.20
    if len(final_boxes) == 1:
        score += 0.25
    if all_boxes and len(all_boxes) == len(final_boxes):
        score += 0.10
    if (marker_matches or final_boxes) and not boxed_contents(pre_region):
        score += 0.10

    boxed = last_boxed_only_string(source)
    if boxed:
        tail_idx = source.rfind(boxed) + len(boxed)
        if source[tail_idx:].strip() == "":
            score += 0.15

    if _shape_ok(source, question, options, answer, expected_count):
        score += 0.20
    return max(0.0, min(1.0, score))


def _length_score(text: str, question: str = "") -> float:
    source = str(text).strip()
    if not source:
        return 0.0
    length = len(source)
    target = 4500
    soft_cap = 9000
    hard_cap = 14000
    if length <= target:
        score = 1.0
    elif length <= soft_cap:
        score = 1.0 - 0.4 * ((length - target) / (soft_cap - target))
    elif length <= hard_cap:
        score = 0.6 - 0.6 * ((length - soft_cap) / (hard_cap - soft_cap))
    else:
        score = 0.0

    lowered = source.lower()
    if lowered.count("wait") >= 4 or lowered.count("let me check") >= 3:
        score *= 0.7
    if final_answer_inner(source) == "":
        score = min(score, 0.25)
    return max(0.0, min(1.0, score))


def _correctness_score(text: str, question: str, options: Any, answer: Any) -> float:
    gold = answer_values(answer)
    if not gold or any(value == "" for value in gold):
        return 0.0

    if is_mcq_like(question, options, answer) and len(gold) == 1:
        letters = extract_mcq_letter(text, options)
        return 1.0 if letters and letters == gold[0].upper().replace(" ", "") else 0.0

    pred = _normalized_response_for_judge(text, question)
    inner = final_answer_inner(pred)
    pred_parts = split_top_level_commas(inner)
    if len(pred_parts) == len(gold):
        pred_clean = [clean_answer_text(value, question) for value in pred_parts]
        gold_clean = [clean_answer_text(value, question) for value in gold]
        if pred_clean == gold_clean:
            return 1.0

    try:
        ok = _judger().auto_judge(pred=pred, gold=gold, options=[[]] * len(gold))
        return 1.0 if ok else 0.0
    except Exception:
        return 0.0


def _is_string_batch(value: Any, n: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == n and all(
        item is None or isinstance(item, str) for item in value
    )


def _is_sequence_batch(value: Any, n: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == n and (
        not value
        or all(item is None or isinstance(item, (list, tuple, dict)) for item in value)
    )


def _select(value: Any, index: int, is_batch: bool) -> Any:
    return value[index] if is_batch else value


def _reward_examples(
    completions: Sequence[Any],
    *,
    question: Any = None,
    options: Any = None,
    answer: Any = None,
    expected_answer_count: Any = None,
    **_: Any,
) -> list[tuple[str, str, Any, Any, int | None]]:
    n = len(completions)
    question_batch = _is_string_batch(question, n)
    options_batch = question_batch or _is_sequence_batch(options, n)
    answer_batch = question_batch or _is_sequence_batch(answer, n)
    count_batch = isinstance(expected_answer_count, (list, tuple)) and len(expected_answer_count) == n

    examples: list[tuple[str, str, Any, Any, int | None]] = []
    for idx, completion in enumerate(completions):
        q = _select(question, idx, question_batch)
        opts = _select(options, idx, options_batch)
        ans = _select(answer, idx, answer_batch)
        count = _select(expected_answer_count, idx, count_batch)
        try:
            count_int = None if count is None else int(count)
        except Exception:
            count_int = None
        examples.append((completion_to_text(completion), str(q or ""), opts, ans, count_int))
    return examples


def _resolve_completions(completions: Sequence[Any] | None, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Sequence[Any]:
    # TRL versions commonly call reward(completions, **columns), but some
    # examples use reward(prompts, completions, **columns). Support both.
    if args:
        return args[0]
    if completions is None and "completions" in kwargs:
        return kwargs.pop("completions")
    return [] if completions is None else completions


def correctness_reward(completions: Sequence[Any] | None = None, *args: Any, **kwargs: Any) -> list[float]:
    """Sparse exact-answer reward, aligned with the local benchmark judger."""
    resolved = _resolve_completions(completions, args, kwargs)
    return [
        _correctness_score(text, question, options, answer)
        for text, question, options, answer, _count in _reward_examples(resolved, **kwargs)
    ]


def format_reward(completions: Sequence[Any] | None = None, *args: Any, **kwargs: Any) -> list[float]:
    """Dense formatting reward for final boxed answer structure."""
    resolved = _resolve_completions(completions, args, kwargs)
    return [
        _format_score(text, question, options, answer, count)
        for text, question, options, answer, count in _reward_examples(resolved, **kwargs)
    ]


def length_reward(completions: Sequence[Any] | None = None, *args: Any, **kwargs: Any) -> list[float]:
    """Dense reward that discourages overthinking and truncation-prone outputs."""
    resolved = _resolve_completions(completions, args, kwargs)
    return [
        _length_score(text, question)
        for text, question, _options, _answer, _count in _reward_examples(resolved, **kwargs)
    ]


def full_grpo_reward(completions: Sequence[Any] | None = None, *args: Any, **kwargs: Any) -> list[float]:
    """Single weighted reward for direct use in GRPOTrainer."""
    resolved = _resolve_completions(completions, args, kwargs)
    rewards: list[float] = []
    for text, question, options, answer, count in _reward_examples(resolved, **kwargs):
        correct = _correctness_score(text, question, options, answer)
        fmt = _format_score(text, question, options, answer, count)
        length = _length_score(text, question)
        rewards.append(
            REWARD_WEIGHTS["correctness"] * correct
            + REWARD_WEIGHTS["format"] * fmt
            + REWARD_WEIGHTS["length"] * length
        )
    return rewards


def reward_breakdown(completion: Any, row: Mapping[str, Any]) -> dict[str, float]:
    """Debug one completion against one source row."""
    question = str(row.get("question", row.get("prompt", row.get("instruction", ""))))
    options = row.get("options") or None
    answer = row.get("answer", row.get("gold", row.get("output")))
    count = expected_answer_count(question, answer)
    text = completion_to_text(completion)
    correct = _correctness_score(text, question, options, answer)
    fmt = _format_score(text, question, options, answer, count)
    length = _length_score(text, question)
    return {
        "correctness": correct,
        "format": fmt,
        "length": length,
        "full_grpo": (
            REWARD_WEIGHTS["correctness"] * correct
            + REWARD_WEIGHTS["format"] * fmt
            + REWARD_WEIGHTS["length"] * length
        ),
    }


__all__ = [
    "GRPO_BASE_SYSTEM",
    "GRPO_MCQ_SYSTEM",
    "GRPO_FREE_SYSTEM",
    "REWARD_WEIGHTS",
    "build_grpo_prompt",
    "build_grpo_messages",
    "make_grpo_row",
    "make_grpo_dataset_rows",
    "completion_to_text",
    "final_answer_inner",
    "boxed_contents",
    "split_top_level_commas",
    "correctness_reward",
    "format_reward",
    "length_reward",
    "full_grpo_reward",
    "reward_breakdown",
]
