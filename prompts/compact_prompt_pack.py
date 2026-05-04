"""Compact judge-compatible solver prompt pack.

The prompts in this module prioritize reliable final-answer extraction over
long explanations.  ``route_prompt`` returns a ``(system, user)`` pair suitable
for chat-template based runners.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


BASE_SYSTEM = r"""
You are solving a math benchmark problem. Solve efficiently and keep visible
work short. Do not box anything before the final answer block.
""".strip()


MCQ_SYSTEM = r"""
Choose the correct option. Keep reasoning short.

Final answer rules:
- Output exactly one final \boxed{}.
- The box must contain only the option letter.
- For select-all/check-all, use combined letters with no commas, e.g. BCE.
- Never box the option value or explanation.
- If unsure, still choose the best option and finish.

Output format:
FINAL_ANSWERS:
\boxed{A}
""".strip()


FREE_SYSTEM = r"""
Solve the problem. Keep reasoning short.

Final answer rules:
- Output exactly one final \boxed{}.
- If there are multiple [ANS] blanks, put answers in the same box separated by commas.
- Answer every [ANS] blank in order; do not add extra answers.
- If one answer is a point, interval, set, list, solution list, or confidence interval, wrap it in parentheses/brackets.
- No units, labels, [ANS], explanations, or words inside the box.
- Only round if the problem explicitly says to round.
- Never apply standard rounding. For money, percent, units, statistics, and trig values, keep 10-15 significant digits unless rounding is stated.
- For formula/expression/equation blanks, use ASCII: *, **, sqrt(...), pi, ln(...). Do not use LaTeX.
- If asked for a formula, model, equation, quotient, or composite function, output that expression, not just a simplified number.
- Preserve the requested form; do not expand or substitute unless asked.

Output format:
FINAL_ANSWERS:
\boxed{answer1, answer2, answer3}
""".strip()


SUFFIXES = {
    "FORMULA_COMPOSITE_SUFFIX": (
        "Expression task: keep the requested symbolic form. Do not replace it "
        "with only a number."
    ),
    "PERCENTILE_SUFFIX": (
        "Percentile task: use the introductory-statistics rank method unless "
        "another method is stated."
    ),
    "STATS_SOFTWARE_SUFFIX": (
        "Software/statistics task: give the requested numerical outputs with "
        "the shown precision style; do not invent extra outputs."
    ),
    "SELECTION_MATCHING_SUFFIX": (
        "This is a letter-answer or matching problem. Do not write option text. "
        "For several blanks, put the letters inside the one final box separated "
        "by commas. For select all that apply in one blank, output concatenated "
        "letters in option order, e.g. BCEG."
    ),
    "MCQ_ALGORITHM_SUFFIX": (
        "For sequence/algorithm questions, compare the proposed output lists "
        "and choose the matching option letter."
    ),
}


MCQ_FALLBACK_SYSTEM = r"""
/no_think
Choose the best option.

Output only:
FINAL_ANSWERS:
\boxed{A}
""".strip()

FREE_FALLBACK_SYSTEM = r"""
/no_think
Rewrite only the final answers. Do not recompute.

Output exactly one \boxed{}.
If multiple answers are needed, separate them by commas inside the box.
No labels, units, or explanation.

FINAL_ANSWERS:
\boxed{answer1, answer2}
""".strip()


SHORT_FALLBACK_SYSTEM = r"""
/no_think
Answer directly.

Output exactly one final \boxed{}.
For multiple answers, separate them by commas inside the box.
For MCQ, box only the uppercase option letter(s).

FINAL_ANSWERS:
\boxed{answer}
""".strip()


# Legacy runner compatibility.  These names let scripts such as
# ``prompt_sweep.py`` and ``create_submission.py --routing-mode legacy`` load
# this module via ``--prompt-module prompts.compact_prompt_pack``.
MCQ_PROMPTS = [("compact", MCQ_SYSTEM, None)]
FREE_PROMPTS = [("compact", FREE_SYSTEM, None)]
FALLBACK_PROMPTS = {"compact": SHORT_FALLBACK_SYSTEM}


_SELECTION_RE = re.compile(
    r"\b("
    r"embedded\s+[A-Z]/[A-Z]|matching|match|select|select\s+all|best\s+response|"
    r"conclusion\s+is|which\s+statement|which\s+of|choice|choices?|options?"
    r")\b|(?<![A-Za-z0-9])A[\.\)]\s+.*(?<![A-Za-z0-9])B[\.\)]\s+",
    re.IGNORECASE | re.DOTALL,
)
_FORMULA_TASK_RE = re.compile(
    r"\b("
    r"formula|model|equation|quotient|composite(?:\s+function)?|"
    r"expression|function\s+rule|write\s+an?\s+equation"
    r")\b|N\(T\(t\)\)|P\(c\)",
    re.IGNORECASE,
)
_PERCENTILE_RE = re.compile(
    r"\bpercentile\b|\bP_?\d+\b",
    re.IGNORECASE,
)
_STATS_SOFTWARE_RE = re.compile(
    r"\b("
    r"use\s+software|software|chi-?square|chi-?squared|regression|"
    r"correlation|p-?value"
    r")\b",
    re.IGNORECASE,
)
_MCQ_ALGORITHM_RE = re.compile(
    r"\b(sequence|algorithm|output\s+list|oeis|recurrence|iteration|program)\b",
    re.IGNORECASE,
)


def _format_options(options: Sequence[object]) -> str:
    return "\n".join(
        f"{chr(65 + idx)}. {str(option).strip()}"
        for idx, option in enumerate(options)
    )


def _choose_suffix(question: str) -> str | None:
    if _FORMULA_TASK_RE.search(question):
        return "FORMULA_COMPOSITE_SUFFIX"
    if _PERCENTILE_RE.search(question):
        return "PERCENTILE_SUFFIX"
    if _STATS_SOFTWARE_RE.search(question):
        return "STATS_SOFTWARE_SUFFIX"
    if _SELECTION_RE.search(question):
        return "SELECTION_MATCHING_SUFFIX"
    return None


def _answer_count_line(question: str) -> str:
    ans_count = question.count("[ANS]")
    if ans_count == 1:
        return "This problem has 1 [ANS] blank. Return exactly 1 answer."
    if ans_count > 1:
        return f"This problem has {ans_count} [ANS] blanks. Return exactly {ans_count} answers."
    return "No [ANS] marker appears. Return exactly one final answer."


def route_prompt(question: str, options: Sequence[object] | None = None) -> tuple[str, str]:
    """Route a problem to the compact MCQ/free prompt plus one broad suffix."""
    has_options = bool(options)
    suffix_name = _choose_suffix(question)
    if has_options:
        parts = [BASE_SYSTEM, MCQ_SYSTEM]
        if _MCQ_ALGORITHM_RE.search(question):
            parts.append(SUFFIXES["MCQ_ALGORITHM_SUFFIX"])
    else:
        parts = [BASE_SYSTEM, _answer_count_line(question), FREE_SYSTEM]
        if suffix_name:
            parts.append(SUFFIXES[suffix_name])
    system = "\n\n".join(parts)

    user = question.strip()
    if has_options:
        user = f"{user}\n\nOptions:\n{_format_options(options or [])}"
    return system, user


def _lookup(prompt_name: str, prompts: Sequence[tuple[str, str, object]]) -> None:
    names = {name for name, *_ in prompts}
    if prompt_name not in names:
        raise KeyError(f"Unknown compact prompt name: {prompt_name}")


def build_mcq_prompt(name: str, question: str, options: Sequence[object]) -> tuple[str, str]:
    """Legacy-compatible MCQ prompt builder."""
    _lookup(name, MCQ_PROMPTS)
    return route_prompt(question, options)


def build_free_prompt(name: str, question: str) -> tuple[str, str]:
    """Legacy-compatible free-response prompt builder."""
    _lookup(name, FREE_PROMPTS)
    return route_prompt(question, None)


def build_routed_prompt(
    name: str,
    question: str,
    options: Sequence[object] | None = None,
    fallback: bool = False,
) -> tuple[str, str]:
    """Legacy-compatible routed prompt builder."""
    if fallback:
        return build_fallback_prompt(name, question, options)

    if options:
        _lookup(name, MCQ_PROMPTS)
    else:
        _lookup(name, FREE_PROMPTS)
    return route_prompt(question, options)


def build_fallback_prompt(
    name: str,
    question: str,
    options: Sequence[object] | None = None,
    raw_response: str = "",
    required_answers: int | None = None,
) -> tuple[str, str]:
    """Build a short no-think fallback prompt for format failures."""
    if options:
        _lookup(name, MCQ_PROMPTS)
        user = f"Problem:\n{question.strip()}\n\nOptions:\n{_format_options(options)}"
        if raw_response.strip():
            user += f"\n\nPrevious response:\n{raw_response.strip()}"
        return MCQ_FALLBACK_SYSTEM, user

    _lookup(name, FREE_PROMPTS)
    count = required_answers if required_answers is not None else max(1, question.count("[ANS]"))
    user = (
        f"Problem:\n{question.strip()}\n\n"
        f"Required number of answers: {count}"
    )
    if raw_response.strip():
        user += f"\n\nPrevious response:\n{raw_response.strip()}"
    return "\n\n".join([FREE_FALLBACK_SYSTEM, _answer_count_line(question)]), user


def normalize_expression_style(text: str) -> str:
    """Normalize common LaTeX-ish expression notation to ASCII."""
    text = str(text)
    text = text.replace("\\cdot", "*")
    text = text.replace("\\times", "*")
    text = text.replace("\\pi", "pi")
    text = text.replace("\\ln", "ln")
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\(?:dfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    # Avoid corrupting scientific notation such as 1e-5.
    text = re.sub(r"(?<=\d)(?=[A-DF-Za-df-z])", "*", text)
    text = re.sub(r"(?<=\))(?=[a-zA-Z])", "*", text)
    return text.strip()


def maybe_convert_power_to_python(text: str, question: str = "") -> str:
    q = question.lower()
    expression_task = any(
        key in q
        for key in ("formula", "expression", "equation", "model", "quotient", "composite", "function")
    )
    has_variable = bool(re.search(r"[a-zA-Z]", text))
    if expression_task and has_variable:
        return text.replace("^", "**")
    return text


def clean_answer_text(answer: object, question: str = "") -> str:
    """Small cleanup hook used by submission post-processing."""
    text = str(answer).strip()
    text = re.sub(r"^(?:answer|ans)\s*[:=]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = normalize_expression_style(text)
    text = maybe_convert_power_to_python(text, question)
    return text.strip().strip(".")


def rebuild_final_response(answers: Sequence[object], question: str = "") -> str:
    """Rebuild a judge-compatible final answer block from extracted answers."""
    joined = ", ".join(clean_answer_text(answer, question) for answer in answers)
    return f"FINAL_ANSWERS:\n\\boxed{{{joined}}}"


__all__ = [
    "BASE_SYSTEM",
    "MCQ_SYSTEM",
    "FREE_SYSTEM",
    "SUFFIXES",
    "route_prompt",
    "MCQ_FALLBACK_SYSTEM",
    "FREE_FALLBACK_SYSTEM",
    "SHORT_FALLBACK_SYSTEM",
    "MCQ_PROMPTS",
    "FREE_PROMPTS",
    "FALLBACK_PROMPTS",
    "build_mcq_prompt",
    "build_free_prompt",
    "build_routed_prompt",
    "build_fallback_prompt",
    "normalize_expression_style",
    "maybe_convert_power_to_python",
    "clean_answer_text",
    "rebuild_final_response",
]
