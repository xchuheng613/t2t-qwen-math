"""Compact judge-compatible solver prompt pack.

The prompts in this module prioritize reliable final-answer extraction over
long explanations.  ``route_prompt`` returns a ``(system, user)`` pair suitable
for chat-template based runners.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


BASE_SYSTEM = r"""
You are solving a math benchmark problem. Solve efficiently, keep visible work
brief, and verify only the final format.

Final output must end with one contiguous final block:
FINAL_ANSWERS:
\boxed{answer1}
\boxed{answer2}

Use one boxed answer per requested blank/value, in order. For one answer, use
one box. For a single [ANS] that asks for a point, interval, list, set, vector,
or multiple-select choice, keep that whole answer in one box.

Do not box anything before the final answer block. Do not put labels, units,
reasoning, option text, or commentary inside boxes unless the blank specifically
asks for an equation. If using a thinking model, all final boxes must appear
after </think>. Nothing should appear after the final boxed answer.

Prefer exact forms unless decimals are requested. If no rounding instruction is
given and a decimal is necessary, use 12-15 significant digits. Use parseable
notation: 5/8, sqrt(3), pi, e^x, exp(x), sin(x), cos(x), tan(x), log(x). Use *
for multiplication when helpful, e.g. 3*x+2. Use standard interval notation
like (a,b), [a,b], (a,\infty), and \cup for unions. Use parentheses for ordered
pairs, vectors, and lists: (x,y), (a,b,c). For unordered multiple roots in one
blank, use a comma-separated list inside parentheses if needed.
""".strip()


MCQ_SYSTEM = r"""
This is a multiple-choice problem with an explicit options list. Solve the
problem and select the correct option. Output only the option letter(s), not the
option text.

For single-choice, output one uppercase letter such as A. For multiple-select,
output concatenated uppercase letters in option order, such as BCE.

Required final format:
FINAL_ANSWERS:
\boxed{A}
""".strip()


FREE_SYSTEM = r"""
This is a free-response problem. Solve efficiently. Determine how many final
answers are required from the [ANS] blanks or the final question wording.
Output one boxed answer per blank/value in order.

If a blank asks for a choice letter, output only the letter. If a blank asks for
a conclusion with choices A/B/C, output only A, B, or C. If there is no [ANS],
infer the final requested value and output one boxed answer unless the problem
clearly asks for multiple final values.

Required final format:
FINAL_ANSWERS:
\boxed{answer1}
\boxed{answer2}
""".strip()


SUFFIXES = {
    "GENERAL_ALGEBRA_SUFFIX": (
        "Use direct computation/symbolic simplification. Return the simplified "
        "value or expression only. Do not include variable labels unless the "
        "blank itself asks for an equation."
    ),
    "CALCULUS_SUFFIX": (
        "Compute the requested derivative/integral/limit/ODE/optimization "
        "result. Prefer exact symbolic form when reasonable; otherwise use a "
        "high-precision decimal. For optimization, return the requested "
        "variable value(s) and objective value(s) in the order asked."
    ),
    "GEOMETRY_TRIG_SUFFIX": (
        "Track units internally but omit units in the box. For coordinates, "
        "vectors, angles, and intervals, preserve the requested format. For "
        "trig general solutions, include the requested period/parameter form "
        "only if the problem asks for it."
    ),
    "STATS_PROB_SUFFIX": (
        "For hypothesis tests or confidence intervals, return statistics, "
        "critical values, p-values, and final conclusion letters exactly in "
        "the order asked. If conclusion options are A/B/C, box only the "
        "letter. Follow requested rounding carefully."
    ),
    "COMBINATORICS_NUMBER_THEORY_SUFFIX": (
        "Solve structurally and avoid brute-force-looking rambling. Return "
        "the final integer, fraction, expression, or requested set only. If "
        "the problem asks for a guarantee/minimum/maximum, output that value "
        "only."
    ),
    "SELECTION_MATCHING_SUFFIX": (
        "This is a letter-answer or matching problem. Do not write option text. "
        "For several blanks, output one letter per box. For select all that "
        "apply in one blank, output concatenated letters in option order, e.g. "
        "BCEG."
    ),
}


SHORT_FALLBACK_SYSTEM = r"""
Answer directly with no reasoning.

Final output only:
FINAL_ANSWERS:
\boxed{answer}

Use one boxed answer per requested blank/value, in order. For MCQ, box only the
uppercase option letter(s). No labels, units, explanations, or text after the
boxes.
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
_STATS_RE = re.compile(
    r"\b("
    r"hypothesis|confidence|sample|population|p-?value|z\s*-?\s*score|"
    r"t\s*-?\s*score|normal|binomial|poisson|regression|probability|"
    r"statistic|variance|standard\s+deviation|mean|median"
    r")\b",
    re.IGNORECASE,
)
_CALCULUS_RE = re.compile(
    r"\b("
    r"integral|integrate|derivative|differentiate|limit|differential\s+equation|"
    r"optimization|optimize|maximize|minimize|series|taylor|maclaurin"
    r")\b|\\int|\\lim|\\frac\{d",
    re.IGNORECASE,
)
_GEOMETRY_RE = re.compile(
    r"\b("
    r"triangle|circle|angle|area|volume|vector|coordinate|polygon|radius|"
    r"diameter|perimeter|sin|cos|tan|trigonometric|geometry"
    r")\b|\\sin|\\cos|\\tan",
    re.IGNORECASE,
)
_COMBO_NT_RE = re.compile(
    r"\b("
    r"integer|prime|divisible|mod|modulo|remainder|sequence|combinatorics|"
    r"counting|game|graph|permutation|combination|factor|gcd|lcm"
    r")\b",
    re.IGNORECASE,
)


def _format_options(options: Sequence[object]) -> str:
    return "\n".join(
        f"{chr(65 + idx)}. {str(option).strip()}"
        for idx, option in enumerate(options)
    )


def _choose_suffix(question: str) -> str:
    if _SELECTION_RE.search(question):
        return "SELECTION_MATCHING_SUFFIX"
    if _STATS_RE.search(question):
        return "STATS_PROB_SUFFIX"
    if _CALCULUS_RE.search(question):
        return "CALCULUS_SUFFIX"
    if _GEOMETRY_RE.search(question):
        return "GEOMETRY_TRIG_SUFFIX"
    if _COMBO_NT_RE.search(question):
        return "COMBINATORICS_NUMBER_THEORY_SUFFIX"
    return "GENERAL_ALGEBRA_SUFFIX"


def route_prompt(question: str, options: Sequence[object] | None = None) -> tuple[str, str]:
    """Route a problem to the compact MCQ/free prompt plus one broad suffix."""
    has_options = bool(options)
    suffix_name = _choose_suffix(question)
    system = "\n\n".join(
        [
            BASE_SYSTEM,
            MCQ_SYSTEM if has_options else FREE_SYSTEM,
            SUFFIXES[suffix_name],
        ]
    )

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
        user = question.strip()
        if options:
            user = f"{user}\n\nOptions:\n{_format_options(options)}"
        return FALLBACK_PROMPTS.get(name, SHORT_FALLBACK_SYSTEM), user + "\n\n/no_think"

    if options:
        _lookup(name, MCQ_PROMPTS)
    else:
        _lookup(name, FREE_PROMPTS)
    return route_prompt(question, options)


def clean_answer_text(answer: object) -> str:
    """Small cleanup hook used by submission post-processing."""
    text = str(answer).strip()
    text = re.sub(r"^(?:answer|ans)\s*[:=]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip(".")


def rebuild_final_response(answers: Sequence[object]) -> str:
    """Rebuild a judge-compatible final answer block from extracted answers."""
    boxes = "\n".join(f"\\boxed{{{clean_answer_text(answer)}}}" for answer in answers)
    return "FINAL_ANSWERS:\n" + boxes


__all__ = [
    "BASE_SYSTEM",
    "MCQ_SYSTEM",
    "FREE_SYSTEM",
    "SUFFIXES",
    "route_prompt",
    "SHORT_FALLBACK_SYSTEM",
    "MCQ_PROMPTS",
    "FREE_PROMPTS",
    "FALLBACK_PROMPTS",
    "build_mcq_prompt",
    "build_free_prompt",
    "build_routed_prompt",
    "clean_answer_text",
    "rebuild_final_response",
]
