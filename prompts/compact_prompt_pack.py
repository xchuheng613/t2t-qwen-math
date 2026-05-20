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
- For formula/expression/equation blanks, use explicit multiplication and clear powers.
- Follow the notation requested by the problem.
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


CONTINUATION_FALLBACK_SYSTEM = r"""
Continue from the previous reasoning. Do not restart the solution. Use the
previous work to finish the answer.

Output only:
FINAL_ANSWERS:
\boxed{answer}

No explanation. For MCQ, box only uppercase option letter(s). For free-response
with multiple answers, put all answers in one box separated by commas.
""".strip()


BOUNDED_FALLBACK_SYSTEM = r"""
Solve concisely. Do not verify repeatedly. Use at most 8 short reasoning steps,
then finish the answer.

Output only:
FINAL_ANSWERS:
\boxed{answer}

No explanation after the final box. For MCQ, box only uppercase option letter(s).
For free-response with multiple answers, put all answers in one box separated by
commas.
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


def build_continuation_fallback_prompt(
    name: str,
    question: str,
    options: Sequence[object] | None = None,
    raw_response: str = "",
    previous_tail: str = "",
    required_answers: int | None = None,
) -> tuple[str, str]:
    """Build the first retry: continue a truncated solution and finish only."""
    count = required_answers if required_answers is not None else max(1, question.count("[ANS]"))
    if options:
        _lookup(name, MCQ_PROMPTS)
        user = (
            f"Problem:\n{question.strip()}\n\n"
            f"Options:\n{_format_options(options)}\n\n"
            f"Previous response tail:\n{previous_tail.strip() or raw_response.strip()}"
        )
        return CONTINUATION_FALLBACK_SYSTEM, user

    _lookup(name, FREE_PROMPTS)
    user = (
        f"Problem:\n{question.strip()}\n\n"
        f"Required number of answers: {count}\n\n"
        f"Previous response tail:\n{previous_tail.strip() or raw_response.strip()}"
    )
    return "\n\n".join([CONTINUATION_FALLBACK_SYSTEM, _answer_count_line(question)]), user


def build_bounded_fallback_prompt(
    name: str,
    question: str,
    options: Sequence[object] | None = None,
    raw_response: str = "",
    previous_tail: str = "",
    required_answers: int | None = None,
) -> tuple[str, str]:
    """Build the second retry: concise fresh solve with a bounded reasoning budget."""
    count = required_answers if required_answers is not None else max(1, question.count("[ANS]"))
    if options:
        _lookup(name, MCQ_PROMPTS)
        user = f"Problem:\n{question.strip()}\n\nOptions:\n{_format_options(options)}"
        return BOUNDED_FALLBACK_SYSTEM, user

    _lookup(name, FREE_PROMPTS)
    user = (
        f"Problem:\n{question.strip()}\n\n"
        f"Required number of answers: {count}"
    )
    return "\n\n".join([BOUNDED_FALLBACK_SYSTEM, _answer_count_line(question)]), user


# ---------------------------------------------------------------------
# Robust final-answer normalization for submitted answers.
# Do NOT modify judger.py. This code only cleans our model output before
# writing submission.csv.
# ---------------------------------------------------------------------

_FUNC_NAMES = (
    "sin", "cos", "tan", "sec", "csc", "cot",
    "asin", "acos", "atan",
    "ln", "log", "exp", "sqrt",
)

_WORD_ANSWERS = {
    "NONE", "TRUE", "FALSE", "YES", "NO",
    "T", "F", "Y", "N",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
}


def _question_requests_ascii_sqrt(question: str) -> bool:
    q = question.lower()
    return (
        "use sqrt()" in q
        or "sqrt() for the square root" in q
        or "type sqrt" in q
    )


def _looks_like_word_answer(text: str) -> bool:
    t = text.strip()
    if t.upper() in _WORD_ANSWERS:
        return True
    # Avoid changing natural-language labels such as "Tommy" or "origin".
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z ]{1,25}", t)) and not any(
        op in t for op in ("+", "-", "*", "/", "^", "=", "(", ")")
    )


def _looks_expression_like(text: str, question: str = "") -> bool:
    """Conservative detector for expression-style answers."""
    t = text.strip()
    q = question.lower()

    if _looks_like_word_answer(t):
        return False

    expression_keywords = (
        "formula", "expression", "equation", "model", "quotient",
        "composite", "function", "simplify", "factor", "polynomial",
        "derivative", "integral", "write", "solve for",
    )

    if any(k in q for k in expression_keywords):
        return True

    # Contains variables plus mathematical operators.
    if re.search(r"[A-Za-z]", t) and re.search(r"[\+\-\*/\^=]", t):
        return True

    # Function calls or roots/logs/trig.
    if re.search(r"\b(?:sin|cos|tan|sec|csc|cot|ln|log|exp|sqrt)\s*[\(\{]", t):
        return True

    # Powers.
    if "^" in t or "**" in t:
        return True

    return False


def _repair_known_artifacts(text: str) -> str:
    """Repair artifacts introduced by previous normalization paths."""
    text = str(text)

    # Bad artifact from judger.normalize_answer: sqrt(11) -> sqrt{(}11)
    text = text.replace("sqrt{(}", "sqrt(")

    # Sometimes output may contain mismatched sqrt artifact without closing paren.
    text = re.sub(r"sqrt\(\s*([^,\]\}\)\s]+)\s*(?=,|$)", r"sqrt(\1)", text)

    # Normalize common unicode math symbols.
    text = text.replace("−", "-")
    text = text.replace("×", "*")
    text = text.replace("·", "*")

    return text


def _latex_to_ascii_core(text: str) -> str:
    """Convert simple LaTeX syntax to ASCII/Python-ish syntax."""
    text = str(text)

    # Remove harmless LaTeX wrappers.
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")

    # Multiplication / constants / functions.
    text = text.replace("\\cdot", "*")
    text = text.replace("\\times", "*")
    text = text.replace("\\pi", "pi")
    text = text.replace("\\ln", "ln")
    text = text.replace("\\log", "log")
    text = text.replace("\\sin", "sin")
    text = text.replace("\\cos", "cos")
    text = text.replace("\\tan", "tan")
    text = text.replace("\\sec", "sec")
    text = text.replace("\\csc", "csc")
    text = text.replace("\\cot", "cot")

    # Fractions with a sqrt numerator need to be handled before the generic
    # simple-fraction rule because the numerator contains nested braces.
    text = re.sub(
        r"\\(?:dfrac|tfrac|frac)\{\\sqrt\{([^{}]+)\}\}\{([^{}]+)\}",
        r"\\sqrt{\1}/\2",
        text,
    )

    # Fractions: \frac{a}{b} -> (a)/(b)
    # Handles simple non-nested fractions, which covers most final answers.
    text = re.sub(
        r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}",
        r"(\1)/(\2)",
        text,
    )

    # LaTeX sqrt -> ASCII sqrt(...) internally.
    # We will convert sqrt(...) back to \sqrt{...} at final boxing time
    # unless the problem explicitly requires sqrt().
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)

    return text


def _normalize_requested_variable(text: str, question: str = "") -> str:
    """Honor prompts that request a specific variable spelling."""
    q = question.lower()
    if "enter t for the variable" not in q or ("theta" not in q and "\\theta" not in q):
        return text

    func_alt = "|".join(_FUNC_NAMES)

    # Handle function-variable adjacency first: cos\theta -> cos(t).
    text = re.sub(
        rf"\b({func_alt})\\theta\b",
        r"\1(t)",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b({func_alt})\s*theta\b",
        r"\1(t)",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("\\theta", "t")
    text = re.sub(r"\btheta\b", "t", text, flags=re.IGNORECASE)
    text = re.sub(
        rf"\b({func_alt})\s+t\b",
        r"\1(t)",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _normalize_yes_no_answers(text: str, question: str = "") -> str:
    q = question.lower()
    if "yes or no" not in q and "input yes or no" not in q:
        return text

    parts = [part.strip() for part in text.split(",")]
    if not parts:
        return text

    fixed: list[str] = []
    changed = False
    for part in parts:
        low = part.lower()
        if low == "yes":
            fixed.append("Yes")
            changed = changed or part != "Yes"
        elif low == "no":
            fixed.append("No")
            changed = changed or part != "No"
        else:
            fixed.append(part)
    return ", ".join(fixed) if changed else text


def _fix_short_implicit_product(expr: str) -> str:
    """
    Fix very short implicit products inside exponents:
    kt -> k*t, mn -> m*n.
    Avoid doing this to words like theta or alpha.
    """
    expr = expr.strip()
    if re.fullmatch(r"[A-Za-z]{2,3}", expr) and expr.lower() not in _FUNC_NAMES:
        return "*".join(expr)
    return expr


def _normalize_power_syntax(text: str, question: str = "") -> str:
    """
    Normalize common power syntax:
    x^{10} -> x**(10)
    x^10   -> x**(10)
    e**{kt} -> e**(k*t)
    sin^2(x) / sin**2(x) -> sin(x)**2
    """
    if not _looks_expression_like(text, question):
        return text

    # Function powers: sin^2(x), sin**2(x) -> sin(x)**2
    func_alt = "|".join(_FUNC_NAMES)
    text = re.sub(
        rf"\b({func_alt})\s*(?:\^|\*\*)\s*\(?(\d+)\)?\s*\(([^()]*)\)",
        lambda m: f"{m.group(1)}({m.group(3)})**{m.group(2)}",
        text,
        flags=re.IGNORECASE,
    )

    # Function powers without parentheses: cos^2*x -> cos(x)**2
    text = re.sub(
        rf"\b({func_alt})\s*(?:\^|\*\*)\s*(\d+)\s*\*?\s*([A-Za-z])\b",
        lambda m: f"{m.group(1)}({m.group(3)})**{m.group(2)}",
        text,
        flags=re.IGNORECASE,
    )

    # Existing Python power with braces: e**{kt} -> e**(k*t)
    text = re.sub(
        r"\*\*\{([^{}]+)\}",
        lambda m: "**(" + _fix_short_implicit_product(m.group(1)) + ")",
        text,
    )

    # LaTeX braced powers: x^{10} -> x**(10)
    text = re.sub(
        r"\^\{([^{}]+)\}",
        lambda m: "**(" + _fix_short_implicit_product(m.group(1)) + ")",
        text,
    )

    # Parenthesized powers: x^(n+1) -> x**(n+1)
    text = re.sub(r"\^\(([^()]+)\)", r"**(\1)", text)

    # Simple numeric / signed powers: x^-2 -> x**(-2)
    text = re.sub(r"\^(-?\d+(?:\.\d+)?)", r"**(\1)", text)

    # Simple variable powers: r^n -> r**(n)
    text = re.sub(r"\^([A-Za-z]+)", lambda m: "**(" + _fix_short_implicit_product(m.group(1)) + ")", text)

    return text


def _insert_explicit_multiplication(text: str, question: str = "") -> str:
    """
    Insert explicit * in common safe cases:
    7x -> 7*x
    6F -> 6*F
    2pi -> 2*pi
    5(x+2) -> 5*(x+2)
    (x+2)(x+3) -> (x+2)*(x+3)
    76e**(k*t) -> 76*e**(k*t)
    """
    if not _looks_expression_like(text, question):
        return text

    # 5(x+2) -> 5*(x+2)
    text = re.sub(r"(?<=\d)\s*(?=\()", "*", text)

    # (x+2)(x+3) -> (x+2)*(x+3)
    text = re.sub(r"(?<=\))\s*(?=\()", "*", text)

    # (x+2)y -> (x+2)*y
    text = re.sub(r"(?<=\))\s*(?=[A-Za-z])", "*", text)

    # 2sin(x), 2sqrt(3), 2pi, 3ln(x)
    func_alt = "|".join(_FUNC_NAMES + ("pi",))
    text = re.sub(
        rf"(?<=\d)\s*(?=(?:{func_alt})\b)",
        "*",
        text,
        flags=re.IGNORECASE,
    )

    # 6e as Euler's constant, but avoid scientific notation like 1e-5 or 2e10.
    text = re.sub(r"(?<=\d)\s*e\b(?![+-]?\d)", "*e", text)

    # 7x, 6F, 10A. Exclude e/E here to avoid 1e-5.
    text = re.sub(r"(?<=\d)\s*(?=[A-DF-Za-df-z])", "*", text)

    # x sqrt style after conversion is usually handled by )/digit rules,
    # but this catches x sqrt(3) if it appears.
    text = re.sub(
        rf"(?<=[A-Za-z0-9\)])\s+(?=(?:{func_alt})\b)",
        "*",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize spaces around multiplication.
    text = re.sub(r"\s*\*\s*", "*", text)

    return text


def _normalize_function_call_spacing(text: str, question: str = "") -> str:
    """
    Convert common forms:
    ln p -> ln(p)
    log x -> log(x)
    sqrt x -> sqrt(x)
    but avoid changing plain words.
    """
    if not _looks_expression_like(text, question):
        return text

    func_alt = "|".join(_FUNC_NAMES)
    text = re.sub(
        rf"\b({func_alt})\s+([A-Za-z0-9_]+)\b",
        r"\1(\2)",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _ascii_sqrt_to_latex_for_box(text: str) -> str:
    r"""
    Important: boxed sqrt(...) is unsafe with the official judger because
    normalize_answer can turn sqrt(11) into sqrt{(}11.
    Use \sqrt{...} in boxed submissions instead.
    """
    text = str(text)
    out: list[str] = []
    pos = 0
    pattern = re.compile(r"(?<![A-Za-z\\])sqrt\(")
    while True:
        match = pattern.search(text, pos)
        if not match:
            out.append(text[pos:])
            break

        out.append(text[pos : match.start()])
        content_start = match.end()
        depth = 1
        i = content_start
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1

        if depth != 0:
            out.append(text[match.start() :])
            break

        content = text[content_start : i - 1]
        out.append(r"\sqrt{" + _ascii_sqrt_to_latex_for_box(content) + "}")
        pos = i

    return "".join(out)


def repair_sqrt_artifacts(text: str) -> str:
    """Compatibility wrapper for callers that repair sqrt artifacts directly."""
    return _repair_known_artifacts(text)


def ascii_sqrt_to_latex(text: str) -> str:
    """Compatibility wrapper for callers that box ASCII sqrt expressions."""
    return _ascii_sqrt_to_latex_for_box(str(text))


def normalize_expression_style(text: str, question: str = "") -> str:
    """
    Main cleanup hook for answer strings. This returns an internal ASCII-ish
    representation; rebuild_final_response decides final boxed notation.
    """
    text = str(text).strip()
    text = _repair_known_artifacts(text)
    text = _latex_to_ascii_core(text)
    text = _normalize_requested_variable(text, question)
    text = _normalize_yes_no_answers(text, question)
    text = _normalize_function_call_spacing(text, question)
    text = _normalize_power_syntax(text, question)
    text = _insert_explicit_multiplication(text, question)

    # Remove repeated spaces.
    text = re.sub(r"\s+", " ", text).strip()

    # Clean some common accidental spaces around operators.
    text = re.sub(r"\s*([=+\-/,])\s*", r"\1", text)

    return text


def maybe_convert_power_to_python(text: str, question: str = "") -> str:
    """
    Kept for compatibility with existing code. The real work now happens in
    normalize_expression_style.
    """
    return _normalize_power_syntax(text, question)


def clean_answer_text(answer: object, question: str = "") -> str:
    """Cleanup hook used by submission post-processing."""
    text = str(answer).strip()

    # Remove accidental nested box if passed in.
    m = re.fullmatch(r"\\boxed\{(.*)\}", text)
    if m:
        text = m.group(1).strip()

    # Remove obvious leading labels.
    text = re.sub(r"^(?:answer|ans|final answer)\s*[:=]\s*", "", text, flags=re.IGNORECASE)

    text = normalize_expression_style(text, question)

    # Remove harmless trailing period.
    text = text.strip().strip(".")

    return text


def rebuild_final_response(answers: Sequence[object], question: str = "") -> str:
    """
    Rebuild a judge-compatible final answer block from extracted answers.

    Usually we use one boxed comma-separated answer. For explicit sqrt()
    prompts, using unboxed 'answer:' can avoid the official judger's sqrt()
    boxed-normalization bug. Leave this branch off unless public validation
    confirms it helps.
    """
    cleaned = [clean_answer_text(answer, question) for answer in answers]
    joined = ", ".join(cleaned)

    # Default safer boxed output: convert sqrt(...) to \sqrt{...} before
    # boxing to avoid the official judge's sqrt{(} artifact.
    boxed_joined = _ascii_sqrt_to_latex_for_box(joined)
    return f"FINAL_ANSWERS:\n\\boxed{{{boxed_joined}}}"


__all__ = [
    "BASE_SYSTEM",
    "MCQ_SYSTEM",
    "FREE_SYSTEM",
    "SUFFIXES",
    "route_prompt",
    "MCQ_FALLBACK_SYSTEM",
    "FREE_FALLBACK_SYSTEM",
    "CONTINUATION_FALLBACK_SYSTEM",
    "BOUNDED_FALLBACK_SYSTEM",
    "SHORT_FALLBACK_SYSTEM",
    "MCQ_PROMPTS",
    "FREE_PROMPTS",
    "FALLBACK_PROMPTS",
    "build_mcq_prompt",
    "build_free_prompt",
    "build_routed_prompt",
    "build_fallback_prompt",
    "build_continuation_fallback_prompt",
    "build_bounded_fallback_prompt",
    "repair_sqrt_artifacts",
    "ascii_sqrt_to_latex",
    "normalize_expression_style",
    "maybe_convert_power_to_python",
    "clean_answer_text",
    "rebuild_final_response",
]
