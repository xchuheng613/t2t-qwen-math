"""Updated prompt variants with routed free-response suffixes.

This module intentionally leaves :mod:`prompt_variants` untouched. It exposes
the same prompt-builder API so sweep/submission scripts can opt into this file
with ``--prompt-module prompt_variant_updated``.
"""

from __future__ import annotations

import re

import prompt_variants as legacy


# Reuse the existing MCQ prompt families. The requested changes target
# free-response behavior.
MCQ_PROMPTS = legacy.MCQ_PROMPTS
FEW_SHOT_MCQ = legacy.FEW_SHOT_MCQ
FEW_SHOT_FREE = legacy.FEW_SHOT_FREE


FREE_FINAL_RULES_V2 = (
    "Final output only:\n"
    "FINAL_ANSWERS:\n"
    "\\boxed{answer1}\n"
    "\\boxed{answer2}\n"
    "\n"
    "Rules: one box per actual requested answer, in order. "
    "If duplicate adjacent [ANS] markers refer to one yes/no or conclusion answer, answer it once. "
    "If one answer is a point, interval, list, set, or confidence interval, keep it in one box. "
    "No labels, units, [ANS], or explanations inside boxes. "
    "Do not round unless asked; if no rounding is stated, use 12-15 significant digits. "
    "If the problem says 'at least' N digits, give more digits, not fewer. "
    "Do not pad exact integers with .000 unless fixed decimal places are explicitly required. "
    "Use ASCII: sqrt(...), pi, ln(...), exp(...), *, infinity."
)


FREE_SUFFIXES = {
    "numeric": (
        "Answer style: high-precision decimal unless rounding is explicitly requested."
    ),
    "exact": (
        "Answer style: exact ASCII form. Prefer sqrt(...), pi, ln(...), atan(...), "
        "powers, or fractions over decimals."
    ),
    "symbolic_then_numeric": (
        "Some blanks ask for symbolic formulas and later blanks ask for numbers. "
        "Keep symbolic blanks in variables; substitute numbers only for numeric blanks. "
        "If earlier blanks ask for a formula, do not substitute given numbers into those blanks."
    ),
    "multi_blank": (
        "Fill requested answers in order. Do not add extra conclusion answers."
    ),
    "single_object": (
        "This is one answer object. Keep all comma-separated values inside one boxed object."
    ),
    "table": (
        "Fill table entries in row order. Do not pad exact integers with trailing .000 "
        "unless fixed decimals are required."
    ),
    "bernstein": (
        "For Bernstein polynomial of degree n and index k, use C(n,k)*t^k*(1-t)^(n-k). "
        "Use the stated ordinal as k."
    ),
    "trig_general": (
        "For trig general solutions, use exact atan(...), asin(...), acos(...), and pi "
        "for periods unless the blank explicitly asks for decimals."
    ),
}


SYS_FREE_BASELINE = (
    "You are an expert mathematician. Solve the problem carefully. "
    "Reason as needed, then write only the final answer block.\n\n"
    + FREE_FINAL_RULES_V2
)

SYS_FREE_STRICT_FORMAT = (
    "You are an expert mathematician. Solve rigorously and verify the requested answer format.\n\n"
    + FREE_FINAL_RULES_V2
)

SYS_FREE_VERIFY = (
    "You are an expert mathematician. Solve the problem, then verify by substitution, "
    "re-derivation, or a magnitude/sign check before writing the final block.\n\n"
    + FREE_FINAL_RULES_V2
)

SYS_FREE_CONCISE = (
    "You are an expert mathematician. Solve efficiently and keep visible output minimal.\n\n"
    + FREE_FINAL_RULES_V2
)

SYS_FREE_STATISTICS = (
    "You are an expert in statistics and probability. Track whether the problem asks for a "
    "probability, test statistic, p-value, interval, hypothesis conclusion, or option letter.\n\n"
    + FREE_FINAL_RULES_V2
)

SYS_FREE_CALCULUS = (
    "You are an expert in calculus. Match exact vs decimal form to the problem's wording, "
    "not merely to whether constants like pi or sqrt appear.\n\n"
    + FREE_FINAL_RULES_V2
)


SHORT_FINAL_RULES_V2 = (
    "Output only:\n"
    "FINAL_ANSWERS:\n"
    "\\boxed{answer}\n"
    "One box per actual requested answer. No reasoning, labels, units, [ANS], or text after boxes. "
    "Do not round unless requested. Do not pad exact integers with .000 unless fixed decimals are required. "
    "Use plain ASCII."
)


FALLBACK_PROMPTS = {
    "baseline": "Answer directly with no reasoning. " + SHORT_FINAL_RULES_V2,
    "routed_v2": "Answer directly with no reasoning. " + SHORT_FINAL_RULES_V2,
    "general_mcq_eliminate": legacy.FALLBACK_PROMPTS["general_mcq_eliminate"],
    "algorithm_sequence_mcq": legacy.FALLBACK_PROMPTS["algorithm_sequence_mcq"],
    "statistics_prompt": "No reasoning. Give the unrounded statistics answer unless rounding is explicit. "
    + SHORT_FINAL_RULES_V2,
    "calculus_prompt": "No reasoning. Match exact vs decimal form to the wording. " + SHORT_FINAL_RULES_V2,
    "multi_blank_free_response": "No reasoning. Count actual requested answers and output them in order. "
    + SHORT_FINAL_RULES_V2,
}


# Each entry: (name, system_prompt, few_shot_block_or_None)
FREE_PROMPTS = [
    ("baseline", SYS_FREE_BASELINE, None),
    ("routed_v2", SYS_FREE_BASELINE, None),
    ("strict_format", SYS_FREE_STRICT_FORMAT, None),
    ("verify", SYS_FREE_VERIFY, None),
    ("concise", SYS_FREE_CONCISE, None),
    ("statistics_prompt", SYS_FREE_STATISTICS, None),
    ("calculus_prompt", SYS_FREE_CALCULUS, None),
    ("multi_blank_free_response", SYS_FREE_BASELINE, None),
    ("multi_answer", SYS_FREE_BASELINE, None),
    ("few_shot", SYS_FREE_BASELINE, FEW_SHOT_FREE),
    ("strict_few_shot", SYS_FREE_STRICT_FORMAT, FEW_SHOT_FREE),
    ("numeric", SYS_FREE_BASELINE, None),
    ("exact", SYS_FREE_BASELINE, None),
    ("symbolic_then_numeric", SYS_FREE_BASELINE, None),
    ("multi_blank", SYS_FREE_BASELINE, None),
    ("single_object", SYS_FREE_BASELINE, None),
    ("table", SYS_FREE_BASELINE, None),
    ("bernstein", SYS_FREE_BASELINE, None),
    ("trig_general", SYS_FREE_CALCULUS, None),
]


def _lookup(prompts: list, name: str):
    for prompt_name, system_prompt, few_shot in prompts:
        if prompt_name == name:
            return system_prompt, few_shot
    raise KeyError(f"Unknown prompt name: {name}")


def route_free_suffix(question: str) -> str:
    q = question.lower()
    ans_count = question.count("[ANS]")

    if "bernstein polynomial" in q:
        return "bernstein"

    if any(k in q for k in ("theta=[ans]+[ans] n", "theta = [ans]+[ans] n", "general solution")) and any(
        fn in q for fn in ("tan(", "sin(", "cos(", "trig", "theta")
    ):
        return "trig_general"

    if ans_count == 1 and any(
        k in q
        for k in (
            "confidence interval",
            "interval",
            "ordered pair",
            "point",
            "coordinate",
            "solutions",
            "separated by commas",
        )
    ):
        return "single_object"

    if any(
        k in q
        for k in (
            "simplified to the form",
            "contain no fractions",
            "a/b",
            "where a and b",
            "where a=",
            "where b=",
        )
    ):
        return "symbolic_then_numeric"

    if re.search(r"\bph\s*(?:=|\[|is\b|of\b|$)", q):
        if "must be a decimal" in q or "answer must be a decimal" in q:
            return "numeric"
        return "exact"

    if any(
        k in q
        for k in (
            "exact value",
            "formula",
            "expression",
            "in terms of",
            "general solution",
            "theta=",
            "tan(",
            "sin(",
            "cos(",
            "half-life",
            "fraction remaining",
            "decays by",
        )
    ):
        if "must be a decimal" in q or "answer must be a decimal" in q:
            return "numeric"
        return "exact"

    if "table" in q or "\\begin{array}" in q or "\\begin{tabular}" in q or ans_count >= 6:
        return "table"

    if ans_count > 1:
        return "multi_blank"

    return "numeric"


def _suffix_for_prompt(name: str, question: str) -> str:
    suffix_name = name if name in FREE_SUFFIXES else route_free_suffix(question)
    suffix = FREE_SUFFIXES[suffix_name]

    # Keep the explicit multi-blank prompt forceful even when the router would
    # pick a numeric/exact suffix.
    if name in {"multi_blank_free_response", "multi_answer"} and suffix_name != "multi_blank":
        suffix = suffix + " " + FREE_SUFFIXES["multi_blank"]
    return suffix


def clean_answer_text(s: str) -> str:
    s = s.strip()

    # Common malformed sqrt artifacts.
    s = re.sub(r"sqrt\{\(?\}([A-Za-z0-9.+\-*/^ ]+)\)", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)

    # Fractions and powers from common LaTeX final-answer syntax.
    s = re.sub(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    s = re.sub(r"([A-Za-z0-9_)])\^\{([^{}]+)\}", r"\1^(\2)", s)

    # Constants / functions.
    s = s.replace("\\pi", "pi")
    s = s.replace("\\infty", "infinity")
    s = s.replace("∞", "infinity")
    s = s.replace("\\ln", "ln")
    s = s.replace("\\cos", "cos")
    s = s.replace("\\sin", "sin")
    s = s.replace("\\tan", "tan")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")

    # Remove obvious labels.
    s = re.sub(r"^(answer|final answer|ans)\s*[:=]\s*", "", s, flags=re.I)

    # Normalize spaces.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def rebuild_final_response(answers: list[str]) -> str:
    boxes = "\n".join(f"\\boxed{{{clean_answer_text(answer)}}}" for answer in answers)
    return "FINAL_ANSWERS:\n" + boxes


def build_mcq_prompt(name: str, question: str, options: list) -> tuple[str, str]:
    return legacy.build_mcq_prompt(name, question, options)


def build_free_prompt(name: str, question: str) -> tuple[str, str]:
    system_prompt, few_shot = _lookup(FREE_PROMPTS, name)
    system_prompt = system_prompt + "\n\n" + _suffix_for_prompt(name, question)
    user = question
    if few_shot:
        user = few_shot + user
    return system_prompt, user


def build_routed_prompt(
    name: str,
    question: str,
    options: list | None = None,
    fallback: bool = False,
) -> tuple[str, str]:
    """Build a prompt family selected by a caller-side router."""
    if fallback:
        system_prompt = FALLBACK_PROMPTS.get(name, FALLBACK_PROMPTS["baseline"])
        few_shot = None
    elif options:
        system_prompt, few_shot = _lookup(MCQ_PROMPTS, name)
    else:
        system_prompt, few_shot = _lookup(FREE_PROMPTS, name)
        system_prompt = system_prompt + "\n\n" + _suffix_for_prompt(name, question)

    if options:
        labels = [chr(65 + idx) for idx in range(len(options))]
        opts_text = "\n".join(f"{label}. {str(option).strip()}" for label, option in zip(labels, options))
        user = f"{question}\n\nOptions:\n{opts_text}"
    else:
        user = question
    if few_shot:
        user = few_shot + user
    if fallback:
        user = user + "\n\n/no_think"
    return system_prompt, user
