"""Prompt variants for the sweep. MCQ and free-form prompts are independent
lists so they can be swept separately. Edit either list freely."""


# ── Shared final-answer rules ───────────────────────────────────────────────
FINAL_ANSWER_RULES = (
    "Final-answer formatting is mandatory:\n"
    "  - After solving, output only this final answer block and nothing after it:\n"
    "    FINAL_ANSWERS:\n"
    "    \\boxed{answer1}\n"
    "    \\boxed{answer2}\n"
    "  - Use one \\boxed{...} per [ANS] blank, in order. For one blank, use one box.\n"
    "  - Do not put labels, units, [ANS], explanations, or reasoning inside the boxes.\n"
    "  - Keep intervals, ordered pairs, coordinate pairs, and confidence intervals inside one box, "
    "for example \\boxed{(-8,infinity)} or \\boxed{(34.9245581634078,42.2754418365922)}.\n"
    "  - Do not split on commas inside an interval or ordered pair.\n"
    "  - Do not round unless the problem explicitly asks for rounding. If no rounding is specified, "
    "keep 12-15 significant digits for decimal answers.\n"
    "  - Preserve exact symbolic form when the problem asks for a formula, expression, exact answer, "
    "simplification, answer in terms of constants, or uses ln, e^, sqrt, pi, or similar notation.\n"
    "  - Prefer calculator/WebWork-style notation when possible: e^(16*x), sqrt(11), ln(0.5), "
    "10^6, (-8,infinity). Use * between multiplied terms."
)

SHORT_FINAL_RULES = (
    "Output only:\n"
    "FINAL_ANSWERS:\n"
    "\\boxed{answer}\n"
    "Use one box per [ANS] blank. No reasoning, labels, units, [ANS], or text after the boxes. "
    "Do not round unless requested. Use * for multiplication and WebWork-style functions."
)


# ── Free-form system prompts ────────────────────────────────────────────────
SYS_FREE_BASELINE = (
    "You are an expert mathematician. Solve the problem carefully. "
    "Reason as needed, but keep the visible final response clean.\n\n"
    + FINAL_ANSWER_RULES
)

SYS_FREE_STRICT_FORMAT = (
    "You are an expert mathematician. Solve the problem rigorously and verify the requested answer format.\n\n"
    + FINAL_ANSWER_RULES
)

SYS_FREE_VERIFY = (
    "You are an expert mathematician. Solve the problem, then verify by substitution, re-derivation, "
    "or a magnitude/sign check before writing the final block.\n\n"
    + FINAL_ANSWER_RULES
)

SYS_FREE_CONCISE = (
    "You are an expert mathematician. Solve efficiently and avoid unnecessary visible explanation.\n\n"
    + FINAL_ANSWER_RULES
)

SYS_FREE_MULTI_ANSWER = (
    "You are an expert mathematician. First count the [ANS] blanks or requested outputs, then solve. "
    "For select-all blanks, put the combined capital letters in that blank's box with no spaces or commas. "
    "For True/False blanks, output T or F when the problem asks for T/F; if the blank gives lettered "
    "True/False choices, output the corresponding option letter. Use one box per statement.\n\n"
    + FINAL_ANSWER_RULES
)

SYS_FREE_STATISTICS = (
    "You are an expert in statistics and probability. Track whether the problem asks for a probability, "
    "test statistic, p-value, interval, hypothesis conclusion, or option letter. Do not round z/t values, "
    "interval endpoints, proportions, money, or rates unless a rounding rule is stated.\n\n"
    + FINAL_ANSWER_RULES
)

SYS_FREE_CALCULUS = (
    "You are an expert in calculus. Preserve exact expressions unless a decimal approximation is requested. "
    "Use calculator/WebWork-style notation for functions and explicit multiplication signs.\n\n"
    + FINAL_ANSWER_RULES
)


# ── MCQ system prompts ──────────────────────────────────────────────────────
SYS_MCQ_BASELINE = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the best answer. "
    "If the problem says select all, check all, or there may be more than one answer, output the combined "
    "capital letters only, e.g. BCEG. Otherwise output one capital letter.\n\n"
    "After solving, output only this final block, replacing C with your chosen letter(s):\n"
    "FINAL_ANSWERS:\n\\boxed{C}"
)

SYS_MCQ_STRICT_FORMAT = (
    "You are an expert mathematician answering a multiple-choice question.\n"
    "Mandatory procedure:\n"
    "  1. Solve the problem to get a numerical/symbolic answer.\n"
    "  2. Compare your answer against EACH listed option (A, B, C, ...) and identify the matching letter.\n"
    "  3. If the problem is select-all/check-all, output combined capital letters only, e.g. BCEG.\n"
    "  4. Never box the numerical value, expression, or option text. Never leave the box empty.\n"
    "  5. If your computed answer doesn't match any option exactly, pick the closest one.\n\n"
    "After solving, output only:\nFINAL_ANSWERS:\n\\boxed{<letter-or-letters>}"
)

SYS_MCQ_ELIMINATE = (
    "You are an expert mathematician answering a multiple-choice question. "
    "Eliminate options that are clearly wrong by sign, magnitude, constraints, units, or answer type, "
    "then choose among the remaining options. "
    "If the problem says select all, check all, or there may be more than one answer, output the combined "
    "capital letters only, e.g. BCEG. Otherwise output one capital letter. Never box the option value.\n\n"
    "After solving, output only:\nFINAL_ANSWERS:\n\\boxed{<letter-or-letters>}"
)

SYS_MCQ_MATCH_BACK = (
    "You are an expert mathematician answering a multiple-choice question. "
    "Solve the problem first, then EXPLICITLY map your computed answer to the option list. "
    "If the problem is select-all/check-all, output combined capital letters only, e.g. BCEG. "
    "The box must contain only option letter(s), not the value.\n\n"
    "After solving, output only:\nFINAL_ANSWERS:\n\\boxed{<letter-or-letters>}"
)

SYS_MCQ_ALGORITHM_SEQUENCE = (
    "You are answering an algorithm-sequence multiple-choice question. "
    "Use the definition and x_list to infer the y_list pattern, then compare against each option list. "
    "Check differences, offsets, monotonicity, parity, recurrence behavior, and obvious perturbations. "
    "Output only the matching option letter, not the sequence.\n\n"
    "After solving, output only:\nFINAL_ANSWERS:\n\\boxed{<letter>}"
)


# ── Short fallback prompts ──────────────────────────────────────────────────
FALLBACK_PROMPTS = {
    "baseline": (
        "Answer directly with no reasoning. " + SHORT_FINAL_RULES
    ),
    "general_mcq_eliminate": (
        "No reasoning. Choose the option letter. For select-all, output combined letters only. "
        "Output only:\nFINAL_ANSWERS:\n\\boxed{<letter-or-letters>}"
    ),
    "algorithm_sequence_mcq": (
        "No reasoning. Compare the x_list/y_list pattern to the option lists and output only:\n"
        "FINAL_ANSWERS:\n\\boxed{<letter>}"
    ),
    "statistics_prompt": (
        "No reasoning. Give the unrounded statistics answer unless rounding is explicitly requested. "
        + SHORT_FINAL_RULES
    ),
    "calculus_prompt": (
        "No reasoning. Prefer exact WebWork-style calculus notation. " + SHORT_FINAL_RULES
    ),
    "multi_blank_free_response": (
        "No reasoning. Count [ANS] blanks and output one box per blank in order. "
        "For select-all blanks use combined letters only; for True/False use the problem's requested T/F or option-letter style. "
        + SHORT_FINAL_RULES
    ),
}


# ── Few-shot blocks (prepended to user message) ─────────────────────────────
FEW_SHOT_FREE = (
    "Here is a worked example of the expected format:\n\n"
    "Example problem: Compute x+y. Sum: [ANS] Compute x-y. Difference: [ANS], where x=3 and y=2.\n"
    "Example solution: x+y=5 and x-y=1.\n"
    "FINAL_ANSWERS:\n\\boxed{5}\n\\boxed{1}\n\n"
    "Now solve the following problem:\n"
)

FEW_SHOT_MCQ = (
    "Here is a worked example showing the required compute-then-match-then-box procedure:\n\n"
    "Example problem: Solve x^2 = 4 for the positive root.\n"
    "Options:\nA. -2\nB. 0\nC. 2\nD. 4\n"
    "Example reasoning: x^2 = 4 gives x = ±2; the positive root is 2. "
    "Matching against options: 2 corresponds to option C.\n"
    "FINAL_ANSWERS:\n\\boxed{C}\n\n"
    "Notice the box contains the LETTER C, not the value 2. Now solve the following problem:\n"
)


# ── Independent prompt registries ──────────────────────────────────────────
# Each entry: (name, system_prompt, few_shot_block_or_None)

MCQ_PROMPTS = [
    ("general_mcq_eliminate", SYS_MCQ_ELIMINATE,       None),
    ("algorithm_sequence_mcq", SYS_MCQ_ALGORITHM_SEQUENCE, None),
    ("baseline",            SYS_MCQ_BASELINE,        None),
    ("strict_format",       SYS_MCQ_STRICT_FORMAT,   None),
    ("eliminate",           SYS_MCQ_ELIMINATE,       None),
    ("match_back",          SYS_MCQ_MATCH_BACK,      None),
    ("few_shot",            SYS_MCQ_BASELINE,        FEW_SHOT_MCQ),
    ("strict_few_shot",     SYS_MCQ_STRICT_FORMAT,   FEW_SHOT_MCQ),
    ("match_back_few_shot", SYS_MCQ_MATCH_BACK,      FEW_SHOT_MCQ),
]

FREE_PROMPTS = [
    ("statistics_prompt",          SYS_FREE_STATISTICS,    None),
    ("calculus_prompt",            SYS_FREE_CALCULUS,      None),
    ("multi_blank_free_response",  SYS_FREE_MULTI_ANSWER,  None),
    ("baseline",            SYS_FREE_BASELINE,       None),
    ("strict_format",       SYS_FREE_STRICT_FORMAT,  None),
    ("verify",              SYS_FREE_VERIFY,         None),
    ("concise",             SYS_FREE_CONCISE,        None),
    ("multi_answer",        SYS_FREE_MULTI_ANSWER,   None),
    ("few_shot",            SYS_FREE_BASELINE,       FEW_SHOT_FREE),
    ("strict_few_shot",     SYS_FREE_STRICT_FORMAT,  FEW_SHOT_FREE),
]


def _lookup(prompts: list, name: str):
    for n, sys_p, fs in prompts:
        if n == name:
            return sys_p, fs
    raise KeyError(f"Unknown prompt name: {name}")


def build_mcq_prompt(name: str, question: str, options: list) -> tuple[str, str]:
    sys_p, fs = _lookup(MCQ_PROMPTS, name)
    labels = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
    user = f"{question}\n\nOptions:\n{opts_text}"
    if fs:
        user = fs + user
    return sys_p, user


def build_free_prompt(name: str, question: str) -> tuple[str, str]:
    sys_p, fs = _lookup(FREE_PROMPTS, name)
    user = question
    if fs:
        user = fs + user
    return sys_p, user


def build_routed_prompt(
    name: str,
    question: str,
    options: list | None = None,
    fallback: bool = False,
) -> tuple[str, str]:
    """Build a prompt family selected by format_router."""
    if fallback:
        sys_p = FALLBACK_PROMPTS.get(name, FALLBACK_PROMPTS["baseline"])
        fs = None
    elif options:
        sys_p, fs = _lookup(MCQ_PROMPTS, name)
    else:
        sys_p, fs = _lookup(FREE_PROMPTS, name)

    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user = f"{question}\n\nOptions:\n{opts_text}"
    else:
        user = question
    if fs:
        user = fs + user
    if fallback:
        user = user + "\n\n/no_think"
    return sys_p, user
