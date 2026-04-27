"""Prompt variants for the sweep. MCQ and free-form prompts are independent
lists so they can be swept separately. Edit either list freely."""


# ── Free-form system prompts ────────────────────────────────────────────────
SYS_FREE_BASELINE = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYS_FREE_STRICT_FORMAT = (
    "You are an expert mathematician. Solve the problem rigorously.\n"
    "Output rules for the FINAL ANSWER (very important):\n"
    "  1. Put exactly ONE \\boxed{...} at the VERY END of your response. No boxes anywhere else.\n"
    "  2. Inside the box: ONLY the final answer value/expression. NO units, NO words, NO 'x =' prefix, NO equals signs.\n"
    "     Wrong: \\boxed{x = 2}, \\boxed{2 cm}, \\boxed{answer is 2}\n"
    "     Right: \\boxed{2}\n"
    "  3. For multiple sub-answers, comma-separate inside ONE box: \\boxed{3, 7}. Never use multiple boxes.\n"
    "  4. Use \\frac{a}{b} (not a/b), \\sqrt{n} (not sqrt n), and standard LaTeX.\n"
    "  5. Do not box intermediate results — only the final answer."
)

SYS_FREE_VERIFY = (
    "You are an expert mathematician. Solve the problem step-by-step, then VERIFY your answer "
    "by substitution, re-derivation, or a sanity check (units, magnitude, edge cases). "
    "Only after verification, write the final answer inside a single \\boxed{}. "
    "For multiple sub-answers, comma-separate inside one box: \\boxed{3, 7}."
)

SYS_FREE_CONCISE = (
    "You are an expert mathematician. Solve the problem. Keep your written explanation brief — "
    "the grader only reads the final \\boxed{} answer. "
    "Put exactly one \\boxed{...} at the end with the final answer (no units, no words). "
    "For multiple sub-answers, comma-separate inside one box: \\boxed{3, 7}."
)

SYS_FREE_MULTI_ANSWER = (
    "You are an expert mathematician. Solve the problem carefully.\n"
    "Before the final answer, count how many [ANS] blanks or requested outputs the problem has.\n"
    "Your final line must contain exactly one \\boxed{...}.\n"
    "Inside that box, write one answer for each [ANS] blank/requested output, in the same order, "
    "separated by commas. Do not omit earlier parts. Do not include labels, units, words, or extra boxes.\n"
    "For equations or models, include the full equation if the problem asks for one."
)


# ── MCQ system prompts ──────────────────────────────────────────────────────
SYS_MCQ_BASELINE = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

SYS_MCQ_STRICT_FORMAT = (
    "You are an expert mathematician answering a multiple-choice question.\n"
    "Mandatory procedure:\n"
    "  1. Solve the problem to get a numerical/symbolic answer.\n"
    "  2. Compare your answer against EACH listed option (A, B, C, ...) and identify the matching letter.\n"
    "  3. Your final line MUST be \\boxed{X} where X is a single capital letter (A-Z) — the letter, NOT the value.\n"
    "  4. Never box the numerical value, expression, or option text. Never leave the box empty.\n"
    "  5. If your computed answer doesn't match any option exactly, pick the closest one."
)

SYS_MCQ_ELIMINATE = (
    "You are an expert mathematician answering a multiple-choice question. "
    "First, briefly eliminate options that are clearly wrong (wrong sign, wrong magnitude, "
    "wrong units, contradicts a known constraint). Then choose among the remaining options. "
    "Output ONLY the chosen letter inside \\boxed{}, e.g. \\boxed{C} — never the value."
)

SYS_MCQ_MATCH_BACK = (
    "You are an expert mathematician answering a multiple-choice question. "
    "Solve the problem first, then EXPLICITLY map your computed answer to the option list. "
    "End your response with two lines:\n"
    "  Line 1: 'My computed answer is <value>, which matches option <letter>.'\n"
    "  Line 2: \\boxed{<letter>}\n"
    "The box must contain only a single capital letter (A-Z), not the value."
)


# ── Few-shot blocks (prepended to user message) ─────────────────────────────
FEW_SHOT_FREE = (
    "Here is a worked example of the expected format:\n\n"
    "Example problem: Find all real x such that x^2 - 5x + 6 = 0.\n"
    "Example solution: Factor as (x-2)(x-3)=0, so x = 2 or x = 3.\n"
    "Final answer: \\boxed{2, 3}\n\n"
    "Now solve the following problem:\n"
)

FEW_SHOT_MCQ = (
    "Here is a worked example showing the required compute-then-match-then-box procedure:\n\n"
    "Example problem: Solve x^2 = 4 for the positive root.\n"
    "Options:\nA. -2\nB. 0\nC. 2\nD. 4\n"
    "Example reasoning: x^2 = 4 gives x = ±2; the positive root is 2. "
    "Matching against options: 2 corresponds to option C.\n"
    "Final answer: \\boxed{C}\n\n"
    "Notice the box contains the LETTER C, not the value 2. Now solve the following problem:\n"
)


# ── Independent prompt registries ──────────────────────────────────────────
# Each entry: (name, system_prompt, few_shot_block_or_None)

MCQ_PROMPTS = [
    ("baseline",            SYS_MCQ_BASELINE,        None),
    ("strict_format",       SYS_MCQ_STRICT_FORMAT,   None),
    ("eliminate",           SYS_MCQ_ELIMINATE,       None),
    ("match_back",          SYS_MCQ_MATCH_BACK,      None),
    ("few_shot",            SYS_MCQ_BASELINE,        FEW_SHOT_MCQ),
    ("strict_few_shot",     SYS_MCQ_STRICT_FORMAT,   FEW_SHOT_MCQ),
    ("match_back_few_shot", SYS_MCQ_MATCH_BACK,      FEW_SHOT_MCQ),
]

FREE_PROMPTS = [
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
