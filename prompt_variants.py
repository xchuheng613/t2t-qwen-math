"""Prompt variants for the sweep. Edit freely; each entry must define
system prompts for both MCQ and free-form, plus an optional few-shot block
inserted into the user message."""

from typing import Optional


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
    "  1. Put exactly ONE \\boxed{...} at the very end of your response.\n"
    "  2. Inside the box: NO units, NO words, NO equals signs, NO 'x =' prefix.\n"
    "  3. For multiple sub-answers, comma-separate inside the single box: \\boxed{3, 7}.\n"
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


# ── MCQ system prompts ──────────────────────────────────────────────────────
SYS_MCQ_BASELINE = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

SYS_MCQ_STRICT_FORMAT = (
    "You are an expert mathematician answering a multiple-choice question.\n"
    "Output rules (very important):\n"
    "  1. Your final line must be \\boxed{X} where X is a single capital letter (A-Z).\n"
    "  2. Do NOT box anything else. Do NOT include the option text.\n"
    "  3. If you are unsure, pick your best guess — never leave the box empty."
)

SYS_MCQ_ELIMINATE = (
    "You are an expert mathematician answering a multiple-choice question. "
    "First, briefly eliminate options that are clearly wrong (wrong sign, wrong magnitude, "
    "wrong units, contradicts a known constraint). Then choose among the remaining options. "
    "Output ONLY the chosen letter inside \\boxed{}, e.g. \\boxed{C}."
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
    "Here is a worked example of the expected format:\n\n"
    "Example problem: What is 2 + 2?\n"
    "Options:\nA. 3\nB. 4\nC. 5\nD. 6\n"
    "Example reasoning: 2+2 = 4, so option B.\n"
    "Final answer: \\boxed{B}\n\n"
    "Now solve the following problem:\n"
)


# ── Variant registry ────────────────────────────────────────────────────────
# Each variant: (name, sys_free, sys_mcq, few_shot_free, few_shot_mcq)
VARIANTS = [
    ("baseline",
     SYS_FREE_BASELINE, SYS_MCQ_BASELINE, None, None),

    ("strict_format",
     SYS_FREE_STRICT_FORMAT, SYS_MCQ_STRICT_FORMAT, None, None),

    ("verify",
     SYS_FREE_VERIFY, SYS_MCQ_BASELINE, None, None),

    ("concise",
     SYS_FREE_CONCISE, SYS_MCQ_BASELINE, None, None),

    ("mcq_eliminate",
     SYS_FREE_BASELINE, SYS_MCQ_ELIMINATE, None, None),

    ("few_shot",
     SYS_FREE_BASELINE, SYS_MCQ_BASELINE, FEW_SHOT_FREE, FEW_SHOT_MCQ),

    ("strict_plus_few_shot",
     SYS_FREE_STRICT_FORMAT, SYS_MCQ_STRICT_FORMAT, FEW_SHOT_FREE, FEW_SHOT_MCQ),
]


def build_prompt(variant_name: str, question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a given variant and question."""
    for name, sys_free, sys_mcq, fs_free, fs_mcq in VARIANTS:
        if name != variant_name:
            continue
        if options:
            labels = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
            user = f"{question}\n\nOptions:\n{opts_text}"
            if fs_mcq:
                user = fs_mcq + user
            return sys_mcq, user
        else:
            user = question
            if fs_free:
                user = fs_free + user
            return sys_free, user
    raise KeyError(f"Unknown variant: {variant_name}")
