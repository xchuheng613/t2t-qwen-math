"""Three-way comparison prompt module: baseline vs blank vs best-of-sweep.

Designed to be used by ``scripts/prompt_sweep.py``::

    python scripts/prompt_sweep.py --prompt-module prompts.comparison_prompts

Variants (same name available for both MCQ and free-response):

  - ``baseline`` -- the original starter-notebook prompt
    (``notebooks/starter_code_cse151b_comp.ipynb``: ``SYSTEM_PROMPT_MATH`` /
    ``SYSTEM_PROMPT_MCQ``).
  - ``blank`` -- empty system prompt; user message is the raw question (plus
    rendered options for MCQ). Used as the absolute floor.
  - ``best`` -- top performer from ``results/sweep/summary.csv``:
    ``match_back_few_shot`` for MCQ (0.77) and ``strict_few_shot`` for
    free-response (0.57).

The module reuses prompt strings from :mod:`prompts.legacy_prompts` so the
``best`` variants stay byte-identical to the sweep entries that produced those
numbers.
"""

from __future__ import annotations

from . import legacy_prompts as legacy


# ── Baseline (starter notebook) ────────────────────────────────────────────
SYS_FREE_BASELINE_STARTER = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYS_MCQ_BASELINE_STARTER = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


# ── Blank (no system prompt) ───────────────────────────────────────────────
SYS_BLANK = ""


# ── Registries (name, system_prompt, few_shot_block_or_None) ───────────────
MCQ_PROMPTS = [
    ("baseline", SYS_MCQ_BASELINE_STARTER, None),
    ("blank",    SYS_BLANK,                None),
    ("best",     legacy.SYS_MCQ_MATCH_BACK, legacy.FEW_SHOT_MCQ),
]

FREE_PROMPTS = [
    ("baseline", SYS_FREE_BASELINE_STARTER,    None),
    ("blank",    SYS_BLANK,                    None),
    ("best",     legacy.SYS_FREE_STRICT_FORMAT, legacy.FEW_SHOT_FREE),
]


def _lookup(prompts: list, name: str):
    for prompt_name, system_prompt, few_shot in prompts:
        if prompt_name == name:
            return system_prompt, few_shot
    raise KeyError(f"Unknown prompt name: {name}")


def build_mcq_prompt(name: str, question: str, options: list) -> tuple[str, str]:
    system_prompt, few_shot = _lookup(MCQ_PROMPTS, name)
    labels = [chr(65 + idx) for idx in range(len(options))]
    opts_text = "\n".join(f"{label}. {str(option).strip()}" for label, option in zip(labels, options))
    user = f"{question}\n\nOptions:\n{opts_text}"
    if few_shot:
        user = few_shot + user
    return system_prompt, user


def build_free_prompt(name: str, question: str) -> tuple[str, str]:
    system_prompt, few_shot = _lookup(FREE_PROMPTS, name)
    user = question
    if few_shot:
        user = few_shot + user
    return system_prompt, user
