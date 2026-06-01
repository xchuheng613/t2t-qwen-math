"""Active prompt package for the t2t-qwen-math project.

The current submission/SFT path uses :mod:`prompts.compact_prompt_pack`.
The GRPO path uses :mod:`prompts.grpo_prompt_pack`, which keeps the compact
final-answer contract and adds reward functions for training.

The older exploratory prompt families were removed to keep dynamic prompt
loading focused on the active path.
"""

from .compact_prompt_pack import (
    BASE_SYSTEM,
    MCQ_SYSTEM,
    FREE_SYSTEM,
    SUFFIXES,
    route_prompt,
)
from .grpo_prompt_pack import (
    GRPO_BASE_SYSTEM,
    GRPO_MCQ_SYSTEM,
    GRPO_FREE_SYSTEM,
    REWARD_WEIGHTS,
    build_grpo_prompt,
    build_grpo_messages,
    make_grpo_row,
    make_grpo_dataset_rows,
    correctness_reward,
    format_reward,
    length_reward,
    full_grpo_reward,
    reward_breakdown,
)

__all__ = [
    "BASE_SYSTEM",
    "MCQ_SYSTEM",
    "FREE_SYSTEM",
    "SUFFIXES",
    "route_prompt",
    "GRPO_BASE_SYSTEM",
    "GRPO_MCQ_SYSTEM",
    "GRPO_FREE_SYSTEM",
    "REWARD_WEIGHTS",
    "build_grpo_prompt",
    "build_grpo_messages",
    "make_grpo_row",
    "make_grpo_dataset_rows",
    "correctness_reward",
    "format_reward",
    "length_reward",
    "full_grpo_reward",
    "reward_breakdown",
]
