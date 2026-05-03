"""Prompt package for the t2t-qwen-math project.

The unified prompt package lives in ``math_reasoning_prompts``. It provides
two output modes (``internal_answer_json_mode`` and
``submission_response_mode``), a classifier prompt, format/domain prompt
fragments, repair prompts, validator notes, and builder helpers.

The legacy single-mode prompt constants in :mod:`prompts.legacy_prompts` are
still used by ``scripts/prompt_sweep.py`` and ``scripts/create_submission.py``;
this package is
meant to be imported alongside (or to gradually replace) them.
"""

from .math_reasoning_prompts import (
    GLOBAL_SYSTEM_PROMPT,
    INTERNAL_MODE_INSTRUCTIONS,
    SUBMISSION_MODE_INSTRUCTIONS,
    CLASSIFIER_SYSTEM_PROMPT,
    FORMAT_PROMPTS,
    DOMAIN_PROMPTS,
    INTERNAL_JSON_SCHEMA,
    SUBMISSION_RESPONSE_CONVENTION,
    REPAIR_INTERNAL_SYSTEM_PROMPT,
    REPAIR_SUBMISSION_SYSTEM_PROMPT,
    VALIDATOR_DESIGN_NOTES,
    FEW_SHOT_EXAMPLES,
    USAGE_NOTES,
    Mode,
    build_classifier_prompt,
    build_internal_prompt,
    build_submission_prompt,
    build_repair_internal_prompt,
    build_repair_submission_prompt,
)
from .compact_prompt_pack import (
    BASE_SYSTEM,
    MCQ_SYSTEM,
    FREE_SYSTEM,
    SUFFIXES,
    route_prompt,
)

__all__ = [
    "GLOBAL_SYSTEM_PROMPT",
    "INTERNAL_MODE_INSTRUCTIONS",
    "SUBMISSION_MODE_INSTRUCTIONS",
    "CLASSIFIER_SYSTEM_PROMPT",
    "FORMAT_PROMPTS",
    "DOMAIN_PROMPTS",
    "INTERNAL_JSON_SCHEMA",
    "SUBMISSION_RESPONSE_CONVENTION",
    "REPAIR_INTERNAL_SYSTEM_PROMPT",
    "REPAIR_SUBMISSION_SYSTEM_PROMPT",
    "VALIDATOR_DESIGN_NOTES",
    "FEW_SHOT_EXAMPLES",
    "USAGE_NOTES",
    "Mode",
    "build_classifier_prompt",
    "build_internal_prompt",
    "build_submission_prompt",
    "build_repair_internal_prompt",
    "build_repair_submission_prompt",
    "BASE_SYSTEM",
    "MCQ_SYSTEM",
    "FREE_SYSTEM",
    "SUFFIXES",
    "route_prompt",
]
