#!/usr/bin/env python3
"""Balanced 32 GB preset (e.g. RTX 5090 / A6000 / V100-32G).

Goal: a sensible tradeoff between speed and accuracy. Self-consistency with
n=3 (matching ``create_submission.CONFIGS['sc_n3']``), full reasoning budget,
and standard prompt-package settings (no few-shot, repair enabled).

Defaults vs. ``run_math_prompts.py``:
  - max_tokens                : 16384  (full Qwen3-Thinking budget)
  - max_model_len             : 32768
  - gpu_memory_utilization    : 0.85
  - max_num_seqs              : 32
  - max_num_batched_tokens    : 16384
  - enforce_eager             : False  (use CUDA graphs for throughput)
  - sampling                  : sc_n3 (3 samples, majority pick of valid)
  - few-shot                  : disabled
  - repair                    : enabled (default)
  - free final-answer cleanup : enabled
  - free sample ranking       : enabled

CLI args from ``run_math_prompts`` are still accepted and override these
defaults. Example:

    python run_32gb_balanced.py --mode submission_response_mode \
        --data-path data/private.jsonl \
        --output-dir results/32gb_balanced_submission
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_math_prompts import main  # noqa: E402


PRESET_DEFAULTS: dict[str, object] = {
    "max_tokens": 16384,
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.85,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 16384,
    "enforce_eager": False,
    "config": "sc_n3",
    "include_few_shot": False,
    "no_repair": False,
    "normalize_free_final_answers": True,
    "rank_free_samples": True,
}


def _apply_preset() -> None:
    user_argv = sys.argv[1:]
    flags = {token.split("=", 1)[0] for token in user_argv if token.startswith("--")}

    extra: list[str] = []
    if "--max-tokens" not in flags:
        extra += ["--max-tokens", str(PRESET_DEFAULTS["max_tokens"])]
    if "--max-model-len" not in flags:
        extra += ["--max-model-len", str(PRESET_DEFAULTS["max_model_len"])]
    if "--gpu-memory-utilization" not in flags:
        extra += ["--gpu-memory-utilization", str(PRESET_DEFAULTS["gpu_memory_utilization"])]
    if "--max-num-seqs" not in flags:
        extra += ["--max-num-seqs", str(PRESET_DEFAULTS["max_num_seqs"])]
    if "--max-num-batched-tokens" not in flags:
        extra += ["--max-num-batched-tokens", str(PRESET_DEFAULTS["max_num_batched_tokens"])]
    if "--config" not in flags:
        extra += ["--config", str(PRESET_DEFAULTS["config"])]
    if "--enforce-eager" not in flags and PRESET_DEFAULTS["enforce_eager"]:
        extra += ["--enforce-eager"]
    if "--include-few-shot" not in flags and PRESET_DEFAULTS["include_few_shot"]:
        extra += ["--include-few-shot"]
    if "--no-repair" not in flags and PRESET_DEFAULTS["no_repair"]:
        extra += ["--no-repair"]
    if "--normalize-free-final-answers" not in flags and PRESET_DEFAULTS["normalize_free_final_answers"]:
        extra += ["--normalize-free-final-answers"]
    if "--rank-free-samples" not in flags and PRESET_DEFAULTS["rank_free_samples"]:
        extra += ["--rank-free-samples"]

    sys.argv[1:] = extra + user_argv


if __name__ == "__main__":
    _apply_preset()
    print(
        "[preset] 32GB BALANCED  config=sc_n3  max_tokens=16384  "
        "max_model_len=32768  free_postprocess=default-on",
        flush=True,
    )
    main()
