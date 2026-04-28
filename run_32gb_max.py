#!/usr/bin/env python3
"""Max-performance 32 GB preset (e.g. RTX 5090 / A6000 / V100-32G).

Goal: best leaderboard accuracy, willing to spend extra wall time and
tokens. Self-consistency with n=5, full 32k context, full 32k reasoning
budget, few-shot examples enabled, repair enabled.

Defaults vs. ``run_math_prompts.py``:
  - max_tokens                : 32768  (let Qwen3-Thinking think to its limit)
  - max_model_len             : 65536  (large prefix-cache window)
  - gpu_memory_utilization    : 0.90
  - max_num_seqs              : 64
  - max_num_batched_tokens    : 32768
  - enforce_eager             : False  (CUDA graphs)
  - sampling                  : sc_n5 (n=5 majority over valid samples)
  - few-shot                  : enabled (canonical examples in user msg)
  - repair                    : enabled

This preset registers a NEW sampling config ``sc_n5`` (n=5, t=0.7, top_p=0.95,
top_k=20) into ``run_math_prompts.CONFIGS`` before delegating to the runner.

CLI args from ``run_math_prompts`` are still accepted and override these
defaults. Example:

    python run_32gb_max.py --mode submission_response_mode \
        --data-path data/private.jsonl \
        --output-dir results/32gb_max_submission
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_math_prompts  # noqa: E402
from run_math_prompts import main, SamplingConfig  # noqa: E402


# Register a beefier self-consistency config so --config sc_n5 is accepted.
run_math_prompts.CONFIGS.setdefault(
    "sc_n5",
    SamplingConfig("sc_n5", temperature=0.7, top_p=0.95, top_k=20, n=5),
)


PRESET_DEFAULTS: dict[str, object] = {
    "max_tokens": 32768,
    "max_model_len": 65536,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 64,
    "max_num_batched_tokens": 32768,
    "enforce_eager": False,
    "config": "sc_n5",
    "include_few_shot": True,
    "no_repair": False,
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

    sys.argv[1:] = extra + user_argv


if __name__ == "__main__":
    _apply_preset()
    print(
        "[preset] 32GB MAX  config=sc_n5  max_tokens=32768  max_model_len=65536  few-shot=on",
        flush=True,
    )
    main()
