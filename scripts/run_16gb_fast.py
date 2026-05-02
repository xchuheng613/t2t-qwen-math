#!/usr/bin/env python3
"""Fast verification preset for 16 GB VRAM cards (e.g. RTX 4060 Ti / 4070 / A4000).

Goal: run as quickly as possible to validate that the new prompt package
works end-to-end. Uses tight token budgets, greedy sampling (no
self-consistency), and a smaller `max_model_len` so the KV cache fits in
~13-14 GB. Reasoning is intentionally limited.

Defaults vs. ``run_math_prompts.py``:
  - max_tokens                : 2048   (vs. 16384) — short answers, less thinking
  - max_model_len             : 8192   (vs. 32768)
  - gpu_memory_utilization    : 0.85   (push the small card)
  - max_num_seqs              : 8
  - max_num_batched_tokens    : 4096
  - enforce_eager             : True   (skip CUDA-graph capture, faster startup)
  - sampling                  : greedy_n1 (no SC, no temperature)
  - few-shot                  : disabled (saves ~1k prompt tokens)
  - repair                    : disabled by default (verification, not quality)

CLI args from ``run_math_prompts`` are still accepted and override these
defaults. Example:

    python run_16gb_fast.py --mode submission_response_mode \
        --data-path data/public.jsonl --limit 50 \
        --output-dir results/16gb_fast_check
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_math_prompts import main, parse_args  # noqa: E402


PRESET_DEFAULTS: dict[str, object] = {
    "max_tokens": 2048,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.85,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 4096,
    "enforce_eager": True,
    "config": "greedy_n1",
    "include_few_shot": False,
    "no_repair": True,
}


def _apply_preset() -> None:
    """Inject preset defaults into argv before parse_args sees it.

    We only insert flags that the user did NOT pass on the command line, so
    any explicit override on the CLI still wins.
    """
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
    print("[preset] 16GB FAST  config=greedy_n1  max_tokens=2048  max_model_len=8192", flush=True)
    main()
