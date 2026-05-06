# t2t-qwen-math
CSE 151B Final Project — Qwen3-4B-Thinking on the public/private math dataset.

Open **`notebooks/starter_code_cse151b_comp.ipynb`** to set up the environment
(installs vLLM, pulls the model, runs a few baseline samples, and scores them
against `data/public.jsonl`). Once the venv is created the rest of the work
happens through the scripts under `scripts/`.

## Repository layout

```
t2t-qwen-math/
├── data/                            # public.jsonl, private.jsonl, sample_submission.csv
├── prompts/                         # prompt package (importable)
│   ├── math_reasoning_prompts.py    # NEW two-mode prompt package
│   ├── legacy_prompts.py            # was prompt_variants.py
│   └── legacy_prompts_v2.py         # was prompt_variant_updated.py
├── scripts/                         # all executable runners
│   ├── run_math_prompts.py          # generic runner for the new prompt package
│   ├── run_16gb_fast.py             # 16 GB / fast verification preset
│   ├── run_32gb_balanced.py         # 32 GB / balanced preset (sc_n3, repair on)
│   ├── run_32gb_max.py              # 32 GB / max-quality preset (sc_n5, few-shot on)
│   ├── create_submission.py         # legacy + new hybrid submission pipeline
│   ├── prompt_sweep.py              # prompt-vs-config sweep
│   ├── run_sweep_best_combo.py      # best-of-sweep submission run
│   └── verify_public.py             # score a results dir against data/public.jsonl
├── analysis/                        # was result_analyze/
│   ├── visualize_wrong.py           # per-run wrong-answer HTML (uses gold)
│   ├── visualize_submission.py      # no-gold heuristic flag report (per submission)
│   ├── visualizations/              # rendered HTML reports
│   └── *.jsonl, *.csv               # per-run scoring artefacts
├── results/                         # generated CSVs / JSONL audit logs
├── docs/                            # final_report_draft.tex, references.bib
├── notebooks/                       # starter_code_cse151b_comp.ipynb
├── judger.py                        # answer scoring (kept at root — library)
├── utils.py                         # LaTeX / answer normalization helpers
├── requirements.txt
└── README.md
```

`judger.py` and `utils.py` stay at the project root because every script
imports them by bare name; moving them would force every consumer to update.

## Common tasks

All commands assume you are in the project root and have activated the venv
(`.venv/Scripts/python` on Windows, `.venv/bin/python` on Linux/Mac).

### 1. Build a private submission with the new prompt package

```bash
# 32 GB balanced preset (sc_n3 self-consistency, full token budget):
python scripts/run_32gb_balanced.py \
  --mode submission_response_mode \
  --data-path data/private.jsonl \
  --output-dir results/private_submission_v3
```

Other presets:

* `scripts/run_16gb_fast.py` – 16 GB VRAM, greedy, smaller `max_tokens` for
  fast verification.
* `scripts/run_32gb_max.py`  – 32 GB VRAM, `sc_n5`, few-shot examples on,
  highest accuracy.
* `scripts/run_math_prompts.py` – the underlying runner if you want to set
  every flag manually.

### 2. Run the legacy / hybrid pipeline

```bash
# Default: hybrid format-router using prompts.legacy_prompts
python scripts/create_submission.py
# Pure legacy MCQ/FREE split:
python scripts/create_submission.py --routing-mode legacy \
  --prompt-module prompts.legacy_prompts \
  --mcq-prompt eliminate --free-prompt baseline --mcq-config greedy_n1 --free-config sc_n3
# Try the updated routed free-response prompts:
python scripts/create_submission.py --routing-mode legacy \
  --prompt-module prompts.legacy_prompts_v2
# Run the compact judge-compatible prompt pack:
python scripts/create_submission.py --routing-mode legacy \
  --prompt-module prompts.compact_prompt_pack \
  --mcq-prompt compact --free-prompt compact
# Optional high-budget stage for rows still failing after continuation/bounded fallback:
python scripts/create_submission.py --routing-mode legacy \
  --prompt-module prompts.compact_prompt_pack \
  --mcq-prompt compact --free-prompt compact \
  --max-model-len 65536 --high-budget-fallback
```

### 3. Sweep prompts on a stratified public subset

```bash
python scripts/prompt_sweep.py --qtype all --config sc_n3
# Or with the v2 prompt module:
python scripts/prompt_sweep.py --prompt-module prompts.legacy_prompts_v2 \
  --qtype free --free-prompt routed_v2
```

### 4. Score a public-set run

```bash
python scripts/verify_public.py results/32gb_balanced_public
```

This writes `analysis/<run_name>.jsonl`, appends to
`analysis/public_verification_summary.csv`, and refreshes
`analysis/visualizations/<run_name>.html`.

### 5. Visualize a submission with no gold standard

```bash
# Heuristic flags (truncation, missing boxed, MCQ letter validity, ...):
python analysis/visualize_submission.py results/private_submission_v3
# Auto-discover every results/<dir>/submission.csv:
python analysis/visualize_submission.py
```

The HTML lands in `analysis/visualizations/`.

### 6. Track power usage and estimated electricity cost

On an Ubuntu machine, wrap any model run with the power/cost monitor:

```bash
python scripts/power_cost_monitor.py --label 16gb_fast_public -- \
  python scripts/run_16gb_fast.py \
    --mode submission_response_mode \
    --data-path data/public.jsonl \
    --limit 50 \
    --output-dir results/16gb_fast_check
```

The monitor samples NVIDIA GPU power through `nvidia-smi` and CPU package
energy through Linux RAPL when available. It logs each run to
`results/power_usage_runs.jsonl` and `results/power_usage_runs.csv`.

By default, cost uses SDG&E residential TOU-DR1 bundled rates for San Diego,
excluding fixed monthly/base service charges. Override this with your actual
bill rate if needed:

```bash
python scripts/power_cost_monitor.py --rate-usd-per-kwh 0.52 -- \
  python scripts/run_32gb_balanced.py --mode submission_response_mode
```

If you have a wall meter or want to account for unmeasured system overhead,
use `--fixed-watts` or `--extra-watts`:

```bash
python scripts/power_cost_monitor.py --power-source fixed --fixed-watts 620 -- \
  python scripts/run_32gb_max.py --mode submission_response_mode
```

## Prompt module reference

| Module                                  | Purpose |
|---|---|
| `prompts.math_reasoning_prompts`        | New unified prompt package: two output modes (internal JSON / submission text), classifier prompt, format and math-domain profiles, repair prompts, validator notes. |
| `prompts.legacy_prompts`                | Original `FINAL_ANSWERS:` prompt families used by `prompt_sweep`, `create_submission --routing-mode legacy`, and `run_sweep_best_combo`. |
| `prompts.legacy_prompts_v2`             | Drop-in replacement for `legacy_prompts` with refined free-response routing; opt in via `--prompt-module prompts.legacy_prompts_v2`. |
| `prompts.compact_prompt_pack`           | Compact judge-compatible prompt pack with broad suffix routing; opt in via `--prompt-module prompts.compact_prompt_pack --mcq-prompt compact --free-prompt compact`. |

The runner scripts that take a `--prompt-module` flag default to
`prompts.legacy_prompts`; all dynamic loads use `importlib.import_module`,
so any module exposing the same builder API can be plugged in.
