# t2t-qwen-math
CSE 151B Final Project — Qwen3-4B-Thinking on the public/private math dataset.

## Final code submission

Single entry point:

```python
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",
    output_dir="results/final_run_inference",
    submission_name="submission.csv",
    free_model_id="sengBJY/CSE151B_FinalProject",
)
```

Equivalent CLI:

```bash
python run_inference.py \
  --data-path data/private.jsonl \
  --output-dir results/final_run_inference \
  --submission-name submission.csv
```

The function runs the full final pipeline: it loads the base
`Qwen/Qwen3-4B-Thinking-2507` model for MCQ-like rows, loads the fine-tuned
GRPO checkpoint from `sengBJY/CSE151B_FinalProject` by default for
free-response rows, applies the compact prompt post-processing/fallback logic,
merges the routed outputs, and writes a Kaggle-compatible CSV. MCQ-like rows
run with greedy `n=1`; free-response rows run with self-consistency `n=3`.

Model weights:

- Base MCQ model: downloaded automatically by HuggingFace/vLLM from
  `Qwen/Qwen3-4B-Thinking-2507`.
- Fine-tuned GRPO free-response model: upload the checkpoint-81 model to
  HuggingFace Hub at `sengBJY/CSE151B_FinalProject`. That repo ID is the
  default in `run_inference.py`, so no manual weight placement is needed. To
  override it, set `T2T_QWEN_GRPO_MODEL_ID`, call
  `run_inference(free_model_id="...")`, or pass
  `python run_inference.py --free-model-id ...`.

Hardware/runtime used for final generation:

- GPU type for private-set inference: NVIDIA RTX 5090.
- Approximate total private-set generation/inference time: about 2.5 hours.
- GRPO training/checkpoint generation also used a dual NVIDIA RTX PRO 6000
  96GB setup; see `docs/full_grpo_runbook.md`.

Final submitted CSV in this working tree:

- `submissions/20260531_final/final_private_submission.csv`
- Merge audit: `submissions/20260531_final/final_private_merge_validation.json`

Representative LoRA artifacts are consolidated under `experiments/lora/`.
That folder keeps only the selected reports, summaries, and validation outputs
needed to explain the LoRA experiments; generated LoRA datasets and duplicate
per-step outputs were removed from the public branch.

Prompt/autoresearch summaries are consolidated under `experiments/prompting/`.
Power-monitor logs are consolidated under `experiments/power/`. Bulky generated
outputs under `results/`, `analysis/visualizations/`, and generated training
corpora are intentionally ignored and can be regenerated from the scripts.

Open **`notebooks/starter_code_cse151b_comp.ipynb`** to set up the environment
(installs vLLM, pulls the model, runs a few baseline samples, and scores them
against `data/public.jsonl`). Once the venv is created the rest of the work
happens through the scripts under `scripts/`.

## Repository layout

```
t2t-qwen-math/
├── data/                            # public.jsonl, private.jsonl, sample_submission.csv
│   └── sft_free_v1/                 # small retained SFT split
├── prompts/                         # active prompt package (importable)
│   ├── compact_prompt_pack.py       # current submission/SFT prompt
│   ├── grpo_prompt_pack.py          # full-GRPO prompt + reward helpers
│   └── worker_prompt.md             # single-example data worker prompt
├── scripts/                         # all executable runners
│   ├── train_grpo_full.py           # full-parameter free-response GRPO
│   ├── eval_grpo_checkpoint.py      # checkpoint generation + local scoring
│   ├── create_submission.py         # compact prompt submission pipeline
│   ├── prompt_sweep.py              # prompt-vs-config sweep
│   ├── run_sweep_best_combo.py      # best-of-sweep submission run
│   └── verify_public.py             # score a results dir against data/public.jsonl
├── analysis/                        # was result_analyze/
│   ├── visualize_wrong.py           # per-run wrong-answer HTML (uses gold)
│   ├── visualize_submission.py      # no-gold heuristic flag report (per submission)
│   ├── hf_dataset_similarity/       # retained dataset-search notes
│   └── public_verification_summary.csv
├── experiments/                     # curated experiment artifacts
│   ├── lora/                        # representative LoRA runs only
│   ├── prompting/                   # compact prompt/autoresearch summaries
│   └── power/                       # archived power/cost monitor logs
├── submissions/
│   └── 20260531_final/              # final CSV and merge audit artifacts
├── results/                         # ignored generated CSVs / JSONL audit logs
├── docs/                            # final_report_draft.tex, references.bib
├── notebooks/                       # starter_code_cse151b_comp.ipynb
├── judger.py                        # answer scoring (kept at root — library)
├── utils.py                         # LaTeX / answer normalization helpers
├── requirements.txt
└── README.md
```

Active prompt files are `prompts/compact_prompt_pack.py`,
`prompts/grpo_prompt_pack.py`, and `prompts/worker_prompt.md`. The old
legacy/comparison/problem-type prompt modules were removed.

`judger.py` and `utils.py` stay at the project root because every script
imports them by bare name; moving them would force every consumer to update.

Generated folders intentionally not tracked: `results/`, `outputs/`,
`single_tasks/`, `logs/`, `analysis/visualizations/`, and large generated
training corpora such as `data/hf_mixed_math_*` and
`data/autoresearch_free_*`.

## Common tasks

All commands assume you are in the project root and have activated the venv
(`.venv/Scripts/python` on Windows, `.venv/bin/python` on Linux/Mac).

### 1. Run full-parameter GRPO on cloud

```bash
python scripts/train_grpo_full.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --train-file data/public_free_response.jsonl \
  --eval-file data/public_dev.jsonl \
  --output-dir checkpoints/full_grpo_free/qwen3_4b
```

For dual RTX PRO 6000 96GB, use the wrapper:

```bash
bash scripts/run_grpo_dual_pro6000.sh
```

The default dual-card settings use `per_device_train_batch_size=4`,
`num_generations=8`, `max_completion_length=4096`, `save_steps=50`, and
`eval_steps=50`. See `docs/full_grpo_runbook.md`.

### 2. Run the compact submission pipeline

```bash
# Default dynamic prompt module is prompts.compact_prompt_pack
python scripts/create_submission.py
# Pure compact MCQ/FREE split:
python scripts/create_submission.py --routing-mode compact \
  --prompt-module prompts.compact_prompt_pack \
  --mcq-prompt compact --free-prompt compact --mcq-config greedy_n1 --free-config sc_n3
# Optional high-budget stage for rows still failing after continuation/bounded fallback:
python scripts/create_submission.py --routing-mode compact \
  --prompt-module prompts.compact_prompt_pack \
  --mcq-prompt compact --free-prompt compact \
  --max-model-len 65536 --high-budget-fallback
```

### 3. Sweep prompts on a stratified public subset

```bash
python scripts/prompt_sweep.py --qtype all --config sc_n3
# Or explicitly select the compact prompt:
python scripts/prompt_sweep.py --prompt-module prompts.compact_prompt_pack \
  --qtype free --free-prompt compact
```

### 4. Score a public-set run

```bash
python scripts/verify_public.py results/32gb_balanced_public
```

This writes `analysis/<run_name>.jsonl`, appends to
`analysis/public_verification_summary.csv`, and refreshes
`analysis/visualizations/<run_name>.html`. The JSONL and HTML outputs are
ignored generated artifacts; only the summary CSV is retained.

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
python scripts/power_cost_monitor.py --label grpo_smoke -- \
  python scripts/train_grpo_full.py \
    --train-limit 16 \
    --eval-limit 16 \
    --max-steps 5 \
    --num-generations 4 \
    --max-completion-length 1024 \
    --output-dir checkpoints/full_grpo_free/smoke
```

The monitor samples NVIDIA GPU power through `nvidia-smi` and CPU package
energy through Linux RAPL when available. It logs each run to
`results/power_usage_runs.jsonl` and `results/power_usage_runs.csv`; archived
project logs are under `experiments/power/`.

By default, cost uses SDG&E residential TOU-DR1 bundled rates for San Diego,
excluding fixed monthly/base service charges. Override this with your actual
bill rate if needed:

```bash
python scripts/power_cost_monitor.py --rate-usd-per-kwh 0.52 -- \
  python scripts/train_grpo_full.py --max-steps 50
```

If you have a wall meter or want to account for unmeasured system overhead,
use `--fixed-watts` or `--extra-watts`:

```bash
python scripts/power_cost_monitor.py --power-source fixed --fixed-watts 620 -- \
  python scripts/train_grpo_full.py --max-steps 50
```

### 7. Evaluate a GRPO checkpoint

Generate and score a checkpoint on the free-response dev file:

```bash
python scripts/eval_grpo_checkpoint.py \
  --model checkpoints/full_grpo_free/qwen3_4b/checkpoint-50 \
  --tokenizer Qwen/Qwen3-4B-Thinking-2507 \
  --data-file data/public_dev.jsonl \
  --output-dir results/full_grpo_free/checkpoint_50_dev
```

This writes `submission.jsonl`, `submission.csv`, and `score_summary.csv`.

## Prompt module reference

| Module                                  | Purpose |
|---|---|
| `prompts.compact_prompt_pack`           | Active compact judge-compatible prompt pack and final-answer normalization. |
| `prompts.grpo_prompt_pack`              | Full-GRPO rollout prompt plus correctness, format, length, and combined reward functions. |

The runner scripts that take a `--prompt-module` flag default to
`prompts.compact_prompt_pack`; all dynamic loads use `importlib.import_module`,
so any module exposing the same builder API can be plugged in.
