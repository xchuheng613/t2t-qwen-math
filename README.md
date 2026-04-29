# t2t-qwen-math
CSE 151B Final Project

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |

## Prompt experiments

Run these commands from the `Python (cse151b)` environment created by the notebook setup.

Run the real 100-problem starter baseline first:

```bash
python run_prompt_experiments.py --experiments starter --num-examples 100
```

Then compare prompt variants on the same 100 examples:

```bash
python run_prompt_experiments.py --experiments strict_boxed verify concise detailed validate_random --num-examples 100
```

Self-consistency with the best prompt so far:

```bash
python run_prompt_experiments.py --experiments verify --num-examples 100 --samples 4 --temperature 0.6
```

Self-consistency only for free-response problems:

```bash
python run_prompt_experiments.py --experiments verify --num-examples 100 --mcq-samples 1 --free-samples 4 --temperature 0.6
```

Use `--num-examples -1` for the full public set. Results and a 20-wrong-example error-analysis sheet are written to `results/prompt_experiments/`.

The legacy sweep scripts default to `prompt_variants.py`. To try the updated routed free-response prompts without changing the old prompts:

```bash
python prompt_sweep.py --prompt-module prompt_variant_updated --qtype free --free-prompt routed_v2 --config sc_n3
```

To run the current problem-type-separated prompt package on labeled public data:

```bash
python run_32gb_balanced.py --mode submission_response_mode \
  --data-path data/public.jsonl \
  --output-dir results/32gb_balanced_public
```

Then verify it with:

```bash
.venv/bin/python verify_public.py results/32gb_balanced_public
```

## Verify public runs

Score a public-set output directory or submission file and refresh the wrong-answer HTML report:

```bash
.venv/bin/python verify_public.py results/32gb_balanced_public_smoke
```

This writes `result_analyze/<run_name>.jsonl`, appends to `result_analyze/public_verification_summary.csv`, and refreshes `result_analyze/visualizations/<run_name>.html`.

## Private submission

The default submission path uses the latest problem-type-separated prompt package from `run_math_prompts.py`:

```bash
python create_submission.py
```

Use `--routing-mode legacy` to run the older MCQ/free split with explicit `--mcq-prompt`, `--free-prompt`, `--prompt-module`, and config flags.
