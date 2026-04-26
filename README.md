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
