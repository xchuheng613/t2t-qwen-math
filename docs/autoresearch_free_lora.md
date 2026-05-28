# AutoResearch Free-Response LoRA Notes

The proposed AutoResearch plan is suitable for this project with two repo-specific adjustments:

1. Use the small random search split in `data/autoresearch_free_v1/train.jsonl` for hyperparameter search.
2. Retrain the best configuration on the full free-response dataset only after dev/holdout confirmation.

## Data Files

Use these files for the first AutoResearch stage:

```bash
FREE_TRAIN_FILE=data/autoresearch_free_v1/train.jsonl
FREE_DEV_SFT_FILE=data/autoresearch_free_v1/dev_sft.jsonl
FREE_DEV_FILE=data/autoresearch_free_v1/dev_benchmark.jsonl
FREE_HOLDOUT_FILE=data/autoresearch_free_v1/holdout_benchmark.jsonl
FULL_FREE_TRAIN_FILE=data/hf_mixed_math_free_100k/train.jsonl
BASE_MODEL=Qwen/Qwen3-4B-Thinking-2507
```

`FREE_DEV_SFT_FILE` is for token-loss sanity checks only. Model selection should use generated-answer accuracy on `FREE_DEV_FILE`.

## Current Prompt/Post-Processing

For free-response evaluation, use the compact prompt/post-processing path:

```bash
COMMON_POST="--routing-mode legacy \
  --prompt-module prompts.compact_prompt_pack \
  --free-prompt compact \
  --free-config greedy_n1 \
  --stage0-postprocess \
  --normalize-free-final-answers \
  --rank-free-samples"
```

The current `create_submission.py` does not have a literal `$COMMON_POST` flag; this is just a shell variable containing the flags above.

## Training Command Template

For effective batch size 32, use `--batch-size 1 --grad-accum 32`.

```bash
python scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train-file "$FREE_TRAIN_FILE" \
  --eval-file "$FREE_DEV_SFT_FILE" \
  --output-dir "checkpoints/<run_name>" \
  --rank <r> \
  --alpha <alpha> \
  --dropout <dropout> \
  --lr <learning_rate> \
  --epochs <epochs> \
  --max-seq-length <max_seq_length> \
  --batch-size 1 \
  --grad-accum <grad_accum>
```

## Evaluation Template

```bash
python scripts/create_submission.py \
  --model "checkpoints/merged_<run_name>" \
  --data-path "$FREE_DEV_FILE" \
  --output-dir "results/autoresearch/<run_name>_dev" \
  $COMMON_POST

python scripts/score_benchmark.py \
  "results/autoresearch/<run_name>_dev" \
  --data-path "$FREE_DEV_FILE" \
  --name "<run_name>_dev" \
  --summary-csv "results/autoresearch/summary.csv"
```

Only evaluate `FREE_HOLDOUT_FILE` for the top few dev runs.

## Benchmark Sanity

Fresh oracle scoring was run for the prepared benchmark files:

```text
dev_benchmark:     1000/1000 = 100.00%
holdout_benchmark: 1000/1000 = 100.00%
```

This confirms the benchmark files are compatible with the current local scoring pipeline.

## Run Tag

Suggested fresh branch tag for the actual AutoResearch loop:

```bash
autoresearch/free-may27
```

Before creating it, verify it does not already exist:

```bash
git branch --list "autoresearch/free-may27"
git ls-remote --heads origin "autoresearch/free-may27"
```

