# Full-Parameter GRPO Runbook

This is the current mainline for cloud training.

## Defaults

```text
model: Qwen/Qwen3-4B-Thinking-2507
train_file: data/public_free_response.jsonl
eval_file: data/public_dev.jsonl
training: full parameter, no LoRA
target GPU: dual RTX PRO 6000 96GB, single node
```

Dual RTX PRO 6000 default parameters:

```text
launch: accelerate, 2 processes
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_generations: 8
max_prompt_length: 4096
max_completion_length: 4096
learning_rate: 5e-7
warmup_ratio: 0.03
beta: 0.04
bf16: true
gradient_checkpointing: true
attn_implementation: flash_attention_2
save_steps: 50
eval_steps: 50
logging_steps: 5
```

The GRPO effective batch is:

```text
2 GPUs * per_device_train_batch_size 4 * gradient_accumulation_steps 8 = 64 completions/step
```

With `num_generations=8`, that is 8 prompt groups per optimizer step. The
micro global batch is `2 * 4 = 8`, so it is also divisible by
`num_generations=8`, which keeps this compatible with older TRL checks.

## Cloud Setup

```bash
pip install -r requirements.txt
pip install -U trl transformers accelerate datasets
pip install flash-attn --no-build-isolation
```

## Short Smoke Test

Use this before a long run:

```bash
accelerate launch --num_processes 2 --num_machines 1 --mixed_precision bf16 \
  scripts/train_grpo_full.py \
  --train-limit 16 \
  --eval-limit 16 \
  --max-steps 5 \
  --per-device-train-batch-size 4 \
  --num-generations 4 \
  --max-completion-length 1024 \
  --logging-steps 1 \
  --save-steps 5 \
  --eval-steps 5 \
  --output-dir checkpoints/full_grpo_free/smoke
```

## Main Dual RTX PRO 6000 Run

Use the wrapper:

```bash
bash scripts/run_grpo_dual_pro6000.sh
```

Equivalent explicit command:

```bash
accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  scripts/train_grpo_full.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --train-file data/public_free_response.jsonl \
  --eval-file data/public_dev.jsonl \
  --output-dir checkpoints/full_grpo_free/qwen3_4b_dual_pro6000 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --num-generations 8 \
  --max-prompt-length 4096 \
  --max-completion-length 4096 \
  --learning-rate 5e-7 \
  --save-steps 50 \
  --eval-steps 50 \
  --logging-steps 5
```

If memory is still available, increase:

```text
num_generations: 8 -> 12
```

When using `num_generations=12`, also make the effective batch divisible by 12,
for example:

```bash
bash scripts/run_grpo_dual_pro6000.sh \
  --per-device-train-batch-size 6 \
  --num-generations 12
```

If it OOMs, reduce in this order:

```text
max_completion_length: 4096 -> 2048
num_generations: 8 -> 6
```

## Checkpoint Evaluation

```bash
python scripts/eval_grpo_checkpoint.py \
  --model checkpoints/full_grpo_free/qwen3_4b/checkpoint-50 \
  --tokenizer Qwen/Qwen3-4B-Thinking-2507 \
  --data-file data/public_dev.jsonl \
  --output-dir results/full_grpo_free/checkpoint_50_dev
```

The eval script writes:

```text
submission.jsonl
submission.csv
score_summary.csv
```

## Reward

Training uses `prompts.grpo_prompt_pack.full_grpo_reward`:

```text
correctness: 2.0
format:      0.4
length:      0.2
```

The correctness reward uses the project `Judger` plus the compact final-answer
normalization path. Format reward checks `FINAL_ANSWERS` and exactly one final
box. Length reward discourages long, truncation-prone outputs.
