#!/usr/bin/env bash
set -euo pipefail

# Single-node, dual RTX PRO 6000 96GB full-parameter GRPO.
# Extra CLI args are appended at the end and override these defaults.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  scripts/train_grpo_full.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --train-file data/public_free_response.jsonl \
  --eval-file data/public_dev.jsonl \
  --output-dir checkpoints/full_grpo_free/qwen3_4b_dual_pro6000 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --num-generations 8 \
  --max-prompt-length 4096 \
  --max-completion-length 4096 \
  --learning-rate 5e-7 \
  --warmup-ratio 0.03 \
  --beta 0.04 \
  --bf16 \
  --gradient-checkpointing \
  --attn-implementation flash_attention_2 \
  --save-steps 50 \
  --eval-steps 50 \
  --logging-steps 5 \
  --save-total-limit 4 \
  --report-to none \
  "$@"
