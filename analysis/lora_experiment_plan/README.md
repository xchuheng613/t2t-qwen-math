# LoRA Experiment Plan

All experiments are one epoch and use eval/logging every 10 optimizer steps by default.
Compare training loss against validation loss; keep a run only if train loss decreases without validation loss blowing up and task accuracy improves.
The score commands also write `analysis/*.jsonl`; run `python3 analysis/visualize_wrong.py` afterward to inspect train/validation/holdout error patterns in HTML.

## Experiments

### lr2e-5_adamw_all_r16

- LR/optimizer: `2e-05` / `adamw_torch`
- Rank/alpha/dropout: `16` / `32` / `0.05`
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Why: Same adapter capacity as the failed run, but lower LR to test whether 5e-5 was too aggressive.

```bash
python3 scripts/train_lora_sft.py --model Qwen/Qwen3-4B-Thinking-2507 --train-file data/sft_lora_public_v2/train.jsonl --eval-file data/sft_lora_public_v2/dev.jsonl --output-dir checkpoints/lora_sweep_v3/lr2e-5_adamw_all_r16 --epochs 1.0 --lr 2e-05 --optim adamw_torch --rank 16 --alpha 32 --dropout 0.05 --grad-accum 16 --batch-size 1 --max-seq-length 4096 --logging-steps 10 --eval-steps 10 --save-steps 10 --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
python3 scripts/merge_lora.py --adapter checkpoints/lora_sweep_v3/lr2e-5_adamw_all_r16 --out checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_all_r16
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/train_free_keep.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_all_r16/train --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_all_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_all_r16/train --data-path data/lora_public_v2/train_free_keep.jsonl --name lr2e-5_adamw_all_r16_train --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_all_r16_train.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/public_dev.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_all_r16/validation --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_all_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_all_r16/validation --data-path data/lora_public_v2/public_dev.jsonl --name lr2e-5_adamw_all_r16_validation --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_all_r16_validation.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/holdout_free.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_all_r16/holdout --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_all_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_all_r16/holdout --data-path data/lora_public_v2/holdout_free.jsonl --name lr2e-5_adamw_all_r16_holdout --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_all_r16_holdout.jsonl
```

### lr1e-5_adamw_all_r16

- LR/optimizer: `1e-05` / `adamw_torch`
- Rank/alpha/dropout: `16` / `32` / `0.05`
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Why: Slower LR check. If train loss barely moves, the run is underfitting or needs more steps/data.

```bash
python3 scripts/train_lora_sft.py --model Qwen/Qwen3-4B-Thinking-2507 --train-file data/sft_lora_public_v2/train.jsonl --eval-file data/sft_lora_public_v2/dev.jsonl --output-dir checkpoints/lora_sweep_v3/lr1e-5_adamw_all_r16 --epochs 1.0 --lr 1e-05 --optim adamw_torch --rank 16 --alpha 32 --dropout 0.05 --grad-accum 16 --batch-size 1 --max-seq-length 4096 --logging-steps 10 --eval-steps 10 --save-steps 10 --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
python3 scripts/merge_lora.py --adapter checkpoints/lora_sweep_v3/lr1e-5_adamw_all_r16 --out checkpoints/merged_lora_sweep_v3/lr1e-5_adamw_all_r16
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/train_free_keep.jsonl --output-dir results/lora_sweep_v3/lr1e-5_adamw_all_r16/train --model checkpoints/merged_lora_sweep_v3/lr1e-5_adamw_all_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr1e-5_adamw_all_r16/train --data-path data/lora_public_v2/train_free_keep.jsonl --name lr1e-5_adamw_all_r16_train --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr1e-5_adamw_all_r16_train.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/public_dev.jsonl --output-dir results/lora_sweep_v3/lr1e-5_adamw_all_r16/validation --model checkpoints/merged_lora_sweep_v3/lr1e-5_adamw_all_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr1e-5_adamw_all_r16/validation --data-path data/lora_public_v2/public_dev.jsonl --name lr1e-5_adamw_all_r16_validation --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr1e-5_adamw_all_r16_validation.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/holdout_free.jsonl --output-dir results/lora_sweep_v3/lr1e-5_adamw_all_r16/holdout --model checkpoints/merged_lora_sweep_v3/lr1e-5_adamw_all_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr1e-5_adamw_all_r16/holdout --data-path data/lora_public_v2/holdout_free.jsonl --name lr1e-5_adamw_all_r16_holdout --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr1e-5_adamw_all_r16_holdout.jsonl
```

### lr2e-5_adafactor_all_r16

- LR/optimizer: `2e-05` / `adafactor`
- Rank/alpha/dropout: `16` / `32` / `0.05`
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Why: Optimizer swap requested by TA; compare stability and validation loss against AdamW.

```bash
python3 scripts/train_lora_sft.py --model Qwen/Qwen3-4B-Thinking-2507 --train-file data/sft_lora_public_v2/train.jsonl --eval-file data/sft_lora_public_v2/dev.jsonl --output-dir checkpoints/lora_sweep_v3/lr2e-5_adafactor_all_r16 --epochs 1.0 --lr 2e-05 --optim adafactor --rank 16 --alpha 32 --dropout 0.05 --grad-accum 16 --batch-size 1 --max-seq-length 4096 --logging-steps 10 --eval-steps 10 --save-steps 10 --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
python3 scripts/merge_lora.py --adapter checkpoints/lora_sweep_v3/lr2e-5_adafactor_all_r16 --out checkpoints/merged_lora_sweep_v3/lr2e-5_adafactor_all_r16
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/train_free_keep.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adafactor_all_r16/train --model checkpoints/merged_lora_sweep_v3/lr2e-5_adafactor_all_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adafactor_all_r16/train --data-path data/lora_public_v2/train_free_keep.jsonl --name lr2e-5_adafactor_all_r16_train --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adafactor_all_r16_train.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/public_dev.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adafactor_all_r16/validation --model checkpoints/merged_lora_sweep_v3/lr2e-5_adafactor_all_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adafactor_all_r16/validation --data-path data/lora_public_v2/public_dev.jsonl --name lr2e-5_adafactor_all_r16_validation --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adafactor_all_r16_validation.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/holdout_free.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adafactor_all_r16/holdout --model checkpoints/merged_lora_sweep_v3/lr2e-5_adafactor_all_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adafactor_all_r16/holdout --data-path data/lora_public_v2/holdout_free.jsonl --name lr2e-5_adafactor_all_r16_holdout --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adafactor_all_r16_holdout.jsonl
```

### lr2e-5_adamw_attention_r16

- LR/optimizer: `2e-05` / `adamw_torch`
- Rank/alpha/dropout: `16` / `32` / `0.05`
- Target modules: `q_proj, k_proj, v_proj, o_proj`
- Why: Attention-only adapter. Less capacity than all projections, useful when the data is small.

```bash
python3 scripts/train_lora_sft.py --model Qwen/Qwen3-4B-Thinking-2507 --train-file data/sft_lora_public_v2/train.jsonl --eval-file data/sft_lora_public_v2/dev.jsonl --output-dir checkpoints/lora_sweep_v3/lr2e-5_adamw_attention_r16 --epochs 1.0 --lr 2e-05 --optim adamw_torch --rank 16 --alpha 32 --dropout 0.05 --grad-accum 16 --batch-size 1 --max-seq-length 4096 --logging-steps 10 --eval-steps 10 --save-steps 10 --target-modules q_proj k_proj v_proj o_proj
python3 scripts/merge_lora.py --adapter checkpoints/lora_sweep_v3/lr2e-5_adamw_attention_r16 --out checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_attention_r16
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/train_free_keep.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_attention_r16/train --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_attention_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_attention_r16/train --data-path data/lora_public_v2/train_free_keep.jsonl --name lr2e-5_adamw_attention_r16_train --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_attention_r16_train.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/public_dev.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_attention_r16/validation --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_attention_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_attention_r16/validation --data-path data/lora_public_v2/public_dev.jsonl --name lr2e-5_adamw_attention_r16_validation --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_attention_r16_validation.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/holdout_free.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_attention_r16/holdout --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_attention_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_attention_r16/holdout --data-path data/lora_public_v2/holdout_free.jsonl --name lr2e-5_adamw_attention_r16_holdout --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_attention_r16_holdout.jsonl
```

### lr2e-5_adamw_mlp_r16

- LR/optimizer: `2e-05` / `adamw_torch`
- Rank/alpha/dropout: `16` / `32` / `0.05`
- Target modules: `gate_proj, up_proj, down_proj`
- Why: Feed-forward-only adapter. Separates reasoning/style changes from attention routing changes.

```bash
python3 scripts/train_lora_sft.py --model Qwen/Qwen3-4B-Thinking-2507 --train-file data/sft_lora_public_v2/train.jsonl --eval-file data/sft_lora_public_v2/dev.jsonl --output-dir checkpoints/lora_sweep_v3/lr2e-5_adamw_mlp_r16 --epochs 1.0 --lr 2e-05 --optim adamw_torch --rank 16 --alpha 32 --dropout 0.05 --grad-accum 16 --batch-size 1 --max-seq-length 4096 --logging-steps 10 --eval-steps 10 --save-steps 10 --target-modules gate_proj up_proj down_proj
python3 scripts/merge_lora.py --adapter checkpoints/lora_sweep_v3/lr2e-5_adamw_mlp_r16 --out checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_mlp_r16
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/train_free_keep.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_mlp_r16/train --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_mlp_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_mlp_r16/train --data-path data/lora_public_v2/train_free_keep.jsonl --name lr2e-5_adamw_mlp_r16_train --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_mlp_r16_train.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/public_dev.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_mlp_r16/validation --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_mlp_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_mlp_r16/validation --data-path data/lora_public_v2/public_dev.jsonl --name lr2e-5_adamw_mlp_r16_validation --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_mlp_r16_validation.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/holdout_free.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_mlp_r16/holdout --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_mlp_r16 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_mlp_r16/holdout --data-path data/lora_public_v2/holdout_free.jsonl --name lr2e-5_adamw_mlp_r16_holdout --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_mlp_r16_holdout.jsonl
```

### lr2e-5_adamw_all_r8

- LR/optimizer: `2e-05` / `adamw_torch`
- Rank/alpha/dropout: `8` / `16` / `0.05`
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Why: Half-rank adapter. Tests whether the 33M-parameter adapter was too large for 371 examples.

```bash
python3 scripts/train_lora_sft.py --model Qwen/Qwen3-4B-Thinking-2507 --train-file data/sft_lora_public_v2/train.jsonl --eval-file data/sft_lora_public_v2/dev.jsonl --output-dir checkpoints/lora_sweep_v3/lr2e-5_adamw_all_r8 --epochs 1.0 --lr 2e-05 --optim adamw_torch --rank 8 --alpha 16 --dropout 0.05 --grad-accum 16 --batch-size 1 --max-seq-length 4096 --logging-steps 10 --eval-steps 10 --save-steps 10 --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
python3 scripts/merge_lora.py --adapter checkpoints/lora_sweep_v3/lr2e-5_adamw_all_r8 --out checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_all_r8
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/train_free_keep.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_all_r8/train --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_all_r8 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_all_r8/train --data-path data/lora_public_v2/train_free_keep.jsonl --name lr2e-5_adamw_all_r8_train --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_all_r8_train.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/public_dev.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_all_r8/validation --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_all_r8 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_all_r8/validation --data-path data/lora_public_v2/public_dev.jsonl --name lr2e-5_adamw_all_r8_validation --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_all_r8_validation.jsonl
python3 scripts/run_math_prompts.py --mode submission_response_mode --data-path data/lora_public_v2/holdout_free.jsonl --output-dir results/lora_sweep_v3/lr2e-5_adamw_all_r8/holdout --model checkpoints/merged_lora_sweep_v3/lr2e-5_adamw_all_r8 --config greedy_n1 --max-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 --normalize-free-final-answers
python3 scripts/score_benchmark.py results/lora_sweep_v3/lr2e-5_adamw_all_r8/holdout --data-path data/lora_public_v2/holdout_free.jsonl --name lr2e-5_adamw_all_r8_holdout --summary-csv results/lora_sweep_v3/benchmark_summary.csv --scored-jsonl analysis/lora_sweep_v3_lr2e-5_adamw_all_r8_holdout.jsonl
```
