# LoRA Public V1 Same-Model Run

Training used `Qwen/Qwen3-4B-Thinking-2507` with LoRA on attention and MLP projections.

## Best Training Checkpoint

- checkpoint: `checkpoints/lora_public_v1_same_model_e6/checkpoint-150`
- epoch: `6.000`
- eval loss: `0.316569`
- eval token accuracy: `91.47%`
- train loss near best: `0.344112`
- train token accuracy near best: `91.33%`

## Holdout Check

- merged model: `checkpoints/merged_lora_public_v1_same_model_best`
- compact prompt + stage0 postprocess holdout score: `23/100 = 23.00%`
- diagnostics: `truncated=1`, `format_errors=4`, `count_mismatch=3`, `stage0_repair=93`, `avg_raw_response_chars=49.5`

The token loss curve looks good, but generation accuracy is poor because the model learned to emit very short final-answer blocks instead of doing enough reasoning.

## Files

- `loss_curve.svg`
- `token_accuracy_curve.svg`
- `summary.csv`
