# UWNSL Mix-Long LoRA Test

This branch adds a direct Hugging Face dataset path for the UWNSL MATH CoT
datasets and a small command wrapper for the first Mix-Long QLoRA test.

## One Epoch

Print the command:

```bash
python3 scripts/run_uwnsl_lora_experiment.py --stage epoch1
```

Run it:

```bash
python3 scripts/run_uwnsl_lora_experiment.py --stage epoch1 --run
```

Defaults:

```text
dataset: UWNSL/Mix-Long_long_0.2_short_0.8
output_dir: checkpoints/uwnsl_mix_long_lora/mix_long_r16_a32_q4
quantization_bit: 4
lora_rank: 16
lora_alpha: 32
lora_target: q/k/v/o + gate/up/down
cutoff_len: 16384
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true
flash_attn: fa2
num_train_epochs: 1
learning_rate: 1e-4
eval_strategy: epoch
```

The trainer holds out 256 rows from the HF train split for eval by default.

## Resume To Two Epochs

After the one-epoch run finishes, print the resume command:

```bash
python3 scripts/run_uwnsl_lora_experiment.py --stage epoch2
```

Run it only when ready:

```bash
python3 scripts/run_uwnsl_lora_experiment.py --stage epoch2 --run
```

`epoch2` finds the latest `checkpoint-*` under the same output directory and
runs the trainer with `--epochs 2 --resume-from-checkpoint <checkpoint>`, so
optimizer and scheduler state are preserved. The training script uses
`--save-strategy epoch` for this experiment so the epoch-1 checkpoint is usable
for this resume path.
