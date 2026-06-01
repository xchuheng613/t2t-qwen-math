# LoRA Training Sweep Report

Best validation checkpoint:

- experiment: `lr5e-5_adamw_all_r16_e4`
- checkpoint: `checkpoints/lora_training_sweep_v3/lr5e-5_adamw_all_r16_e4/checkpoint-40`
- best epoch: `1.690`
- best step: `40`
- best validation loss: `2.130682`
- best validation token accuracy: `60.02%`
- train loss at best validation point: `0.648491`
- train token accuracy at best validation point: `84.57%`

Recommended hyperparameters:

- optimizer: `adamw_torch`
- learning rate: `5e-05`
- LoRA rank / alpha / dropout: `16 / 32 / 0.05`
- target modules: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`
- stop around epoch: `1.69`; the 4-epoch run's final eval loss was `2.145944`, so later epochs did not improve validation loss.

Plots:

- `training_sweep_best_train_eval_loss.svg`
- `training_sweep_best_token_accuracy.svg`
- `training_sweep_eval_loss_comparison.svg`

## Results

| Experiment | Best epoch | Best eval loss | Eval token acc | Train loss at best | Targets |
| --- | ---: | ---: | ---: | ---: | --- |
| lr5e-5_adamw_all_r16_e4 | 1.690 | 2.1307 | 60.02% | 0.6485 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
| lr5e-5_adamw_all_r16_e2 | 2.000 | 2.1751 | 58.57% | 0.6961 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
| lr5e-5_adamw_attention_r16_e2 | 2.000 | 2.3590 | 54.12% | 1.1573 | q_proj k_proj v_proj o_proj |
| lr2e-5_adamw_attention_r16_e2 | 2.000 | 2.7751 | 48.42% | 1.7282 | q_proj k_proj v_proj o_proj |
| lr2e-5_adamw_mlp_r16_e2 | 2.000 | 2.7802 | 49.19% | 1.6156 | gate_proj up_proj down_proj |
| lr2e-5_adamw_all_r8_e2 | 2.000 | 2.8026 | 48.66% | 1.7146 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
| lr1e-5_adamw_all_r16_e2 | 2.000 | 2.8758 | 47.75% | 1.8048 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
| lr2e-5_adafactor_all_r16_e1 | 1.000 | 2.8873 | 48.25% | 1.7964 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
| lr2e-5_adamw_all_r16_e1 | 1.000 | 2.8905 | 47.46% | 1.8881 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
