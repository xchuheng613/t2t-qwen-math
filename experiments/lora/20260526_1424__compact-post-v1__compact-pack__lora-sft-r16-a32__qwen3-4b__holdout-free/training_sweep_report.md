# LoRA Training Sweep Report

Best validation checkpoint:

- experiment: `lr5e-5_adamw_all_r16_e6`
- checkpoint: `checkpoints/lora_training_sweep_compact_post_v1/lr5e-5_adamw_all_r16_e6/checkpoint-140`
- best epoch: `5.863`
- best step: `140`
- best validation loss: `0.319393`
- best validation token accuracy: `91.44%`
- train loss at best validation point: `0.359769`
- train token accuracy at best validation point: `91.20%`

Recommended hyperparameters:

- optimizer: `adamw_torch`
- learning rate: `5e-05`
- LoRA rank / alpha / dropout: `16 / 32 / 0.05`
- target modules: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`
- stop around epoch: `5.86`; the 4-epoch run's final eval loss was `0.319399`, so later epochs did not improve validation loss.

Plots:

- `training_sweep_best_train_eval_loss.svg`
- `training_sweep_best_token_accuracy.svg`
- `training_sweep_eval_loss_comparison.svg`

## Results

| Experiment | Best epoch | Best eval loss | Eval token acc | Train loss at best | Targets |
| --- | ---: | ---: | ---: | ---: | --- |
| lr5e-5_adamw_all_r16_e6 | 5.863 | 0.3194 | 91.44% | 0.3598 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
| lr5e-5_adamw_all_r16_e4 | 4.000 | 0.3416 | 91.01% | 0.4104 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
| lr5e-5_adamw_all_r16_e2 | 2.000 | 0.7514 | 83.00% | 1.0428 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
| lr5e-5_adamw_attention_r16_e2 | 2.000 | 1.6881 | 59.95% | 1.7996 | q_proj k_proj v_proj o_proj |
| lr2e-5_adamw_mlp_r16_e2 | 2.000 | 2.3613 | 51.49% | 2.3697 | gate_proj up_proj down_proj |
| lr2e-5_adamw_all_r8_e2 | 2.000 | 2.4900 | 50.50% | 2.4775 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
| lr2e-5_adamw_attention_r16_e2 | 2.000 | 2.5051 | 50.41% | 2.4776 | q_proj k_proj v_proj o_proj |
| lr2e-5_adafactor_all_r16_e1 | 1.000 | 2.5603 | 50.39% | 2.5971 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
| lr1e-5_adamw_all_r16_e2 | 2.000 | 2.6174 | 49.52% | 2.5773 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
| lr2e-5_adamw_all_r16_e1 | 1.000 | 2.6565 | 49.30% | 2.6572 | q_proj k_proj v_proj o_proj gate_proj up_proj down_proj |
