# LoRA Experiment Summary for TA/Professor

Date: 2026-05-26

Repo: `t2t-qwen-math`

## Short version

I tried several LoRA training setups on a Qwen math model, but the LoRA is not clearly improving final-answer accuracy. The strongest local signal is that LoRA can slightly improve free-response on one public dev split, but it hurts MCQ and does not beat the base model on my 100-example holdout-free split.

Current best practical local strategy has been hybrid routing:

- Use the base model for MCQ-like questions.
- Use LoRA only for free-response questions.

However, even that conclusion is shaky because:

- On public dev, the hybrid was best: `134/180 = 74.44%`.
- On holdout-free, base beat LoRA slightly: base `66/100`, LoRA `65/100`.
- On the private leaderboard, base-only scored `69.2`, the older MCQ+LoRA hybrid also scored `69.2`, and the newer MCQ+LoRA hybrid scored `68.9`.
- Assistant-only-loss LoRA runs had better-looking token losses, but worse generation accuracy.

My current suspicion is that the train objective is not matching the leaderboard objective well enough. The leaderboard grades final answers, while several LoRA runs optimized teacher-solution tokens or compact answer-format tokens. Token-level validation loss has not reliably predicted final-answer generation accuracy.

## Base model and data

Base model used by the LoRA adapters:

`Qwen/Qwen3-4B-Thinking-2507`

Main LoRA SFT data:

| Split/path | Rows |
|---|---:|
| `data/sft_lora_public_v2/train.jsonl` | 371 |
| `data/sft_lora_public_v2/dev.jsonl` | 100 |
| `data/lora_public_v2/public_dev.jsonl` | 180 |
| `data/lora_public_v2/holdout_free.jsonl` | 100 |
| `data/sft_lora_compact_post_v1/train.jsonl` | 371 |
| `data/sft_lora_compact_post_v1/dev.jsonl` | 100 |
| `data/sft_lora_compact_post_v1/holdout.jsonl` | 100 |

The 180-example public dev set has:

- 81 MCQ-style examples.
- 99 free-response examples.

## Training setup

Training script:

`scripts/train_lora_sft.py`

Main LoRA setup used in the best sweep:

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Thinking-2507` |
| LoRA rank | `r=16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Optimizer | mostly `adamw_torch`; also tried `adafactor` |
| Learning rates tried | `1e-5`, `2e-5`, `5e-5` |
| Target modules, all-module LoRA | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Target modules, attention-only | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Target modules, MLP-only | `gate_proj`, `up_proj`, `down_proj` |
| Epochs tried | 1, 2, 4, 6 depending on sweep |

Important implementation detail:

Earlier flattened-message runs were effectively full flattened-chat loss. I later patched `scripts/train_lora_sft.py` to support manual assistant-only labels for flattened chat rows using tokenizer offset mappings, because the standard assistant-only-loss path was unreliable in this environment.

## Main old/full-chat LoRA sweep

Sweep artifact:

`analysis/lora_training_sweep_v3/summary.csv`

This was trained on `data/sft_lora_public_v2`.

Best token-validation run:

| Setting | Value |
|---|---|
| Experiment | `lr5e-5_adamw_all_r16_e4` |
| LR | `5e-5` |
| Optimizer | `adamw_torch` |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05` |
| Modules | attention + MLP projections |
| Best checkpoint | step 40 |
| Best epoch | 1.69 |
| Train loss at best eval | 0.6485 |
| Train token accuracy at best eval | 84.57% |
| Eval loss | 2.1307 |
| Eval token accuracy | 60.02% |

Other sweep results:

| Experiment | Eval loss | Eval token acc | Note |
|---|---:|---:|---|
| `lr5e-5_adamw_all_r16_e4` | 2.1307 | 60.02% | Best token eval |
| `lr5e-5_adamw_all_r16_e2` | 2.1751 | 58.57% | Slightly worse |
| `lr5e-5_adamw_attention_r16_e2` | 2.3590 | 54.12% | Attention-only worse |
| `lr2e-5_adamw_all_r16_e1` | 2.8905 | 47.46% | Lower LR worse |
| `lr1e-5_adamw_all_r16_e2` | 2.8758 | 47.75% | Lower LR worse |
| `lr2e-5_adamw_all_r8_e2` | 2.8026 | 48.66% | Lower rank worse |
| `lr2e-5_adamw_attention_r16_e2` | 2.7751 | 48.42% | Attention-only lower LR worse |
| `lr2e-5_adamw_mlp_r16_e2` | 2.7802 | 49.19% | MLP-only lower LR worse |
| `lr2e-5_adafactor_all_r16_e1` | 2.8873 | 48.25% | Adafactor worse |

Takeaway from this sweep:

Lower LR did not help token validation loss in this setup. The best token-validation setting was still `5e-5`, `adamw_torch`, all attention + MLP LoRA modules, rank 16.

## Generation-level results for old LoRA B

Benchmark artifact:

`results/lora_bench_v2/benchmark_summary.csv`

| Run | Data | Correct | Accuracy | MCQ | Free |
|---|---|---:|---:|---:|---:|
| Base | public dev | 132/180 | 73.33% | 66/81 = 81.48% | 66/99 = 66.67% |
| LoRA B | public dev | 129/180 | 71.67% | 61/81 = 75.31% | 68/99 = 68.69% |
| Hybrid base-MCQ + LoRA-free | public dev | 134/180 | 74.44% | 66/81 = 81.48% | 68/99 = 68.69% |
| Base | holdout-free | 66/100 | 66.00% | n/a | 66/100 = 66.00% |
| LoRA B | holdout-free | 65/100 | 65.00% | n/a | 65/100 = 65.00% |

Takeaway:

LoRA B slightly helped free-response on public dev by 2 examples, but hurt MCQ by 5 examples and did not beat base on holdout-free.

## Compact post / final-answer-style LoRA

Sweep artifact:

`analysis/lora_training_sweep_compact_post_v1/summary.csv`

This used the newer compact prompt / postprocessing style data.

Best token-validation run:

| Setting | Value |
|---|---|
| Experiment | `lr5e-5_adamw_all_r16_e6` |
| Best checkpoint | step 140 |
| Best epoch | 5.86 |
| Eval loss | 0.3194 |
| Eval token accuracy | 91.44% |
| Train loss | 0.3598 |
| Train token accuracy | 91.20% |

But generation accuracy was poor:

| Run | Data | Correct | Accuracy |
|---|---|---:|---:|
| `lora_compact_post_v1_best_holdout_free` | `data/lora_public_v2/holdout_free.jsonl` | 25/100 | 25.00% |
| `lora_public_v1_same_model_holdout_free` | `data/lora_public_v1/holdout_free.jsonl` | 23/100 | 23.00% |
| `lora_best_epoch1p69_holdout_compact_postprocess` | `data/public_holdout.jsonl` | 53/100 | 53.00% |

Takeaway:

This is the clearest example that token loss/accuracy was not predictive of final-answer benchmark performance. The compact target was easy to fit, but generation quality got much worse.

## Assistant-only-loss runs

After patching `scripts/train_lora_sft.py`, I trained with manual assistant-only labels for flattened chat data.

High-LR assistant-only run:

| Setting | Value |
|---|---|
| Checkpoint dir | `checkpoints/lora_public_v2_assistant_loss_e4` |
| LR | `5e-5` |
| Epochs | 4 |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05` |
| Modules | attention + MLP projections |

Token validation curve:

| Step | Epoch | Eval loss | Eval token acc |
|---:|---:|---:|---:|
| 10 | 0.43 | 1.9918 | 75.71% |
| 20 | 0.86 | 1.3506 | 76.48% |
| 30 | 1.26 | 1.6019 | 76.37% |
| 40 | 1.69 | 1.7841 | 76.31% |
| 96 | 4.00 | 1.9716 | 76.60% |

Generation benchmarks:

| Run | Data | Correct | Accuracy |
|---|---|---:|---:|
| `assistant_step20_holdout_free` | holdout-free | 59/100 | 59.00% |
| `assistant_step30_holdout_free` | holdout-free | 46/100 | 46.00% |
| `assistant_step40_holdout_free` | holdout-free | 48/100 | 48.00% |
| `assistant_step96_holdout_free` | holdout-free | 50/100 | 50.00% |
| `assistant_step20_public_dev` | public dev | 111/180 | 61.67% |

Lower-LR assistant-only run:

| Setting | Value |
|---|---|
| Checkpoint dir | `checkpoints/lora_public_v2_assistant_loss_lr2em5_e2` |
| LR | `2e-5` |
| Epochs | 2 |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05` |
| Modules | attention + MLP projections |

Token validation improved through the run:

| Step | Epoch | Eval loss | Eval token acc |
|---:|---:|---:|---:|
| 5 | 0.22 | 3.0353 | 70.94% |
| 20 | 0.86 | 2.1911 | 75.22% |
| 35 | 1.47 | 1.7926 | 76.08% |
| 48 | 2.00 | 1.7315 | 75.98% |

Generation benchmarks:

| Run | Data | Correct | Accuracy |
|---|---|---:|---:|
| `assistant_lr2em5_step25_holdout_free` | holdout-free | 55/100 | 55.00% |
| `assistant_lr2em5_step35_holdout_free` | holdout-free | 55/100 | 55.00% |
| `assistant_lr2em5_step48_holdout_free` | holdout-free | 56/100 | 56.00% |

I also started/trained an even lower LR run:

| Setting | Value |
|---|---|
| Checkpoint dir | `checkpoints/lora_public_v2_assistant_loss_lr1em5_e2` |
| LR | `1e-5` |
| Epochs | 2 |
| Final step | 48 |
| Final eval loss | 2.4085 |
| Final eval token acc | 72.06% |

Takeaway:

Assistant-only loss did not solve the problem. At `5e-5`, it overfit/collapsed generation after the early checkpoint. At `2e-5`, token loss kept improving, but final-answer accuracy stayed below base and below old LoRA B. At `1e-5`, it looked underfit by token metrics.

## Private submission routing experiments

Private set:

`data/private.jsonl`

Total private rows: 943.

I tried two submission routing strategies:

| Submission dir | Routing | Counts |
|---|---|---|
| `results/private_base_options_lora_B_no_options_upload` | Base for explicit `options`; LoRA for no-options | 300 base, 643 LoRA |
| `results/private_base_mcq_lora_B_free_old97_upload` | Old MCQ-like heuristic: base for explicit options and inline A/B-like rows; LoRA for free-like | 397 base, 546 LoRA |

The second one keeps the 97 no-options-but-MCQ-like rows on the base model side. This matches the older routing logic.

Private leaderboard results from these routing experiments:

| Submission style | Local routing interpretation | Leaderboard score |
|---|---|---:|
| Base-only | Same latest prompt/postprocessing, no LoRA | 69.2 |
| Older MCQ+LoRA hybrid | Base for MCQ-like rows, LoRA for free-like rows | 69.2 |
| Newer MCQ+LoRA hybrid | Base for explicit `options`, LoRA for all no-options rows | 68.9 |

This makes the routing issue more concrete: the older hybrid tied base-only, while the newer version sent 97 no-options-but-MCQ-like rows to LoRA and dropped slightly. That does not prove all 97 rows caused the drop, but it is consistent with the local finding that LoRA hurts MCQ-like behavior. Since base-only ties the best LoRA hybrid on the private leaderboard, there is currently no clear leaderboard evidence that LoRA helps.

## Why I think LoRA is not improving

My current hypotheses:

1. The dataset is very small for LoRA.
   - Only 371 train examples.
   - Qwen already has strong math priors, so a small LoRA can easily overwrite useful behavior instead of improving it.

2. Token validation loss is not aligned with leaderboard accuracy.
   - The leaderboard only checks final answers.
   - Some runs fit answer-format or teacher-solution tokens well, but generation-level exact-answer accuracy got worse.

3. Teacher solutions may teach style more than correctness.
   - Training on multi-step reasoning can reduce loss while still not improving final extracted answers.
   - Compact final-answer training made token metrics excellent but hurt generation badly.

4. MCQ and free-response behave differently.
   - LoRA B hurt MCQ on public dev: base `66/81`, LoRA `61/81`.
   - LoRA B slightly helped free-response on public dev: base `66/99`, LoRA `68/99`.
   - That is why hybrid routing looked better locally.

5. Assistant-only loss did not help enough.
   - It avoids training on user/system prompt tokens, which should be cleaner in theory.
   - In practice, generation accuracy was worse than base on holdout-free.

6. Learning rate changes did not fix it.
   - In the old sweep, lower LR was worse by token eval.
   - In assistant-only training, lower LR improved token loss gradually but did not improve final-answer accuracy.

7. Prompting and postprocessing seem to matter more than LoRA right now.
   - The base model plus postprocessing is competitive or better on holdout-free.
   - Hybrid routing helps on public dev mostly because it avoids using LoRA for MCQ.

8. The private leaderboard is also not showing a LoRA win.
   - Base-only scored `69.2`.
   - Older MCQ+LoRA routing scored `69.2`.
   - Newer MCQ+LoRA routing scored `68.9`.
   - The difference is small, but it points the same direction as the local tests: LoRA is not robustly improving the final answer metric and may hurt when routed onto MCQ-like rows.

## What I would do now

I would pause broad LoRA hyperparameter sweeps and ask for guidance before spending more GPU time. I would also treat the current LoRA as suspicious unless it beats the base model on the exact submission-style generation and postprocessing path. Since base-only ties the best private leaderboard score seen so far, I would prefer base-only for the next serious submission unless there is a new LoRA result that clearly beats it.

If continuing, I would select checkpoints using generation-level exact-answer accuracy, not token validation loss. The minimum useful loop would be:

1. Train one candidate LoRA.
2. Merge the checkpoint.
3. Run the same prompt and postprocessing as the actual submission path.
4. Score final extracted answers on public dev and holdout-free.
5. Pick the checkpoint only if it beats the base model on final-answer accuracy.

Given the current evidence, I would not assume more epochs or lower LR will fix it.

## Questions I want to ask

1. Since the leaderboard only grades final answers, is it valid/recommended to train only on final-answer outputs instead of teacher reasoning steps?

2. Should checkpoint selection be based on token-level validation loss, or final-answer exact-match accuracy after generation and postprocessing?

3. Is 371 training examples too small for this LoRA setup on `Qwen/Qwen3-4B-Thinking-2507`?

4. For this assignment, are external similar math datasets allowed for augmentation? If yes, what kinds are acceptable?

5. Do you expect assistant-only loss to be necessary here, or is full chat/message loss acceptable?

6. Is it okay to use different model routes for different problem types, for example base model for MCQ and LoRA for free-response?

7. Are there known formatting/scoring pitfalls where local exact-match scoring may disagree with leaderboard scoring?

8. If the base model is already strong, should I focus more on prompt/postprocessing/error analysis instead of LoRA?

## Main artifacts in the repo

Training/sweep summaries:

- `analysis/lora_training_sweep_v3/summary.csv`
- `analysis/lora_training_sweep_compact_post_v1/summary.csv`
- `analysis/lora_public_v2_assistant_loss/`
- `analysis/lora_public_v2_assistant_loss_low_lr/`

Benchmark summaries:

- `results/lora_bench_v2/benchmark_summary.csv`
- `results/lora_assistant_loss_benchmark.csv`
- `results/lora_assistant_loss_low_lr_benchmark.csv`
- `results/lora_compact_post_v1_benchmark.csv`
- `results/lora_public_v1_same_model_benchmark.csv`
- `results/lora_best_epoch1p69_compact_benchmark.csv`

Current submission candidates:

- `results/private_base_mcq_lora_B_free_old97_upload/submission.csv`
- `results/private_base_options_lora_B_no_options_upload/submission.csv`

Relevant scripts:

- `scripts/train_lora_sft.py`
- `scripts/run_math_prompts.py`
- `scripts/score_benchmark.py`
- `scripts/build_options_lora_hybrid_submission.py`
