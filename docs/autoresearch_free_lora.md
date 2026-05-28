# autoresearch: Qwen Math Free-Response LoRA

This is an experiment to have the LLM autonomously research LoRA SFT settings for the Qwen math competition.

The current stage is **free-response only**. Do not train on MCQ examples for this run. A new free-response training dataset has already been created.

The goal is to improve free-response final-answer accuracy while preserving the current prompt/post-processing pipeline.

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**

   Propose a tag based on today's date, for example:

   ```bash
   autoresearch/free-may26
   ```

   The branch `autoresearch/<tag>` must not already exist. This should be a fresh run.

2. **Create the branch**

   ```bash
   git checkout -b autoresearch/<tag>
   ```

3. **Read the in-scope files**

   Read these files for full context:

   * `README.md` — repository context.
   * `scripts/train_lora_sft.py` — LoRA SFT training script.
   * `scripts/merge_lora.py` — LoRA merge script.
   * `scripts/create_submission.py` — generation, fallback, routing, and post-processing pipeline.
   * `scripts/score_benchmark.py` — local benchmark scoring.
   * `prompts/compact_prompt_pack.py` — current best prompt and answer-cleaning logic.
   * the current free-response training dataset.
   * the fixed dev and holdout free-response benchmark files.

4. **Verify data exists**

   Confirm the following exist:

   * free-response training JSONL.
   * free-response dev benchmark JSONL.
   * free-response holdout benchmark JSONL.
   * current best prompt/post-processing files.
   * base model path or Hugging Face model name.

   If any required file is missing, stop and tell the human exactly which file is missing.

5. **Initialize results file**

   Create:

   ```bash
   results_autoresearch_free.tsv
   ```

   with this header row:

   ```text
   commit	run_name	train_examples	lora_rank	lora_alpha	lora_dropout	learning_rate	epochs	effective_batch	max_seq_length	dev_free_acc	holdout_free_acc	truncation_count	format_errors	count_mismatch	avg_raw_response_chars	status	description
   ```

   Do not commit this file.

6. **Establish the base-model benchmark**

   Before training any LoRA model, run the current base model on the same free-response dev and holdout sets using the current best prompt/post-processing pipeline.

   This base score is the reference. A LoRA checkpoint is useful only if it beats the base model on generated final-answer accuracy after post-processing.

---

## Experiment goal

The goal is **not** lowest token validation loss.

The goal is:

```text
highest generated free-response final-answer accuracy
after current prompt + routing + fallback + post-processing + official/local judge
```

Model selection must be based on final-answer exact-match accuracy, not token-level validation loss.

Use token loss only as a sanity check that training is running.

---

## Fixed constraints

Do not modify:

* official/local judge logic.
* dev/holdout split.
* private data.
* scoring metric.
* benchmark answer files.
* current best prompt/post-processing pipeline unless explicitly running a separate post-processing experiment.
* MCQ data or MCQ evaluation in this free-response-only run.

Do not use MCQ examples in training for this run.

Do not select checkpoints by training loss, validation loss, or token accuracy.

Do not use holdout for repeated hyperparameter search. The dev set is for search. Holdout is for final confirmation.

---

## What you can modify

You may modify training-related code and config, including:

* LoRA rank.
* LoRA alpha.
* LoRA dropout.
* learning rate.
* number of epochs.
* effective batch size.
* gradient accumulation.
* max sequence length.
* target modules.
* optimizer settings.
* warmup ratio.
* weight decay.
* data subset size.
* data mixture within free-response examples.
* reasoning-target length buckets.

Keep changes simple and reversible.

---

## Starting hyperparameter search space

Use a constrained search space. Do not try random extreme settings.

Allowed values:

```yaml
lora_rank: [8, 16, 32]
lora_alpha: [16, 32, 64]
lora_dropout: [0.0, 0.05, 0.1]
learning_rate: [1e-5, 2e-5, 5e-5]
epochs: [1, 2]
effective_batch_size: [32, 64]
max_seq_length: [4096, 8192]
warmup_ratio: [0.03, 0.05]
weight_decay: [0.0, 0.01]
```

Target module options:

```yaml
attention_only:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

attention_mlp:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

Default first config:

```yaml
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2e-5
epochs: 1
effective_batch_size: 32
max_seq_length: 4096
target_modules: attention_mlp
warmup_ratio: 0.03
weight_decay: 0.0
```

Avoid unless there is strong evidence:

```yaml
learning_rate: 1e-4
epochs: 3+
lora_rank: 64+
max_seq_length: 16384+
```

Previous small-data experiments showed that aggressive LoRA settings can reduce generated accuracy even when validation loss looks good.

---

## Data rules

This run is **free-response only**.

Use the new free-response dataset already prepared by the user.

The training examples should use the format:

```text
<think>
concise reasoning
</think>

FINAL_ANSWERS:
\boxed{answer}
```

Reasoning length guidance:

* Easy problems: 30–80 words.
* Medium problems: 80–200 words.
* Hard/statistics/multi-blank problems: 200–500 words.
* Absolute maximum: 700 words.

Do not train on very long reasoning unless the problem truly needs it.

Reject or downsample examples with:

* missing final answer.
* malformed `FINAL_ANSWERS`.
* missing `\boxed{}`.
* very long rambling reasoning.
* answer-count mismatch.
* known bad gold answers.
* non-free-response format.
* MCQ options or MCQ letter-only targets.

---

## Evaluation protocol

Every model must be evaluated by generation, not token loss.

For each candidate checkpoint:

1. Merge or load the LoRA adapter.
2. Generate answers on the free-response dev set.
3. Run the current post-processing pipeline.
4. Score with `scripts/score_benchmark.py`.
5. Record:

   * dev free accuracy.
   * truncation count.
   * format errors.
   * count mismatch.
   * average raw response length.

Only the top few models on dev should be evaluated on holdout.

Holdout is for final confirmation only.

Keep a model only if it improves holdout free-response accuracy over the base model or gives a very clear diagnostic improvement without hurting accuracy.

---

## First runs

The first run should establish the base model benchmark.

Then try these seed configurations:

### Config A: conservative

```yaml
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
learning_rate: 2e-5
epochs: 1
effective_batch_size: 32
max_seq_length: 4096
target_modules: attention_mlp
```

### Config B: standard

```yaml
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2e-5
epochs: 1
effective_batch_size: 32
max_seq_length: 4096
target_modules: attention_mlp
```

### Config C: slightly stronger

```yaml
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 5e-5
epochs: 1
effective_batch_size: 32
max_seq_length: 4096
target_modules: attention_mlp
```

### Config D: higher rank

```yaml
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
learning_rate: 2e-5
epochs: 1
effective_batch_size: 32
max_seq_length: 4096
target_modules: attention_mlp
```

---

## Commands

Use the existing training script.

Example training command:

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

After training, merge the LoRA:

```bash
python scripts/merge_lora.py \
  --base-model "$BASE_MODEL" \
  --adapter "checkpoints/<run_name>" \
  --out "checkpoints/merged_<run_name>"
```

Evaluate on dev:

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

Evaluate on holdout only for top dev models:

```bash
python scripts/create_submission.py \
  --model "checkpoints/merged_<run_name>" \
  --data-path "$FREE_HOLDOUT_FILE" \
  --output-dir "results/autoresearch/<run_name>_holdout" \
  $COMMON_POST

python scripts/score_benchmark.py \
  "results/autoresearch/<run_name>_holdout" \
  --data-path "$FREE_HOLDOUT_FILE" \
  --name "<run_name>_holdout" \
  --summary-csv "results/autoresearch/summary.csv"
```

---

## Logging results

When an experiment is done, log it to `results_autoresearch_free.tsv`.

Use tab-separated values only.

Columns:

```text
commit	run_name	train_examples	lora_rank	lora_alpha	lora_dropout	learning_rate	epochs	effective_batch	max_seq_length	dev_free_acc	holdout_free_acc	truncation_count	format_errors	count_mismatch	avg_raw_response_chars	status	description
```

Status values:

```text
keep
discard
crash
review
```

Use:

* `keep` if the model improves holdout free accuracy.
* `review` if dev improves but holdout has not yet been checked.
* `discard` if dev or holdout is worse than base.
* `crash` if the run fails.

Do not commit the TSV file.

---

## Keep/discard rule

A run should be kept only if it gives a meaningful improvement in generated final-answer accuracy.

Primary rule:

```text
Keep if holdout_free_acc > base_holdout_free_acc
```

Stronger rule:

```text
Prefer keeping only if holdout improves by at least 2–3 questions per 100 examples.
```

Discard if:

* dev accuracy drops clearly.
* holdout accuracy drops.
* truncation increases sharply.
* format errors increase.
* count mismatch increases.
* average raw response length collapses below 1000 characters.
* average raw response length explodes and causes truncation.
* model learns to output short guesses instead of reasoning.

Do not keep a model just because token loss is low.

---

## Experiment loop

The experiment runs on a dedicated branch, for example:

```bash
autoresearch/free-may26
```

Loop:

1. Check git state.
2. Choose one experimental idea.
3. Modify only training/config-related code.
4. Commit the change.
5. Train LoRA.
6. Merge LoRA.
7. Evaluate on dev free-response accuracy.
8. Record results in TSV.
9. If dev improves meaningfully, evaluate on holdout.
10. If holdout improves, keep the commit.
11. If holdout does not improve, reset or mark as discard.
12. Continue with the next idea.

Do not let the LLM optimize on holdout repeatedly.

Use dev for exploration. Use holdout only for confirmation.

---

## Suggested experiment ideas

Try these in order:

1. Baseline base model dev/holdout.
2. Config A.
3. Config B.
4. Config C.
5. Config D.
6. Compare attention-only vs attention+MLP target modules.
7. Compare dropout 0.0 vs 0.05 vs 0.1.
8. Compare effective batch size 32 vs 64.
9. Compare max sequence length 4096 vs 8192.
10. Compare data subsets by reasoning length:

    * concise only.
    * medium only.
    * mixed concise + medium.
11. Compare data subsets by problem type:

    * numeric.
    * formula/symbolic.
    * statistics/precision.
    * multi-blank.
12. Try a lower learning rate if outputs become unstable.
13. Try smaller rank if generated accuracy drops while loss improves.
14. Try larger data subset only if smaller runs show improvement.

---

## Failure handling

If a run crashes:

1. Inspect the log.
2. If it is a simple typo or missing argument, fix and rerun once.
3. If it is OOM, reduce batch size, sequence length, or rank.
4. If it still fails, mark the run as `crash`.
5. Do not silently skip failed experiments.

If a run takes unexpectedly long, stop it and mark as failed unless there is a clear reason to continue.

---

## Simplicity criterion

All else being equal, simpler is better.

Prefer:

* smaller rank.
* fewer epochs.
* lower learning rate.
* shorter sequence length.
* fewer special cases.

Do not keep complex changes for tiny improvements.

A small accuracy gain from a simple change is valuable. A tiny gain from a fragile complicated change is not worth keeping.

---

## Final decision rule

The final model for this stage should be chosen by:

```text
free-response holdout exact-match accuracy after generation and post-processing
```

not by:

```text
training loss
validation loss
token accuracy
eval_mean_token_accuracy
```

If no LoRA model beats the base model on holdout, keep the base model and focus on prompt/post-processing/error analysis instead of forcing LoRA.

---

## Repository-Specific Configuration

Use these concrete files for the first AutoResearch stage:

```bash
FREE_TRAIN_FILE=data/autoresearch_free_v1/train.jsonl
FREE_DEV_SFT_FILE=data/autoresearch_free_v1/dev_sft.jsonl
FREE_DEV_FILE=data/autoresearch_free_v1/dev_benchmark.jsonl
FREE_HOLDOUT_FILE=data/autoresearch_free_v1/holdout_benchmark.jsonl
FULL_FREE_TRAIN_FILE=data/hf_mixed_math_free_100k/train.jsonl
BASE_MODEL=Qwen/Qwen3-4B-Thinking-2507
```

`FREE_DEV_SFT_FILE` is for token-loss sanity checks only. Model selection should use generated-answer accuracy on `FREE_DEV_FILE`.

For effective batch size 32, use:

```bash
--batch-size 1 --grad-accum 32
```

For effective batch size 64, use:

```bash
--batch-size 1 --grad-accum 64
```

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

## Benchmark Sanity

Fresh oracle scoring was run for the prepared benchmark files:

```text
dev_benchmark:     1000/1000 = 100.00%
holdout_benchmark: 1000/1000 = 100.00%
```

This confirms the benchmark files are compatible with the current local scoring pipeline.

The 5,000-row training split was also audited with the same local judge extraction path:

```text
flagged answer-format patterns: 0
extraction failures: 0
oracle scoring failures: 0
```

## Search-Then-Full-Train Policy

Use `data/autoresearch_free_v1/train.jsonl` for first-stage hyperparameter search.

After selecting the best simple configuration by generated dev accuracy and confirming once on holdout, retrain that configuration on the full free-response dataset:

```bash
FULL_FREE_TRAIN_FILE=data/hf_mixed_math_free_100k/train.jsonl
```

## Suggested Run Tag

Suggested fresh branch tag for the actual AutoResearch loop:

```bash
autoresearch/free-may27
```

Before creating it, verify it does not already exist:

```bash
git branch --list "autoresearch/free-may27"
git ls-remote --heads origin "autoresearch/free-may27"
```

