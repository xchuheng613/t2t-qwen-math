# AutoResearch Free Concise V2

Cleaner free-response SFT data for May 29 AutoResearch experiments.

This dataset is intended to reduce overthinking compared with the previous mixed free-response set. It uses only rows with concise reasoning traces and judge-stable final answers.

## Why These Sources

I compared 20 random public examples and 20 random private examples against candidate Hugging Face math datasets by topic/format and reasoning-trace length. The local examples are a mix of classroom algebra/arithmetic, statistics, symbolic calculus, word problems, and a smaller hard contest subset.

Chosen sources:

- `AI-MO/NuminaMath-CoT`: broadest topic coverage, but only concise rows are kept.
- `EleutherAI/hendrycks_math`: compact contest algebra/geometry/number theory.
- `openai/gsm8k`: concise arithmetic word problems.

Rejected for this version:

- `microsoft/orca-math-word-problems-200k`: many concise explanations, but raw final answers are not reliably extractable enough for clean SFT targets.

## Files

- `search/train.jsonl`: 10,000 rows for hyperparameter search.
- `search/train_parts/train-0000.jsonl`: same 10,000 rows.
- `full/train.jsonl`: 50,000 rows for final retraining after selecting a config.
- `full/train_parts/*.jsonl`: same 50,000 rows split into two 25k chunks.
- `hf_search_summary.json`: source-selection summary.

Use the existing fixed evaluation files:

```bash
FREE_DEV_FILE=data/autoresearch_free_v1/dev_benchmark.jsonl
FREE_HOLDOUT_FILE=data/autoresearch_free_v1/holdout_benchmark.jsonl
```

## Filters

- Free-response only.
- Exact local dev/holdout question text excluded.
- Reasoning trace at most 400 words.
- Assistant reasoning has boxed expressions removed from the trace; only the final block contains `\boxed{...}`.
- Final answer must be stable under the local judge when predicted as:

```text
FINAL_ANSWERS:
\boxed{gold_answer}
```

## Audit

Search set:

```text
rows: 10000
bad_schema: 0
oracle_fail: 0
avg_reasoning_words: 123.29
max_reasoning_words: 400
```

Full set:

```text
rows: 50000
bad_schema: 0
oracle_fail: 0
avg_reasoning_words: 123.22
max_reasoning_words: 400
```

## Source Mix

Full set:

```text
AI-MO/NuminaMath-CoT:     33,500
EleutherAI/hendrycks_math: 8,000
openai/gsm8k:              8,500
```

Search set is a random 10k sample from the full set and preserves approximately the same proportions.

