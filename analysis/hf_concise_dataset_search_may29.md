# Concise HF Dataset Search, May 29

Goal: replace the previous overlong/mixed free-response SFT set with a cleaner dataset whose reasoning traces are more appropriate for a 4B model.

## Local Sample

Sample seed: `529`.

I sampled 20 public and 20 private rows from `data/public.jsonl` and `data/private.jsonl`. The sample included:

- classroom algebra/arithmetic with many `[ANS]` blanks,
- statistics/probability and confidence-test style tasks,
- symbolic calculus and limits,
- unit conversions and word problems,
- exact trigonometry,
- contest-style number theory/combinatorics/geometry,
- a few MCQ-like prompts, which are not used for this free-response training set.

Representative sampled local IDs:

```text
public: 311, 484, 942, 417, 873, 460, 874, 784, 231, 290,
        848, 326, 741, 837, 836, 1113, 661, 709, 444, 394
private: 875, 495, 821, 905, 123, 131, 872, 890, 425, 475,
         173, 350, 443, 864, 725, 128, 312, 895, 158, 132
```

## Candidate Results

I inspected likely Hugging Face math datasets for topic overlap and concise, reliable reasoning traces. Rows were counted as usable only if they had a concise trace, extractable final answer, stable final-answer format, and passed the local judge when the gold answer was used as an oracle prediction.

Approximate scan results:

```text
openai/gsm8k
  scanned: 8792
  concise <=400 words: 8792
  oracle-stable: 8699
  fit: concise arithmetic word problems

EleutherAI/hendrycks_math
  scanned: 12500
  concise <=400 words: 12422
  oracle-stable: 11074 in broad scan, 8120 after exact eval-overlap exclusion and stricter generation filters
  fit: compact contest algebra, geometry, number theory, prealgebra

AI-MO/NuminaMath-CoT
  scanned first 60000 for diagnostics
  concise <=400 words: 54949
  oracle-stable: 31412 in first diagnostic window
  fit: broadest topic match, but only concise rows are kept

microsoft/orca-math-word-problems-200k
  scanned: 80000
  concise <=400 words: 79931
  reliably extractable final answers: 12
  decision: excluded because raw final answers are not reliable enough for clean SFT targets
```

## Selected Mix

The new dataset uses:

```text
AI-MO/NuminaMath-CoT:      33,500
EleutherAI/hendrycks_math:  8,000
openai/gsm8k:               8,500
total:                     50,000
```

The search set is a random 10,000-row sample from the 50,000-row full set.

## Reasoning Length

The new set caps reasoning at 400 words.

Actual audit:

```text
search set:
  rows: 10000
  avg reasoning words: 123.29
  max reasoning words: 400
  oracle failures: 0

full set:
  rows: 50000
  avg reasoning words: 123.22
  max reasoning words: 400
  oracle failures: 0
```

## Output Files

```text
data/autoresearch_free_concise_v2/search/train.jsonl
data/autoresearch_free_concise_v2/full/train.jsonl
```

Use `search/train.jsonl` for hyperparameter search and `full/train.jsonl` only for final retraining after a config is selected.

