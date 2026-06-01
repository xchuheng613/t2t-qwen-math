# 2026-05-31 Final Submission

This folder contains the final private-set submission artifacts with role-based
names. The Kaggle CSV to submit is:

- `final_private_submission.csv`

Supporting files:

- `final_private_submission_audit.jsonl`: row-level audit for the final merge.
- `final_private_merge_validation.json`: merge checks for row count, ids,
  duplicates, and empty responses.
- `mcq_like_source_submission.csv`: source submission used for MCQ-like rows.
- `grpo_free_source_submission.csv`: source submission used for GRPO
  free-response rows.
- `legacy_private_submission.csv`: older private submission retained only for
  comparison.
- `mcq_like_compact_*` and `options_only_compact_*`: alternate hybrid builds
  retained for reproducibility checks.

The main reproducible path is still `run_inference.run_inference()`. These
files are archived outputs from the submitted run.

Submitted-run inference parameters:

- `max_tokens=16384`
- `fallback_max_tokens=8192`
- `fallback_tail_tokens=6000`
- `max_model_len=32768`
- `gpu_memory_utilization=0.85`
- `max_num_seqs=32`
- `max_num_batched_tokens=16384`
- `enforce_eager=False`
- MCQ-like rows: base `Qwen/Qwen3-4B-Thinking-2507`, compact prompt,
  `greedy_n1`
- Free-response rows: GRPO checkpoint, compact prompt, `sc_n3`

As of this archive update, `run_inference.py` defaults to a larger GRPO-only
budget for both MCQ-like and free-response rows (`max_tokens=81920`,
`fallback_max_tokens=81920`, `max_model_len=262144`). Pass the parameters above
plus `--mcq-model-id Qwen/Qwen3-4B-Thinking-2507` explicitly to reproduce the
submitted 32GB balanced hybrid run.
