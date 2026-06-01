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
