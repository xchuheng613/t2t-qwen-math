# Prompting Experiment Summaries

This folder keeps compact CSV summaries from prompt and autoresearch runs after
removing the bulky per-question generation JSONL files.

Naming format:

`YYYYMMDD__experiment-name__summary.csv`

Retained summaries:

- `20260428__prompt-sweep__summary.csv`
- `20260428__prompt-sweep-best100__summary.csv`
- `20260428__compact-sample__summary.csv`
- `20260429__public-format-router__summary.csv`
- `20260503__compact-v2__summary.csv`
- `20260503__public-compact-v3__summary.csv`
- `20260526__compact-v3-expression-postprocess__summary.csv`
- `20260527__autoresearch-free-v1__summary.csv`
- `20260529__autoresearch-free__summary.csv`
- `20260529__autoresearch-free__public-free-verification.csv`

Full generated outputs are intentionally not checked in. Re-run the
corresponding scripts to regenerate them under the ignored `results/` folder.
