# Representative LoRA Experiments

This directory keeps the small set of LoRA/QLoRA artifacts that explain the
main tuning decisions. Generated datasets, duplicate per-step outputs, private
submission variants, and verbose scratch logs were removed from the public
branch to keep the repository reviewable.

Naming format:

```text
YYYYMMDD_HHMM__dataset-or-source__prompt-pack__tuning-method__model__split
```

## Summary

| Directory | Approach | Representative result | Why kept |
|---|---|---:|---|
| `20260526_1424__public-v1-free__compact-pack__lora-full-chat-r16-a32__qwen3-4b__holdout-free` | Early full-chat LoRA SFT on public-v1 free-response data. | `23/100` holdout free. | Shows the first failure mode: token metrics looked good but generated answers were too short and inaccurate. |
| `20260520_1646__public-v2-free__compact-pack__lora-b-r16-a32__qwen3-4b__dev-holdout` | Main public-v2 LoRA-B run plus base and hybrid comparisons. | Hybrid dev `134/180`; LoRA holdout free `65/100` vs base `66/100`. | Most important diagnostic run; it explains why final routing did not rely on this LoRA for every row. |
| `20260526_2037__public-v2-assistant-loss__compact-pack__lora-step20__qwen3-4b__dev-holdout` | Assistant-only-loss LoRA checkpoint selection. | Step20 holdout free `59/100`; public dev `111/180`. | Best non-low-LR assistant-loss checkpoint, but still below the base free-response holdout. |
| `20260526_2037__public-v2-assistant-loss-low-lr__compact-pack__lora-lr2e-5-step48__qwen3-4b__holdout-free` | Lower-LR assistant-only-loss retry. | Step48 holdout free `56/100`. | Shows lower LR reduced collapse but still did not beat the base model. |
| `20260526_1424__compact-post-v1__compact-pack__lora-sft-r16-a32__qwen3-4b__holdout-free` | Compact final-answer post-processing SFT. | Holdout free `25/100`. | Shows why low token loss and high token accuracy were not trusted as final model-selection metrics. |
| `20260530_2252__uwnsl-mix-long__compact-pack__qlora-r16-a32-4bit__qwen3-4b__public-free` | External UWNSL Mix-Long 4-bit QLoRA. | Public free-response validation `326/655 = 49.77%`. | Represents the larger external-dataset approach; useful context for why final performance came from GRPO rather than LoRA. |

## Selection Rule

The kept artifacts are limited to:

- run reports or runbooks;
- benchmark or score summaries;
- selected submissions and scored outputs needed to verify the stated results;
- plots and compact diagnostics that support the conclusions.

Full checkpoints are intentionally not committed. If a LoRA model is needed
again, retrain from the scripts in `scripts/` or load the corresponding model
from external storage.
