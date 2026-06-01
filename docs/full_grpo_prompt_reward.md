# Full GRPO Prompt And Reward

Active module:

```python
from prompts.grpo_prompt_pack import (
    make_grpo_dataset_rows,
    full_grpo_reward,
    correctness_reward,
    format_reward,
    length_reward,
)
```

Dataset conversion:

```python
from datasets import Dataset
from pathlib import Path
from scripts.lora_reasoning_common import load_jsonl
from prompts.grpo_prompt_pack import make_grpo_dataset_rows

rows = load_jsonl(Path("data/public_train.jsonl"))
train_dataset = Dataset.from_list(make_grpo_dataset_rows(rows))
```

Direct reward for `GRPOTrainer`:

```python
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=[full_grpo_reward],
    train_dataset=train_dataset,
)
```

Reward weights:

```text
correctness: 2.0  exact match after compact postprocess + Judger
format:      0.4  FINAL_ANSWERS + one final boxed answer + right answer count
length:      0.2  discourages very long / truncation-prone outputs
```

The prompt is intentionally shorter than the submission prompt. It keeps the
same final-answer contract but removes extra instruction text so GRPO rollouts
spend less budget on prompt tokens and get a denser format signal.

The cloud training entrypoint is `scripts/train_grpo_full.py`. Its main-run
defaults use `save_steps=50` and `eval_steps=50`.
