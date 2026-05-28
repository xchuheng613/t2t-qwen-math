# Hugging Face Dataset Similarity Report

Seed: `151`. Local sample: 20 public + 20 private questions.
Max HF rows per dataset: `50000`.

| Rank | Dataset | Mean best | Public mean | Private mean | >=0.50 | Rows |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `AI-MO/NuminaMath-CoT` | 0.280 | 0.239 | 0.322 | 3 | 50000 |
| 2 | `qwedsacf/competition_math` | 0.231 | 0.201 | 0.261 | 1 | 12487 |
| 3 | `EleutherAI/hendrycks_math` | 0.231 | 0.201 | 0.261 | 1 | 12487 |
| 4 | `LLMcompe-Team-Watanabe/math_AoPS-Instruct_preprocess_fixed` | 0.224 | 0.206 | 0.242 | 1 | 50000 |
| 5 | `allenai/math_qa` | 0.186 | 0.191 | 0.182 | 1 | 37036 |
| 6 | `microsoft/orca-math-word-problems-200k` | 0.161 | 0.168 | 0.153 | 1 | 50000 |
| 7 | `openai/gsm8k` | 0.132 | 0.150 | 0.114 | 0 | 17584 |
| 8 | `di-zhang-fdu/DeepMind_Mathematics_QA` | 0.092 | 0.094 | 0.089 | 0 | 980 |

Skipped datasets:
- `hendrycks/competition_math`: no question text found
