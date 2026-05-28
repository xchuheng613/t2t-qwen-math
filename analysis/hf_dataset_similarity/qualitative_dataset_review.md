# Qualitative Hugging Face Dataset Review

This review avoids cosine similarity. It compares the sampled 20 public and 20 private local questions against example rows from likely Hugging Face math datasets by looking at format, topic coverage, difficulty, and wording.

## Local Dataset Signature

The sampled local public/private questions are not one clean benchmark style. They look like a mixed math assessment corpus with several recurring features:

- **Answer format:** frequent `[ANS]` placeholders, often multiple per problem.
- **Question types:** mix of free-response, multi-part numeric answer, and 10-option multiple-choice.
- **Topic mix:** arithmetic, percentages, fractions, functions, trigonometry, polar coordinates, calculus/integrals/limits, differential equations, statistics/hypothesis testing/confidence intervals, number theory, geometry, cryptography, and generated sequence/algorithm tasks.
- **Difficulty range:** elementary arithmetic through contest geometry and advanced engineering/math topics.
- **Wording artifacts:** some typos and normalization artifacts, for example `frac`, repeated instructions, and long option lists.

The private sample has more advanced MCQ-style technical math than GSM8K-like word problems: integrals, ODEs, steady-state recurrence/signal-style equations, cryptosystem questions, polar conics, and contest geometry.

## Dataset-by-Dataset Fit

## `AI-MO/NuminaMath-CoT`

Best overall simulator.

Why it fits:

- Contains a broad mixture of sources, including competition math, K-12 style math, synthetic math, and GSM8K-like word problems.
- Includes multi-part problems, proof/derivation problems, coordinate/polar geometry, probability, sequences, and algebra.
- Has both simple and advanced problems, which matches the wide difficulty spread in the local public/private sample.
- Some examples are multiple-choice or exam-like, though not consistently with 10 options.

Where it differs:

- It usually stores full solutions, not `[ANS]` placeholders.
- It is cleaner and more standardly formatted than the local data.
- It does not consistently mimic the local 10-option MCQ format.

Verdict: use this as the main external training/simulation source.

## `EleutherAI/hendrycks_math` / `qwedsacf/competition_math`

Best for the contest-math subset.

Why it fits:

- Strong match for problems like the private three-digit-integer sum, contest geometry, algebra identities, number theory, and challenging prealgebra/algebra.
- Clean problem statements with LaTeX and known difficulty/type labels.
- Good source for improving competition-style reasoning.

Where it differs:

- Mostly free-response competition problems, not classroom stats, ODEs, hypothesis testing, cryptography, or long 10-option MCQ.
- Less representative of the public/private dataset’s generated `[ANS]` placeholders and survey/statistics tasks.

Verdict: useful as a secondary source, especially for hard algebra/geometry/number theory. It should not be the only simulator.

## `allenai/math_qa`

Best format match for multiple-choice arithmetic/word problems.

Why it fits:

- Multiple-choice format with options and a correct letter resembles your MCQ rows.
- Has many short applied arithmetic/ratio/rate/probability-style questions.
- Contains noisy wording and imperfect rationales, which is closer to the local data’s occasional artifacts than the very clean competition datasets.

Where it differs:

- Usually 5 options, not 10.
- Mostly aptitude-style arithmetic word problems; it lacks much of your calculus, ODE, polar coordinate, advanced statistics, and contest geometry coverage.

Verdict: useful for MCQ format adaptation, but too narrow as the main dataset.

## `LLMcompe-Team-Watanabe/math_AoPS-Instruct_preprocess_fixed`

Good for proof-heavy olympiad style, but too narrow.

Why it fits:

- Contains high-level AoPS/olympiad-style number theory, geometry, and proof problems.
- Helps simulate the hardest contest/proof-like private questions.

Where it differs:

- Heavily proof/instruction oriented.
- Does not resemble the local `[ANS]` or options-heavy exam format.
- Too advanced and too proof-centric for the broader public/private mix.

Verdict: use only as a supplement for hard olympiad reasoning.

## `microsoft/orca-math-word-problems-200k`

Decent for simple applied word problems.

Why it fits:

- Matches local examples like roses, gas mileage, percent, shopping, ratios, and basic arithmetic.
- Large and easy to use.

Where it differs:

- Mostly natural-language grade-school style.
- Does not cover the advanced technical topics that appear often in the sampled private set.
- Lacks MCQ, `[ANS]`, symbolic calculus, ODEs, and statistics-test formatting.

Verdict: useful for easy word-problem coverage, but not a close overall simulator.

## `openai/gsm8k`

Too narrow for this project.

Why it fits:

- Some local public/private rows are grade-school arithmetic word problems.

Where it differs:

- GSM8K is almost entirely grade-school word problems.
- It misses nearly all advanced topics and MCQ structure.

Verdict: use only for basic arithmetic word-problem augmentation.

## `di-zhang-fdu/DeepMind_Mathematics_QA`

Weak fit.

Why it fits:

- It is math question-answer data.

Where it differs:

- Much narrower and more synthetic than the local mixed exam corpus.
- Does not reproduce the local MCQ/free-response mixture or topic breadth well.

Verdict: low priority.

## Recommended Mixture

For the closest qualitative simulation of the local public/private dataset:

1. **Primary:** `AI-MO/NuminaMath-CoT`
2. **Competition supplement:** `EleutherAI/hendrycks_math` or `qwedsacf/competition_math`
3. **MCQ format supplement:** `allenai/math_qa`
4. **Easy word-problem supplement:** `microsoft/orca-math-word-problems-200k` or `openai/gsm8k`

A practical blend would be:

- 50% `AI-MO/NuminaMath-CoT`
- 20% `EleutherAI/hendrycks_math` / `qwedsacf/competition_math`
- 20% `allenai/math_qa`
- 10% `microsoft/orca-math-word-problems-200k` or `openai/gsm8k`

This combination is more realistic than any single dataset because the local data itself appears assembled from multiple styles: classroom math, competition math, generated MCQ, statistics, calculus, and applied word problems.

