You are processing ONE training example.

Input:
- id
- instruction
- output

Task:
Generate concise visible reasoning explaining how to get the provided output.

Constraints:
- Process exactly this one example.
- 20-120 words for most examples.
- Up to 300 words only for hard/statistics/multi-blank examples.
- Use short, direct calculation steps.
- Account for every [ANS] blank in order.
- Do not add extra answers.
- Preserve the exact output string provided.
- Do not modify instruction.
- Do not modify output.
- Do not include FINAL_ANSWERS.
- Do not include \boxed{}.
- Do not include <think> tags.
- Do not include hidden chain-of-thought or exploratory uncertainty.
- If the output seems inconsistent with the problem, add a brief warning.
- Otherwise warning should be an empty string.

Output JSON ONLY, with exactly these keys:

{
  "id": "...",
  "instruction": "...",
  "reasoning": "...",
  "output": "...",
  "warning": "",
  "source": "worker_single"
}
