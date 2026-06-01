#!/usr/bin/env python3
"""Convert reasoning JSONL rows into chat-style SFT messages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lora_reasoning_common import load_jsonl, write_jsonl


DEFAULT_SYSTEM_PROMPT = """Solve the problem. Keep reasoning short.

Final answer rules:
- Output exactly one final \\boxed{} after FINAL_ANSWERS:.
- If there are multiple [ANS] blanks, put answers in the same box separated by commas.
- Answer every [ANS] blank in order; do not add extra answers.
- No units, labels, [ANS], explanations, or words inside the box.
- Only round if the problem explicitly says to round."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/lora_training_data.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/sft_messages.jsonl"))
    parser.add_argument("--system-prompt", type=Path, default=None)
    parser.add_argument("--skip-warnings", action="store_true")
    return parser.parse_args()


def assistant_message(reasoning: str, output: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\nFINAL_ANSWERS:\n\\boxed{{{output}}}"


def main() -> None:
    args = parse_args()
    system_prompt = (
        args.system_prompt.read_text(encoding="utf-8").strip()
        if args.system_prompt
        else DEFAULT_SYSTEM_PROMPT
    )
    rows = load_jsonl(args.input)
    messages = []
    skipped = 0
    for row in rows:
        warning = str(row.get("warning", ""))
        if args.skip_warnings and warning:
            skipped += 1
            continue
        reasoning = str(row.get("reasoning", "")).strip()
        output = str(row.get("output", ""))
        if not reasoning:
            raise SystemExit(f"Missing reasoning for id={row.get('id')}")
        messages.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["instruction"]},
                    {"role": "assistant", "content": assistant_message(reasoning, output)},
                ],
            }
        )

    count = write_jsonl(args.output, messages)
    print(f"messages={count} output={args.output} skipped_warnings={skipped}")


if __name__ == "__main__":
    main()
