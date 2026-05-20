#!/usr/bin/env python3
"""Build LoRA v2 splits from audited process-supervision SFT messages.

The process SFT file is treated as the training set. Dev/holdout benchmarks are
copied from the existing v1 benchmark split so v2 can be compared directly
against previous base and LoRA runs without data leakage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompts.compact_prompt_pack import build_routed_prompt, rebuild_final_response


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_values(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer")
    values = answer if isinstance(answer, list) else [answer]
    return ["" if value is None else str(value).strip() for value in values]


def make_final_only_sft(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row["question"])
    options = row.get("options") or None
    system, user = build_routed_prompt("compact", question, options)
    return {
        "id": int(row["id"]),
        "format": "mcq" if options else "free",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": rebuild_final_response(answer_values(row), question)},
        ],
    }


def validate_process_rows(rows: list[dict[str, Any]]) -> None:
    ids: set[int] = set()
    for row in rows:
        row_id = int(row["id"])
        if row_id in ids:
            raise ValueError(f"Duplicate process SFT id: {row_id}")
        ids.add(row_id)

        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"Bad messages field for id={row_id}")
        roles = [message.get("role") for message in messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"Bad message roles for id={row_id}: {roles}")
        assistant = str(messages[-1].get("content", ""))
        if "FINAL_ANSWERS:" not in assistant or "\\boxed{" not in assistant:
            raise ValueError(f"Missing final answer block for id={row_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-sft",
        type=Path,
        default=Path("data/lora_public_v2/sft_messages_audited_v2.jsonl"),
    )
    parser.add_argument("--public-path", type=Path, default=Path("data/public.jsonl"))
    parser.add_argument("--base-split-dir", type=Path, default=Path("data/lora_public_v1"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/lora_public_v2"))
    parser.add_argument("--sft-out-dir", type=Path, default=Path("data/sft_lora_public_v2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_rows = load_jsonl(args.process_sft)
    validate_process_rows(process_rows)
    process_ids = {int(row["id"]) for row in process_rows}

    public_by_id = {int(row["id"]): row for row in load_jsonl(args.public_path)}
    missing = sorted(process_ids - set(public_by_id))
    if missing:
        raise ValueError(f"Process SFT ids missing from public data: {missing[:20]}")

    raw_train = [public_by_id[int(row["id"])] for row in process_rows]
    dev = load_jsonl(args.base_split_dir / "public_dev.jsonl")
    holdout = load_jsonl(args.base_split_dir / "public_holdout.jsonl")
    dev_free = load_jsonl(args.base_split_dir / "dev_free.jsonl")
    holdout_free = load_jsonl(args.base_split_dir / "holdout_free.jsonl")

    dev_ids = {int(row["id"]) for row in dev}
    holdout_ids = {int(row["id"]) for row in holdout}
    if process_ids & dev_ids:
        raise ValueError(f"Process training rows overlap dev ids: {sorted(process_ids & dev_ids)[:20]}")
    if process_ids & holdout_ids:
        raise ValueError(f"Process training rows overlap holdout ids: {sorted(process_ids & holdout_ids)[:20]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.sft_out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(args.out_dir / "train_free_keep.jsonl", raw_train)
    shutil.copyfile(args.base_split_dir / "public_train_pool.jsonl", args.out_dir / "public_train_pool.jsonl")
    shutil.copyfile(args.base_split_dir / "public_dev.jsonl", args.out_dir / "public_dev.jsonl")
    shutil.copyfile(args.base_split_dir / "public_holdout.jsonl", args.out_dir / "public_holdout.jsonl")
    shutil.copyfile(args.base_split_dir / "dev_free.jsonl", args.out_dir / "dev_free.jsonl")
    shutil.copyfile(args.base_split_dir / "holdout_free.jsonl", args.out_dir / "holdout_free.jsonl")

    write_jsonl(args.sft_out_dir / "train.jsonl", process_rows)
    write_jsonl(args.sft_out_dir / "dev.jsonl", [make_final_only_sft(row) for row in dev_free])
    write_jsonl(args.sft_out_dir / "holdout.jsonl", [make_final_only_sft(row) for row in holdout_free])

    manifest = {
        "process_sft": str(args.process_sft),
        "public_path": str(args.public_path),
        "base_split_dir": str(args.base_split_dir),
        "out_dir": str(args.out_dir),
        "sft_out_dir": str(args.sft_out_dir),
        "train_process_rows": len(process_rows),
        "dev_rows": len(dev),
        "holdout_rows": len(holdout),
        "dev_free_rows": len(dev_free),
        "holdout_free_rows": len(holdout_free),
    }
    (args.out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"train_process={len(process_rows)} -> {args.sft_out_dir / 'train.jsonl'}")
    print(f"raw_train={len(raw_train)} -> {args.out_dir / 'train_free_keep.jsonl'}")
    print(f"public_dev={len(dev)} dev_free={len(dev_free)}")
    print(f"public_holdout={len(holdout)} holdout_free={len(holdout_free)}")
    print(f"manifest={args.out_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
