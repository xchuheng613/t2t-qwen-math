#!/usr/bin/env python3
"""Split normalized JSONL into one JSON file per training example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lora_reasoning_common import load_jsonl, task_path_for_id, write_json_object, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/normalized/public_normalized.jsonl"))
    parser.add_argument("--task-dir", type=Path, default=Path("single_tasks"))
    parser.add_argument("--manifest", type=Path, default=Path("logs/single_tasks_manifest.jsonl"))
    parser.add_argument("--clean", action="store_true", help="Remove old .json task files in task-dir first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    args.task_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for path in args.task_dir.glob("*.json"):
            path.unlink()

    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    manifest_rows = []
    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id in seen_ids:
            raise SystemExit(f"Duplicate input id: {row_id}")
        seen_ids.add(row_id)

        path = task_path_for_id(args.task_dir, row_id)
        if path in seen_paths:
            raise SystemExit(f"Task filename collision for id={row_id}: {path}")
        seen_paths.add(path)

        task = {
            "id": row_id,
            "instruction": row["instruction"],
            "output": row["output"],
        }
        write_json_object(path, task)
        manifest_rows.append({"id": row_id, "task_path": str(path)})

    write_jsonl(args.manifest, manifest_rows)
    print(f"tasks={len(manifest_rows)} task_dir={args.task_dir}")
    print(f"manifest={args.manifest}")


if __name__ == "__main__":
    main()
