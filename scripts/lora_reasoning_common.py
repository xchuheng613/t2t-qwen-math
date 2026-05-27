#!/usr/bin/env python3
"""Shared helpers for the single-example reasoning data pipeline."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


EXPECTED_WORKER_KEYS = {
    "id",
    "instruction",
    "reasoning",
    "output",
    "warning",
    "source",
}

OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path, limit=limit)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("data", "rows", "examples"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list or an object containing a list")
        rows = [row for row in data if isinstance(row, dict)]
        if len(rows) != len(data):
            raise ValueError(f"{path}: every JSON item must be an object")
        return rows[:limit] if limit is not None else rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        return rows[:limit] if limit is not None else rows
    raise ValueError(f"Unsupported input format for {path}; use .jsonl, .json, or .csv")


def answer_values(answer: Any) -> list[str]:
    values = answer if isinstance(answer, list) else [answer]
    return ["" if value is None else str(value).strip() for value in values]


def answer_to_output(answer: Any) -> str:
    """Convert raw gold answers to the single-box string used by this SFT data."""
    return ", ".join(answer_values(answer)).strip()


def normalize_id(row: dict[str, Any], index: int) -> str:
    for key in ("id", "uid", "example_id", "question_id"):
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return f"example_{index:06d}"


def _format_options(options: Any) -> str:
    if isinstance(options, dict):
        lines = []
        for key in sorted(options):
            lines.append(f"{key}. {str(options[key]).strip()}")
        return "\n".join(lines)
    if isinstance(options, list):
        lines = []
        for idx, option in enumerate(options):
            letter = OPTION_LETTERS[idx] if idx < len(OPTION_LETTERS) else str(idx + 1)
            lines.append(f"{letter}. {str(option).strip()}")
        return "\n".join(lines)
    if options is None or str(options).strip() == "":
        return ""
    return str(options).strip()


def build_instruction(row: dict[str, Any]) -> str:
    for key in ("instruction", "question", "prompt", "input"):
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            question = str(value).strip()
            break
    else:
        raise ValueError("row is missing instruction/question/prompt/input")

    options = _format_options(row.get("options"))
    if options:
        return f"{question}\n\nOptions:\n{options}"
    return question


def normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    answer = row.get("output", row.get("answer"))
    if answer is None:
        raise ValueError(f"row {index} is missing answer/output")
    return {
        "id": normalize_id(row, index),
        "instruction": build_instruction(row),
        "output": answer_to_output(answer),
    }


def safe_task_name(row_id: str) -> str:
    safe = SAFE_ID_RE.sub("_", str(row_id).strip())
    return safe or "blank_id"


def task_path_for_id(task_dir: Path, row_id: str) -> Path:
    return task_dir / f"{safe_task_name(row_id)}.json"


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json_object(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def strict_json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```") or stripped.endswith("```"):
        raise ValueError("markdown code fence wrapper")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("worker response is not a JSON object")
    return value


def validate_worker_record(record: dict[str, Any], source: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(EXPECTED_WORKER_KEYS - set(record))
    extra = sorted(set(record) - EXPECTED_WORKER_KEYS)
    if missing:
        errors.append(f"missing keys: {', '.join(missing)}")
    if extra:
        errors.append(f"unexpected keys: {', '.join(extra)}")

    if str(record.get("id", "")) != str(source.get("id", "")):
        errors.append("id changed")
    if record.get("instruction") != source.get("instruction"):
        errors.append("instruction changed")
    if record.get("output") != source.get("output"):
        errors.append("output changed")

    reasoning = record.get("reasoning")
    if not isinstance(reasoning, str) or reasoning.strip() == "":
        errors.append("reasoning missing or empty")
    elif len(reasoning.split()) > 320:
        errors.append("reasoning exceeds 320 words")
    if isinstance(reasoning, str):
        lowered = reasoning.lower()
        if "final_answers" in lowered:
            errors.append("reasoning contains FINAL_ANSWERS")
        if "\\boxed" in reasoning:
            errors.append("reasoning contains boxed final answer")
        if "<think" in lowered or "</think>" in lowered:
            errors.append("reasoning contains think tags")

    warning = record.get("warning")
    if not isinstance(warning, str):
        errors.append("warning is missing or not a string")
    if record.get("source") != "worker_single":
        errors.append("source must be worker_single")
    return errors


def load_input_by_id(normalized_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    rows = load_jsonl(normalized_path)
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id in by_id:
            duplicate_ids.append(row_id)
        else:
            by_id[row_id] = row
    return rows, by_id, duplicate_ids


def validation_report_markdown(
    *,
    input_count: int,
    success_count: int,
    failure_count: int,
    missing_count: int,
    invalid_count: int,
    duplicate_ids: list[str],
    invalid_details: list[str],
) -> str:
    lines = [
        "# Validation Report",
        "",
        f"- Input examples: {input_count}",
        f"- Successful outputs: {success_count}",
        f"- Permanent failures: {failure_count}",
        f"- Missing examples: {missing_count}",
        f"- Invalid outputs: {invalid_count}",
        f"- Duplicate input IDs: {len(duplicate_ids)}",
        "",
    ]
    if duplicate_ids:
        lines.extend(["## Duplicate IDs", ""])
        lines.extend(f"- {row_id}" for row_id in duplicate_ids[:50])
        if len(duplicate_ids) > 50:
            lines.append(f"- ... {len(duplicate_ids) - 50} more")
        lines.append("")
    if invalid_details:
        lines.extend(["## Invalid Output Details", ""])
        lines.extend(f"- {detail}" for detail in invalid_details[:100])
        if len(invalid_details) > 100:
            lines.append(f"- ... {len(invalid_details) - 100} more")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
