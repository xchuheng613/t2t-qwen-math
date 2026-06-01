#!/usr/bin/env python3
"""Run one API-backed worker per single-example task with retries."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lora_reasoning_common import (
    load_jsonl,
    read_json_object,
    safe_task_name,
    strict_json_from_text,
    task_path_for_id,
    validate_worker_record,
    write_json_object,
    write_jsonl,
)


@dataclass(frozen=True)
class WorkerOutcome:
    row_id: str
    attempt: int
    ok: bool
    record: dict[str, Any] | None = None
    raw_response: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=Path("single_tasks"))
    parser.add_argument(
        "--retry-queue",
        type=Path,
        default=None,
        help="Optional JSONL queue from validate_outputs.py; each row must be one normalized example.",
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_tasks"))
    parser.add_argument("--prompt-template", type=Path, default=Path("prompts/worker_prompt.md"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--failed-out", type=Path, default=Path("output/failed_examples.jsonl"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Re-run tasks even if a valid processed file exists.")
    parser.add_argument("--no-json-mode", action="store_true", help="Do not request JSON object mode from the API.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first worker request and exit.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_tasks(task_dir: Path, limit: int | None, retry_queue: Path | None = None) -> list[dict[str, Any]]:
    if retry_queue is not None:
        rows = load_jsonl(retry_queue, limit=limit)
        tasks = []
        seen: set[str] = set()
        for row in rows:
            row_id = str(row.get("id", ""))
            if row_id in seen:
                raise SystemExit(f"Duplicate retry-queue id: {row_id}")
            seen.add(row_id)
            tasks.append(
                {
                    "id": row_id,
                    "instruction": row["instruction"],
                    "output": row["output"],
                    "_task_path": str(retry_queue),
                }
            )
        return tasks

    paths = sorted(task_dir.glob("*.json"))
    if limit is not None:
        paths = paths[:limit]
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        task = read_json_object(path)
        row_id = str(task.get("id", ""))
        if row_id in seen:
            raise SystemExit(f"Duplicate task id: {row_id}")
        seen.add(row_id)
        task["_task_path"] = str(path)
        tasks.append(task)
    return tasks


def build_worker_messages(template: str, task: dict[str, Any]) -> list[dict[str, str]]:
    worker_input = {
        "id": task["id"],
        "instruction": task["instruction"],
        "output": task["output"],
    }
    return [
        {"role": "system", "content": template.strip()},
        {
            "role": "user",
            "content": "Input JSON:\n" + json.dumps(worker_input, ensure_ascii=False),
        },
    ]


def call_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    json_mode: bool,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    url = api_base.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc

    data = json.loads(body)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected API response shape: {body[:1000]}") from exc


def run_one_worker(
    *,
    task: dict[str, Any],
    template: str,
    args: argparse.Namespace,
    api_key: str,
    attempt: int,
) -> WorkerOutcome:
    row_id = str(task["id"])
    try:
        messages = build_worker_messages(template, task)
        raw_response = call_chat_completion(
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            json_mode=not args.no_json_mode,
        )
        record = strict_json_from_text(raw_response)
        source = {"id": task["id"], "instruction": task["instruction"], "output": task["output"]}
        errors = validate_worker_record(record, source)
        if errors:
            return WorkerOutcome(row_id, attempt, False, raw_response=raw_response, error="; ".join(errors))
        return WorkerOutcome(row_id, attempt, True, record=record, raw_response=raw_response)
    except Exception as exc:  # Worker failures are logged and retried by the supervisor.
        return WorkerOutcome(row_id, attempt, False, error=str(exc))


def load_valid_processed(task: dict[str, Any], processed_dir: Path) -> dict[str, Any] | None:
    row_id = str(task["id"])
    path = task_path_for_id(processed_dir, row_id)
    if not path.exists():
        return None
    try:
        record = read_json_object(path)
    except ValueError:
        return None
    source = {"id": task["id"], "instruction": task["instruction"], "output": task["output"]}
    return record if not validate_worker_record(record, source) else None


def log_failed_attempt(logs_dir: Path, outcome: WorkerOutcome, task: dict[str, Any]) -> None:
    path = logs_dir / "failed_attempts" / f"{safe_task_name(outcome.row_id)}__attempt_{outcome.attempt}.json"
    payload = {
        "id": outcome.row_id,
        "attempt": outcome.attempt,
        "time_utc": utc_now(),
        "error": outcome.error,
        "raw_response": outcome.raw_response,
        "task": {
            "id": task["id"],
            "instruction": task["instruction"],
            "output": task["output"],
        },
    }
    write_json_object(path, payload)


def append_status(logs_dir: Path, event: dict[str, Any]) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "worker_status.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_latest_status(logs_dir: Path, status: dict[str, dict[str, Any]]) -> None:
    write_json_object(logs_dir / "worker_status_latest.json", status)


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.task_dir, args.limit, retry_queue=args.retry_queue)
    if not tasks:
        raise SystemExit(f"No task files found in {args.task_dir}")

    template = args.prompt_template.read_text(encoding="utf-8")
    first_messages = build_worker_messages(template, tasks[0])
    if args.dry_run:
        print(json.dumps({"model": args.model, "messages": first_messages}, ensure_ascii=False, indent=2))
        print(f"tasks={len(tasks)} dry_run=true")
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var {args.api_key_env}. Set it before running workers.")
    if not args.model:
        raise SystemExit("Missing model. Pass --model or set OPENAI_MODEL.")

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    (args.logs_dir / "failed_attempts").mkdir(parents=True, exist_ok=True)
    args.failed_out.parent.mkdir(parents=True, exist_ok=True)

    task_by_id = {str(task["id"]): task for task in tasks}
    status: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for task in tasks:
        row_id = str(task["id"])
        status[row_id] = {"status": "pending", "attempts": 0}
        if not args.overwrite and load_valid_processed(task, args.processed_dir) is not None:
            status[row_id] = {"status": "success", "attempts": 0, "resumed": True}
        else:
            pending.append(task)

    max_attempts = args.max_retries + 1
    backoffs = [2, 5, 10]
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        if attempt > 1:
            delay = backoffs[min(attempt - 2, len(backoffs) - 1)]
            print(f"retry_attempt={attempt} pending={len(pending)} sleeping={delay}s")
            time.sleep(delay)

        for task in pending:
            row_id = str(task["id"])
            status[row_id] = {"status": "running", "attempts": attempt}
            append_status(args.logs_dir, {"time_utc": utc_now(), "id": row_id, "status": "running", "attempt": attempt})

        next_pending: list[dict[str, Any]] = []
        concurrency = max(1, args.max_workers)
        for start in range(0, len(pending), concurrency):
            batch = pending[start : start + concurrency]
            outcomes: list[tuple[dict[str, Any], WorkerOutcome]] = []
            outcomes_lock = threading.Lock()

            def thread_target(batch_task: dict[str, Any]) -> None:
                outcome = run_one_worker(
                    task=batch_task,
                    template=template,
                    args=args,
                    api_key=api_key,
                    attempt=attempt,
                )
                with outcomes_lock:
                    outcomes.append((batch_task, outcome))

            threads = [
                threading.Thread(
                    target=thread_target,
                    args=(task,),
                    name=f"worker_single_{safe_task_name(str(task['id']))}",
                )
                for task in batch
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for task, outcome in outcomes:
                row_id = outcome.row_id
                if outcome.ok and outcome.record is not None:
                    write_json_object(task_path_for_id(args.processed_dir, row_id), outcome.record)
                    status[row_id] = {"status": "success", "attempts": attempt}
                    append_status(args.logs_dir, {"time_utc": utc_now(), "id": row_id, "status": "success", "attempt": attempt})
                else:
                    log_failed_attempt(args.logs_dir, outcome, task)
                    retry_left = attempt < max_attempts
                    status[row_id] = {
                        "status": "retrying" if retry_left else "failed",
                        "attempts": attempt,
                        "error": outcome.error,
                    }
                    append_status(
                        args.logs_dir,
                        {
                            "time_utc": utc_now(),
                            "id": row_id,
                            "status": "retrying" if retry_left else "failed",
                            "attempt": attempt,
                            "error": outcome.error,
                        },
                    )
                    if retry_left:
                        next_pending.append(task)
        pending = next_pending
        write_latest_status(args.logs_dir, status)
        successes = sum(1 for item in status.values() if item["status"] == "success")
        failures = sum(1 for item in status.values() if item["status"] == "failed")
        print(f"attempt={attempt} success={successes} retrying={len(pending)} failed={failures}")

    failed_rows = []
    for row_id, item in status.items():
        if item["status"] == "failed":
            task = task_by_id[row_id]
            failed_rows.append(
                {
                    "id": row_id,
                    "instruction": task["instruction"],
                    "output": task["output"],
                    "status": "failed",
                    "attempts": item.get("attempts", max_attempts),
                    "error": item.get("error", ""),
                }
            )

    if failed_rows:
        write_jsonl(args.failed_out, failed_rows)
    elif args.failed_out.exists():
        args.failed_out.unlink()

    write_latest_status(args.logs_dir, status)
    success_count = sum(1 for item in status.values() if item["status"] == "success")
    print(f"done success={success_count} failed={len(failed_rows)} total={len(tasks)}")


if __name__ == "__main__":
    main()
