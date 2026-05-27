#!/usr/bin/env python3
"""Run the single-example reasoning workers through OpenAI Batch API.

Each JSONL request in the uploaded batch contains exactly one training example.
The script reuses the same worker prompt and validator as ``run_workers.py`` so
the downstream merge/build scripts do not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
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
from scripts.run_workers import build_worker_messages, load_valid_processed


TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("prepare", "submit", "status", "collect", "all"),
        help="prepare requests, submit a batch, check status, collect results, or run submit+poll+collect.",
    )
    parser.add_argument("--task-dir", type=Path, default=Path("single_tasks"))
    parser.add_argument(
        "--retry-queue",
        type=Path,
        default=None,
        help="Optional retry queue JSONL from validate_outputs.py or a previous batch collect.",
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
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument("--metadata-description", default="single-example LoRA reasoning workers")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--poll-timeout", type=float, default=None, help="Seconds to wait in all mode; default waits indefinitely.")
    parser.add_argument("--overwrite", action="store_true", help="Include tasks even if a valid processed file exists.")
    parser.add_argument("--no-json-mode", action="store_true", help="Do not request JSON object mode from the API.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow collect before the batch reaches a terminal status.")
    parser.add_argument("--finalize-failures", action="store_true", help="Write failed examples to output/failed_examples.jsonl now.")
    parser.add_argument("--batch-id", default=None, help="Batch ID for status/collect if not using the meta file.")
    parser.add_argument("--meta", type=Path, default=None, help="Batch metadata JSON path. Defaults to logs/batch_meta_attempt_N.json.")
    parser.add_argument("--requests-out", type=Path, default=None, help="Prepared batch request JSONL path.")
    parser.add_argument("--manifest-out", type=Path, default=None, help="Prepared custom_id-to-example manifest JSONL path.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def api_json_request(
    *,
    api_base: str,
    api_key: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_base.rstrip("/") + path,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    return json.loads(body)


def api_download_text(*, api_base: str, api_key: str, path: str) -> str:
    request = urllib.request.Request(
        api_base.rstrip("/") + path,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc


def api_upload_file(*, api_base: str, api_key: str, path: Path, purpose: str = "batch") -> dict[str, Any]:
    boundary = "----codex-batch-" + uuid.uuid4().hex
    file_bytes = path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
            purpose.encode(),
            b"\r\n",
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
            b"Content-Type: application/jsonl\r\n\r\n",
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        api_base.rstrip("/") + "/files",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc


def default_meta_path(args: argparse.Namespace) -> Path:
    return args.meta or args.logs_dir / f"batch_meta_attempt_{args.attempt}.json"


def default_requests_path(args: argparse.Namespace) -> Path:
    return args.requests_out or args.logs_dir / f"batch_requests_attempt_{args.attempt}.jsonl"


def default_manifest_path(args: argparse.Namespace) -> Path:
    return args.manifest_out or args.logs_dir / f"batch_manifest_attempt_{args.attempt}.jsonl"


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.retry_queue is not None:
        rows = load_jsonl(args.retry_queue, limit=args.limit)
        tasks = []
        seen: set[str] = set()
        for row in rows:
            row_id = str(row.get("id", ""))
            if row_id in seen:
                raise SystemExit(f"Duplicate retry-queue id: {row_id}")
            seen.add(row_id)
            tasks.append({"id": row_id, "instruction": row["instruction"], "output": row["output"]})
        return tasks

    paths = sorted(args.task_dir.glob("*.json"))
    if args.limit is not None:
        paths = paths[: args.limit]
    tasks = []
    seen: set[str] = set()
    for path in paths:
        task = read_json_object(path)
        row_id = str(task.get("id", ""))
        if row_id in seen:
            raise SystemExit(f"Duplicate task id: {row_id}")
        seen.add(row_id)
        tasks.append({"id": row_id, "instruction": task["instruction"], "output": task["output"]})
    return tasks


def make_custom_id(index: int, row_id: str) -> str:
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()[:10]
    return f"worker-{index:06d}-{digest}"


def prepare_batch_files(args: argparse.Namespace) -> tuple[Path, Path, list[dict[str, Any]]]:
    if not args.model:
        raise SystemExit("Missing model. Pass --model or set OPENAI_MODEL.")

    template = args.prompt_template.read_text(encoding="utf-8")
    tasks = load_tasks(args)
    request_rows = []
    manifest_rows = []
    included = 0
    skipped = 0
    for index, task in enumerate(tasks):
        if not args.overwrite and load_valid_processed(task, args.processed_dir) is not None:
            skipped += 1
            continue
        row_id = str(task["id"])
        custom_id = make_custom_id(index, row_id)
        body: dict[str, Any] = {
            "model": args.model,
            "messages": build_worker_messages(template, task),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if not args.no_json_mode:
            body["response_format"] = {"type": "json_object"}
        request_rows.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )
        manifest_rows.append(
            {
                "custom_id": custom_id,
                "id": row_id,
                "instruction": task["instruction"],
                "output": task["output"],
                "attempt": args.attempt,
            }
        )
        included += 1

    if not request_rows:
        raise SystemExit(f"No tasks to submit. skipped_existing_success={skipped}")

    requests_path = default_requests_path(args)
    manifest_path = default_manifest_path(args)
    write_jsonl(requests_path, request_rows)
    write_jsonl(manifest_path, manifest_rows)
    print(f"prepared={included} skipped_existing_success={skipped}")
    print(f"requests={requests_path}")
    print(f"manifest={manifest_path}")
    return requests_path, manifest_path, manifest_rows


def require_api_key(args: argparse.Namespace) -> str:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var {args.api_key_env}.")
    return api_key


def submit_batch(args: argparse.Namespace) -> dict[str, Any]:
    api_key = require_api_key(args)
    requests_path, manifest_path, manifest_rows = prepare_batch_files(args)
    uploaded = api_upload_file(api_base=args.api_base, api_key=api_key, path=requests_path, purpose="batch")
    batch = api_json_request(
        api_base=args.api_base,
        api_key=api_key,
        method="POST",
        path="/batches",
        payload={
            "input_file_id": uploaded["id"],
            "endpoint": "/v1/chat/completions",
            "completion_window": args.completion_window,
            "metadata": {
                "description": args.metadata_description,
                "attempt": str(args.attempt),
                "task_count": str(len(manifest_rows)),
            },
        },
    )
    meta = {
        "created_at_utc": utc_now(),
        "attempt": args.attempt,
        "model": args.model,
        "task_count": len(manifest_rows),
        "request_file": str(requests_path),
        "manifest_file": str(manifest_path),
        "input_file_id": uploaded["id"],
        "batch_id": batch["id"],
        "batch": batch,
    }
    meta_path = default_meta_path(args)
    write_json_object(meta_path, meta)
    print(f"uploaded_file={uploaded['id']}")
    print(f"batch_id={batch['id']} status={batch.get('status')}")
    print(f"meta={meta_path}")
    return meta


def load_meta(args: argparse.Namespace) -> dict[str, Any]:
    meta_path = default_meta_path(args)
    if meta_path.exists():
        return read_json_object(meta_path)
    if args.batch_id:
        return {"batch_id": args.batch_id, "attempt": args.attempt}
    raise SystemExit(f"Missing batch metadata: {meta_path}. Pass --batch-id or run submit first.")


def retrieve_batch(args: argparse.Namespace) -> dict[str, Any]:
    api_key = require_api_key(args)
    meta = load_meta(args)
    batch_id = args.batch_id or meta["batch_id"]
    batch = api_json_request(api_base=args.api_base, api_key=api_key, method="GET", path=f"/batches/{batch_id}")
    meta["batch_id"] = batch_id
    meta["batch"] = batch
    meta["last_retrieved_at_utc"] = utc_now()
    write_json_object(default_meta_path(args), meta)
    counts = batch.get("request_counts") or {}
    print(
        "batch_id="
        f"{batch_id} status={batch.get('status')} "
        f"completed={counts.get('completed')} failed={counts.get('failed')} total={counts.get('total')}"
    )
    return meta


def poll_until_terminal(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        meta = retrieve_batch(args)
        status = str(meta.get("batch", {}).get("status", ""))
        if status in TERMINAL_STATUSES:
            return meta
        if args.poll_timeout is not None and time.monotonic() - started >= args.poll_timeout:
            raise SystemExit(f"Timed out waiting for batch; last status={status}")
        time.sleep(args.poll_interval)


def download_batch_files(args: argparse.Namespace, meta: dict[str, Any]) -> tuple[Path | None, Path | None]:
    api_key = require_api_key(args)
    batch = meta["batch"]
    output_path = None
    error_path = None
    if batch.get("output_file_id"):
        output_path = args.logs_dir / f"batch_output_attempt_{args.attempt}.jsonl"
        text = api_download_text(
            api_base=args.api_base,
            api_key=api_key,
            path=f"/files/{batch['output_file_id']}/content",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    if batch.get("error_file_id"):
        error_path = args.logs_dir / f"batch_error_attempt_{args.attempt}.jsonl"
        text = api_download_text(
            api_base=args.api_base,
            api_key=api_key,
            path=f"/files/{batch['error_file_id']}/content",
        )
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(text, encoding="utf-8")
    return output_path, error_path


def log_failed_attempt(args: argparse.Namespace, task: dict[str, Any], error: str, raw_response: str = "") -> None:
    path = args.logs_dir / "failed_attempts" / f"{safe_task_name(str(task['id']))}__batch_attempt_{args.attempt}.json"
    write_json_object(
        path,
        {
            "id": task["id"],
            "attempt": args.attempt,
            "time_utc": utc_now(),
            "error": error,
            "raw_response": raw_response,
            "task": {
                "id": task["id"],
                "instruction": task["instruction"],
                "output": task["output"],
            },
        },
    )


def extract_chat_content(line: dict[str, Any]) -> str:
    response = line.get("response")
    if not isinstance(response, dict):
        raise ValueError(f"missing response object: {line.get('error')}")
    status_code = response.get("status_code")
    if status_code != 200:
        raise ValueError(f"non-200 response status: {status_code}")
    body = response.get("body")
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected chat completion body: {body}") from exc


def collect_batch(args: argparse.Namespace) -> list[dict[str, Any]]:
    meta = retrieve_batch(args)
    batch = meta["batch"]
    status = str(batch.get("status", ""))
    if status not in TERMINAL_STATUSES and not args.allow_incomplete:
        raise SystemExit(f"Batch is not terminal yet: status={status}")

    output_path, error_path = download_batch_files(args, meta)
    manifest_file = Path(meta.get("manifest_file", default_manifest_path(args)))
    if not manifest_file.exists():
        raise SystemExit(f"Missing manifest file: {manifest_file}")
    manifest_rows = load_jsonl(manifest_file)
    task_by_custom_id = {row["custom_id"]: row for row in manifest_rows}
    seen_custom_ids: set[str] = set()
    success_count = 0
    failures: list[dict[str, Any]] = []

    if output_path and output_path.exists():
        for line in load_jsonl(output_path):
            custom_id = str(line.get("custom_id", ""))
            seen_custom_ids.add(custom_id)
            task = task_by_custom_id.get(custom_id)
            if task is None:
                failures.append({"id": "", "instruction": "", "output": "", "error": f"unknown custom_id {custom_id}"})
                continue
            try:
                raw_response = extract_chat_content(line)
                record = strict_json_from_text(raw_response)
                errors = validate_worker_record(record, task)
                if errors:
                    raise ValueError("; ".join(errors))
                write_json_object(task_path_for_id(args.processed_dir, str(task["id"])), record)
                success_count += 1
            except Exception as exc:
                error = str(exc)
                log_failed_attempt(args, task, error, raw_response=json.dumps(line, ensure_ascii=False))
                failures.append(
                    {
                        "id": task["id"],
                        "instruction": task["instruction"],
                        "output": task["output"],
                        "status": "failed",
                        "attempts": args.attempt,
                        "error": error,
                    }
                )

    if error_path and error_path.exists():
        for line in load_jsonl(error_path):
            custom_id = str(line.get("custom_id", ""))
            seen_custom_ids.add(custom_id)
            task = task_by_custom_id.get(custom_id)
            if task is None:
                continue
            error = json.dumps(line.get("error", line), ensure_ascii=False)
            log_failed_attempt(args, task, error, raw_response=json.dumps(line, ensure_ascii=False))
            failures.append(
                {
                    "id": task["id"],
                    "instruction": task["instruction"],
                    "output": task["output"],
                    "status": "failed",
                    "attempts": args.attempt,
                    "error": error,
                }
            )

    for custom_id, task in task_by_custom_id.items():
        if custom_id not in seen_custom_ids:
            error = "missing from batch output and error files"
            log_failed_attempt(args, task, error)
            failures.append(
                {
                    "id": task["id"],
                    "instruction": task["instruction"],
                    "output": task["output"],
                    "status": "failed",
                    "attempts": args.attempt,
                    "error": error,
                }
            )

    next_retry_path = args.logs_dir / f"retry_queue_batch_attempt_{args.attempt + 1}.jsonl"
    max_attempts = args.max_retries + 1
    if failures and args.attempt < max_attempts and not args.finalize_failures:
        write_jsonl(
            next_retry_path,
            [
                {"id": row["id"], "instruction": row["instruction"], "output": row["output"]}
                for row in failures
                if row.get("id")
            ],
        )
        print(f"retry_queue={next_retry_path}")
    elif failures:
        write_jsonl(args.failed_out, failures)
        print(f"failed_out={args.failed_out}")
    elif args.failed_out.exists() and args.finalize_failures:
        args.failed_out.unlink()

    print(
        f"collected success={success_count} failed={len(failures)} "
        f"output_file={output_path} error_file={error_path}"
    )
    return failures


def run_all(args: argparse.Namespace) -> None:
    attempt = args.attempt
    retry_queue = args.retry_queue
    max_attempts = args.max_retries + 1
    backoffs = [2, 5, 10]

    while attempt <= max_attempts:
        args.attempt = attempt
        args.retry_queue = retry_queue
        args.meta = args.logs_dir / f"batch_meta_attempt_{attempt}.json"
        submit_batch(args)
        poll_until_terminal(args)
        args.finalize_failures = attempt >= max_attempts
        failures = collect_batch(args)
        if not failures:
            return
        retry_queue = args.logs_dir / f"retry_queue_batch_attempt_{attempt + 1}.jsonl"
        delay = backoffs[min(attempt - 1, len(backoffs) - 1)]
        if attempt < max_attempts:
            print(f"retry_attempt={attempt + 1} pending={len(failures)} sleeping={delay}s")
            time.sleep(delay)
        attempt += 1


def main() -> None:
    args = parse_args()
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    (args.logs_dir / "failed_attempts").mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.failed_out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "prepare":
        prepare_batch_files(args)
    elif args.mode == "submit":
        submit_batch(args)
    elif args.mode == "status":
        retrieve_batch(args)
    elif args.mode == "collect":
        collect_batch(args)
    elif args.mode == "all":
        run_all(args)


if __name__ == "__main__":
    main()
