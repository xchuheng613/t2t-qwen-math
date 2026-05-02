#!/usr/bin/env python3
"""Score a public-set run and optionally refresh wrong-answer visualizations.

Examples:

    .venv/bin/python verify_public.py results/32gb_balanced_public_smoke
    .venv/bin/python verify_public.py results/32gb_balanced_public/submission.jsonl
    .venv/bin/python verify_public.py results/32gb_balanced_public --no-visualize

The script writes a visualization-compatible JSONL to ``result_analyze/`` and
appends one row to ``result_analyze/public_verification_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATA_PATH = Path("data/public.jsonl")
DEFAULT_ANALYSIS_DIR = Path("result_analyze")
DEFAULT_SUMMARY_PATH = DEFAULT_ANALYSIS_DIR / "public_verification_summary.csv"

_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")
_LETTER_PHRASE_RE = re.compile(
    r"(?:option|choice|answer\s+is)\s*[:\s]*\(?([A-Z])\)?\b",
    re.IGNORECASE,
)
_OPTION_MARKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z])[\.\)]\s+")


@dataclass
class ScoreSummary:
    run_name: str
    total: int = 0
    correct: int = 0
    mcq_total: int = 0
    mcq_correct: int = 0
    free_total: int = 0
    free_correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def mcq_accuracy(self) -> float:
        return self.mcq_correct / self.mcq_total if self.mcq_total else 0.0

    @property
    def free_accuracy(self) -> float:
        return self.free_correct / self.free_total if self.free_total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prediction",
        type=Path,
        help="Result directory, submission.jsonl, or submission.csv to score.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--name",
        default=None,
        help="Report name. Defaults to result directory name.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Only score and write JSONL; do not regenerate HTML reports.",
    )
    parser.add_argument(
        "--show-wrong",
        type=int,
        default=50,
        help="How many wrong ids to print. Use 0 to suppress.",
    )
    return parser.parse_args()


def resolve_prediction_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Prediction path does not exist: {path}")

    preferred = [
        path / "submission.jsonl",
        path / "submission.csv",
        path / "submission_format_router.jsonl",
        path / "submission_format_router.csv",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    jsonl_files = sorted(path.glob("*.jsonl"))
    if len(jsonl_files) == 1:
        return jsonl_files[0]
    csv_files = sorted(path.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    choices = [p.name for p in jsonl_files + csv_files]
    raise FileNotFoundError(
        f"Could not infer prediction file in {path}. Candidates: {choices}"
    )


def infer_run_name(prediction: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    if prediction.stem in {"submission", "submission_format_router"}:
        return prediction.parent.name
    return prediction.stem


def load_public(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["id"])] = row
    return rows


def load_predictions(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as file:
            return [dict(row) for row in csv.DictReader(file)]

    raise ValueError(f"Unsupported prediction format: {path.suffix}")


def answer_values(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer")
    values = answer if isinstance(answer, list) else [answer]
    return [str(value).strip() for value in values]


def single_letter_gold(row: dict[str, Any]) -> str:
    values = answer_values(row)
    if len(values) == 1 and re.fullmatch(r"[A-Z]", values[0]):
        return values[0]
    return ""


def all_letter_gold(row: dict[str, Any]) -> bool:
    values = answer_values(row)
    return bool(values) and all(re.fullmatch(r"[A-Z]+", value) for value in values)


def has_inline_options(question: str) -> bool:
    markers = [match.group(1) for match in _OPTION_MARKER_RE.finditer(question)]
    return "A" in markers and "B" in markers


def is_mcq_like(row: dict[str, Any]) -> bool:
    return bool(row.get("options")) or (
        all_letter_gold(row) and has_inline_options(str(row.get("question", "")))
    )


def extract_letter(text: str, options: list[str] | None, judger: Any) -> str:
    think_end = text.rfind("</think>")
    tail = text[think_end + len("</think>") :] if think_end >= 0 else text

    match = _BOXED_LETTER_RE.search(tail) or _BOXED_LETTER_RE.search(text)
    if match:
        return match.group(1).upper()

    if options:
        try:
            boxed_contents = judger.extract_all_boxed(tail) or judger.extract_all_boxed(text)
        except Exception:
            boxed_contents = []
        if boxed_contents:
            candidate = boxed_contents[-1]
            try:
                candidate_norm = judger.norm_ans_str(candidate)
            except Exception:
                candidate_norm = candidate
            for idx, option in enumerate(options):
                option_text = str(option).strip()
                if candidate.strip() == option_text or candidate_norm == option_text:
                    return chr(65 + idx)
                try:
                    if judger.is_equal(candidate_norm, judger.norm_ans_str(option_text)):
                        return chr(65 + idx)
                except Exception:
                    pass

    phrase_matches = list(_LETTER_PHRASE_RE.finditer(tail))
    if phrase_matches:
        return phrase_matches[-1].group(1).upper()

    valid = {chr(65 + idx) for idx in range(len(options or []))}
    if not valid:
        valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    matches = [letter for letter in re.findall(r"\b([A-Z])\b", tail.upper()) if letter in valid]
    return matches[-1] if matches else ""


def score_response(judger: Any, row: dict[str, Any], response: str) -> bool:
    gold = answer_values(row)

    if row.get("options") and single_letter_gold(row):
        letter = extract_letter(response, row.get("options"), judger)
        if letter:
            return letter == gold[0].upper()

    try:
        return bool(judger.auto_judge(pred=response, gold=gold, options=[[]] * len(gold)))
    except Exception:
        return False


def extract_answer(judger: Any, response: str) -> str:
    try:
        return judger.extract_ans(response) or ""
    except Exception:
        return ""


def score_records(
    public_rows: dict[int, dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    run_name: str,
) -> tuple[list[dict[str, Any]], ScoreSummary, list[int]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from judger import Judger

    judger = Judger(strict_extract=False)
    summary = ScoreSummary(run_name=run_name)
    scored: list[dict[str, Any]] = []
    wrong_ids: list[int] = []

    for rec in prediction_records:
        row_id = int(rec["id"])
        if row_id not in public_rows:
            raise KeyError(f"Prediction id {row_id} is not present in public data")

        row = public_rows[row_id]
        response = str(rec.get("response", ""))
        correct = score_response(judger, row, response)
        extracted = extract_answer(judger, response)
        mcq = is_mcq_like(row)

        summary.total += 1
        summary.correct += int(correct)
        if mcq:
            summary.mcq_total += 1
            summary.mcq_correct += int(correct)
        else:
            summary.free_total += 1
            summary.free_correct += int(correct)
        if not correct:
            wrong_ids.append(row_id)

        out = dict(rec)
        gold = answer_values(row)
        out.update(
            {
                "id": row_id,
                "is_mcq": mcq,
                "gold": gold if len(gold) != 1 else gold[0],
                "extracted": extracted,
                "correct": correct,
                "error_type": "correct" if correct else "wrong",
            }
        )
        scored.append(out)

    return scored, summary, wrong_ids


def write_scored_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_summary(
    path: Path,
    summary: ScoreSummary,
    prediction_path: Path,
    data_path: Path,
    analysis_path: Path,
    html_path: Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fields = [
        "timestamp",
        "run_name",
        "prediction_path",
        "data_path",
        "total",
        "correct",
        "accuracy",
        "mcq_correct",
        "mcq_total",
        "mcq_accuracy",
        "free_correct",
        "free_total",
        "free_accuracy",
        "analysis_path",
        "html_path",
    ]
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": summary.run_name,
        "prediction_path": str(prediction_path),
        "data_path": str(data_path),
        "total": summary.total,
        "correct": summary.correct,
        "accuracy": f"{summary.accuracy:.6f}",
        "mcq_correct": summary.mcq_correct,
        "mcq_total": summary.mcq_total,
        "mcq_accuracy": f"{summary.mcq_accuracy:.6f}" if summary.mcq_total else "",
        "free_correct": summary.free_correct,
        "free_total": summary.free_total,
        "free_accuracy": f"{summary.free_accuracy:.6f}" if summary.free_total else "",
        "analysis_path": str(analysis_path),
        "html_path": str(html_path or ""),
    }
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def refresh_visualizations() -> None:
    subprocess.run(
        [sys.executable, "result_analyze/visualize_wrong.py"],
        check=True,
    )


def print_summary(summary: ScoreSummary, wrong_ids: list[int], show_wrong: int) -> None:
    print(f"Accuracy: {summary.correct}/{summary.total} = {summary.accuracy:.2%}")
    if summary.mcq_total:
        print(
            f"MCQ-like: {summary.mcq_correct}/{summary.mcq_total} = "
            f"{summary.mcq_accuracy:.2%}"
        )
    if summary.free_total:
        print(
            f"Free-response: {summary.free_correct}/{summary.free_total} = "
            f"{summary.free_accuracy:.2%}"
        )
    if show_wrong and wrong_ids:
        shown = wrong_ids[:show_wrong]
        suffix = "" if len(shown) == len(wrong_ids) else f" ... (+{len(wrong_ids) - len(shown)} more)"
        print("Wrong ids:", ", ".join(str(row_id) for row_id in shown) + suffix)


def main() -> None:
    args = parse_args()
    prediction_path = resolve_prediction_path(args.prediction)
    run_name = infer_run_name(prediction_path, args.name)
    analysis_path = args.analysis_dir / f"{run_name}.jsonl"
    html_path = args.analysis_dir / "visualizations" / f"{run_name}.html"

    public_rows = load_public(args.data_path)
    prediction_records = load_predictions(prediction_path)
    scored, summary, wrong_ids = score_records(public_rows, prediction_records, run_name)

    write_scored_jsonl(analysis_path, scored)
    print_summary(summary, wrong_ids, args.show_wrong)
    print(f"Scored JSONL: {analysis_path}")
    sys.stdout.flush()

    visualized = False
    if not args.no_visualize:
        refresh_visualizations()
        visualized = True
        print(f"HTML report: {html_path}")
        print(f"HTML index: {args.analysis_dir / 'visualizations' / 'index.html'}")

    append_summary(
        args.summary_path,
        summary,
        prediction_path,
        args.data_path,
        analysis_path,
        html_path if visualized else None,
    )
    print(f"Summary CSV: {args.summary_path}")


if __name__ == "__main__":
    main()
