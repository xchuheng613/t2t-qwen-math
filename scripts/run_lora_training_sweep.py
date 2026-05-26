#!/usr/bin/env python3
"""Run a small LoRA training sweep and summarize trainer metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
ALL_TARGETS = ATTENTION + MLP


@dataclass(frozen=True)
class Experiment:
    name: str
    epochs: float
    lr: float
    optim: str
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


EXPERIMENTS = [
    Experiment("lr5e-5_adamw_all_r16_e6", 6.0, 5e-5, "adamw_torch", 16, 32, 0.05, ALL_TARGETS),
    Experiment("lr5e-5_adamw_all_r16_e4", 4.0, 5e-5, "adamw_torch", 16, 32, 0.05, ALL_TARGETS),
    Experiment("lr5e-5_adamw_all_r16_e2", 2.0, 5e-5, "adamw_torch", 16, 32, 0.05, ALL_TARGETS),
    Experiment("lr5e-5_adamw_attention_r16_e2", 2.0, 5e-5, "adamw_torch", 16, 32, 0.05, ATTENTION),
    Experiment("lr2e-5_adamw_all_r16_e1", 1.0, 2e-5, "adamw_torch", 16, 32, 0.05, ALL_TARGETS),
    Experiment("lr1e-5_adamw_all_r16_e2", 2.0, 1e-5, "adamw_torch", 16, 32, 0.05, ALL_TARGETS),
    Experiment("lr2e-5_adamw_all_r8_e2", 2.0, 2e-5, "adamw_torch", 8, 16, 0.05, ALL_TARGETS),
    Experiment("lr2e-5_adamw_attention_r16_e2", 2.0, 2e-5, "adamw_torch", 16, 32, 0.05, ATTENTION),
    Experiment("lr2e-5_adamw_mlp_r16_e2", 2.0, 2e-5, "adamw_torch", 16, 32, 0.05, MLP),
    Experiment("lr2e-5_adafactor_all_r16_e1", 1.0, 2e-5, "adafactor", 16, 32, 0.05, ALL_TARGETS),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable or "python3")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--train-file", default="data/sft_lora_public_v2/train.jsonl")
    parser.add_argument("--eval-file", default="data/sft_lora_public_v2/dev.jsonl")
    parser.add_argument("--output-root", default="checkpoints/lora_training_sweep_v3")
    parser.add_argument("--analysis-dir", default="analysis/lora_training_sweep_v3")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--only", nargs="*", default=None, help="Optional experiment names to run.")
    parser.add_argument("--force", action="store_true", help="Rerun experiments even if a trainer_state.json exists.")
    return parser.parse_args()


def train_command(args: argparse.Namespace, exp: Experiment) -> list[str]:
    return [
        args.python,
        "scripts/train_lora_sft.py",
        "--model",
        args.model,
        "--train-file",
        args.train_file,
        "--eval-file",
        args.eval_file,
        "--output-dir",
        str(Path(args.output_root) / exp.name),
        "--epochs",
        str(exp.epochs),
        "--lr",
        f"{exp.lr:.8g}",
        "--optim",
        exp.optim,
        "--rank",
        str(exp.rank),
        "--alpha",
        str(exp.alpha),
        "--dropout",
        str(exp.dropout),
        "--grad-accum",
        str(args.grad_accum),
        "--batch-size",
        str(args.batch_size),
        "--max-seq-length",
        str(args.max_seq_length),
        "--logging-steps",
        str(args.logging_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--save-steps",
        str(args.save_steps),
        "--target-modules",
        *exp.target_modules,
    ]


def latest_trainer_state(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.rsplit("-", 1)[-1]),
    )
    return candidates[-1] if candidates else None


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def best_eval(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rows = [row for row in log_history if row.get("eval_loss") is not None]
    if not eval_rows:
        return {}
    return min(eval_rows, key=lambda row: float(row["eval_loss"]))


def last_train_before(log_history: list[dict[str, Any]], step: int | None) -> dict[str, Any]:
    train_rows = [row for row in log_history if row.get("loss") is not None]
    if step is None:
        return train_rows[-1] if train_rows else {}
    before = [row for row in train_rows if int(row.get("step") or -1) <= step]
    return before[-1] if before else (train_rows[-1] if train_rows else {})


def summarize_experiment(exp: Experiment, output_root: Path) -> dict[str, Any]:
    output_dir = output_root / exp.name
    state_path = latest_trainer_state(output_dir)
    row: dict[str, Any] = {
        "experiment": exp.name,
        "epochs": exp.epochs,
        "lr": exp.lr,
        "optim": exp.optim,
        "rank": exp.rank,
        "alpha": exp.alpha,
        "dropout": exp.dropout,
        "target_modules": " ".join(exp.target_modules),
        "status": "missing",
    }
    if state_path is None:
        return row
    state = load_json(state_path)
    logs = state.get("log_history", [])
    best = best_eval(logs)
    best_step = int(best.get("step")) if best.get("step") is not None else None
    train_at_best = last_train_before(logs, best_step)
    last_train = last_train_before(logs, None)
    row.update(
        {
            "status": "done",
            "global_step": state.get("global_step"),
            "final_epoch": state.get("epoch"),
            "best_eval_step": best_step,
            "best_eval_epoch": best.get("epoch"),
            "best_eval_loss": best.get("eval_loss"),
            "best_eval_token_accuracy": best.get("eval_mean_token_accuracy"),
            "train_loss_at_best_eval": train_at_best.get("loss"),
            "train_token_accuracy_at_best_eval": train_at_best.get("mean_token_accuracy"),
            "last_train_step": last_train.get("step"),
            "last_train_loss": last_train.get("loss"),
            "last_train_token_accuracy": last_train.get("mean_token_accuracy"),
            "state_path": str(state_path),
        }
    )
    return row


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only or [])
    experiments = [exp for exp in EXPERIMENTS if not wanted or exp.name in wanted]
    if wanted:
        unknown = wanted - {exp.name for exp in EXPERIMENTS}
        if unknown:
            raise SystemExit(f"Unknown experiments: {sorted(unknown)}")

    rows: list[dict[str, Any]] = []
    for idx, exp in enumerate(experiments, start=1):
        output_dir = output_root / exp.name
        state_path = latest_trainer_state(output_dir)
        if state_path and not args.force:
            print(f"[{idx}/{len(experiments)}] skip existing {exp.name}: {state_path}", flush=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = analysis_dir / f"{exp.name}.log"
            cmd = train_command(args, exp)
            print(f"[{idx}/{len(experiments)}] train {exp.name}", flush=True)
            print("  " + " ".join(cmd), flush=True)
            start = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - start
            print(f"  exit={proc.returncode} elapsed={elapsed:.1f}s log={log_path}", flush=True)
            if proc.returncode:
                raise SystemExit(proc.returncode)
        rows = [summarize_experiment(item, output_root) for item in EXPERIMENTS]
        write_summary(analysis_dir / "summary.csv", rows)

    done = [row for row in rows if row.get("status") == "done" and row.get("best_eval_loss") not in (None, "")]
    if done:
        best = min(done, key=lambda row: float(row["best_eval_loss"]))
        print(
            "best_by_eval_loss="
            f"{best['experiment']} step={best['best_eval_step']} "
            f"eval_loss={float(best['best_eval_loss']):.6f} "
            f"eval_acc={float(best['best_eval_token_accuracy']):.2%}",
            flush=True,
        )
    print(f"summary={analysis_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
