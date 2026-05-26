#!/usr/bin/env python3
"""Create plots and a short markdown report for a LoRA training sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("analysis/lora_training_sweep_v3"),
        help="Directory containing summary.csv and sweep logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where report.md and SVG plots should be written. Defaults to sweep-dir.",
    )
    return parser.parse_args()


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [row for row in rows if row.get("status") == "done"]


def load_state(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else float("nan")


def best_checkpoint(row: dict[str, str]) -> Path:
    state_path = Path(row["state_path"])
    run_dir = state_path.parent.parent
    step = int(float(row["best_eval_step"]))
    return run_dir / f"checkpoint-{step}"


def plot_eval_loss_comparison(rows: list[dict[str, str]], out: Path) -> None:
    rows = sorted(rows, key=lambda row: as_float(row, "best_eval_loss"))
    labels = [row["experiment"] for row in rows]
    losses = [as_float(row, "best_eval_loss") for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#27806b" if i == 0 else "#7a8a9a" for i in range(len(rows))]
    ax.bar(range(len(rows)), losses, color=colors)
    ax.set_ylabel("Best validation loss")
    ax.set_title("LoRA sweep: validation loss by experiment")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def split_history(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for item in state.get("log_history", []):
        if "loss" in item:
            train_rows.append(item)
        if "eval_loss" in item:
            eval_rows.append(item)
    return train_rows, eval_rows


def plot_best_curve(state: dict[str, Any], best_row: dict[str, str], out_loss: Path, out_acc: Path) -> None:
    train_rows, eval_rows = split_history(state)
    best_epoch = as_float(best_row, "best_eval_epoch")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        [row["epoch"] for row in train_rows],
        [row["loss"] for row in train_rows],
        marker="o",
        label="train loss",
        color="#2d6cdf",
    )
    ax.plot(
        [row["epoch"] for row in eval_rows],
        [row["eval_loss"] for row in eval_rows],
        marker="o",
        label="validation loss",
        color="#c2410c",
    )
    ax.axvline(best_epoch, color="#27806b", linestyle="--", linewidth=1.5, label="best stop")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Winning run loss curve: {best_row['experiment']}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_loss)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        [row["epoch"] for row in train_rows],
        [100 * row["mean_token_accuracy"] for row in train_rows],
        marker="o",
        label="train token accuracy",
        color="#2d6cdf",
    )
    ax.plot(
        [row["epoch"] for row in eval_rows],
        [100 * row["eval_mean_token_accuracy"] for row in eval_rows],
        marker="o",
        label="validation token accuracy",
        color="#c2410c",
    )
    ax.axvline(best_epoch, color="#27806b", linestyle="--", linewidth=1.5, label="best stop")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Token accuracy (%)")
    ax.set_title(f"Winning run token accuracy: {best_row['experiment']}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_acc)
    plt.close(fig)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Experiment | Best epoch | Best eval loss | Eval token acc | Train loss at best | Targets |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["experiment"],
                    f"{as_float(row, 'best_eval_epoch'):.3f}",
                    f"{as_float(row, 'best_eval_loss'):.4f}",
                    f"{100 * as_float(row, 'best_eval_token_accuracy'):.2f}%",
                    f"{as_float(row, 'train_loss_at_best_eval'):.4f}",
                    row["target_modules"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]], best_row: dict[str, str], out: Path) -> None:
    rows = sorted(rows, key=lambda row: as_float(row, "best_eval_loss"))
    checkpoint = best_checkpoint(best_row)
    final_state = Path(best_row["state_path"])
    state = load_state(final_state)
    train_rows, eval_rows = split_history(state)
    final_eval = eval_rows[-1] if eval_rows else {}

    report = f"""# LoRA Training Sweep Report

Best validation checkpoint:

- experiment: `{best_row['experiment']}`
- checkpoint: `{checkpoint}`
- best epoch: `{as_float(best_row, 'best_eval_epoch'):.3f}`
- best step: `{int(float(best_row['best_eval_step']))}`
- best validation loss: `{as_float(best_row, 'best_eval_loss'):.6f}`
- best validation token accuracy: `{100 * as_float(best_row, 'best_eval_token_accuracy'):.2f}%`
- train loss at best validation point: `{as_float(best_row, 'train_loss_at_best_eval'):.6f}`
- train token accuracy at best validation point: `{100 * as_float(best_row, 'train_token_accuracy_at_best_eval'):.2f}%`

Recommended hyperparameters:

- optimizer: `{best_row['optim']}`
- learning rate: `{best_row['lr']}`
- LoRA rank / alpha / dropout: `{best_row['rank']} / {best_row['alpha']} / {best_row['dropout']}`
- target modules: `{best_row['target_modules']}`
- stop around epoch: `{as_float(best_row, 'best_eval_epoch'):.2f}`; the 4-epoch run's final eval loss was `{float(final_eval.get('eval_loss', float('nan'))):.6f}`, so later epochs did not improve validation loss.

Plots:

- `best_train_eval_loss.svg`
- `best_token_accuracy.svg`
- `eval_loss_comparison.svg`

## Results

{markdown_table(rows)}
"""
    out.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir
    output_dir = args.output_dir or sweep_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_summary(sweep_dir / "summary.csv")
    if not rows:
        raise SystemExit(f"No completed rows found in {sweep_dir / 'summary.csv'}")

    best_row = min(rows, key=lambda row: as_float(row, "best_eval_loss"))
    state = load_state(Path(best_row["state_path"]))

    plot_eval_loss_comparison(rows, output_dir / "eval_loss_comparison.svg")
    plot_best_curve(
        state,
        best_row,
        output_dir / "best_train_eval_loss.svg",
        output_dir / "best_token_accuracy.svg",
    )
    write_report(rows, best_row, output_dir / "report.md")
    print(f"report={output_dir / 'report.md'}")
    print(f"best_checkpoint={best_checkpoint(best_row)}")


if __name__ == "__main__":
    main()
