#!/usr/bin/env python3
"""Write reproducible one-epoch LoRA experiment commands.

The plan is intentionally conservative: lower learning rates, one optimizer
swap, and narrower adapter targets so each run answers one debugging question.
Use --run only when you really want to execute the generated training commands.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
ALL_TARGETS = ATTENTION + MLP


@dataclass(frozen=True)
class Experiment:
    name: str
    lr: float
    optim: str
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    rationale: str


EXPERIMENTS = [
    Experiment(
        name="lr2e-5_adamw_all_r16",
        lr=2e-5,
        optim="adamw_torch",
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=ALL_TARGETS,
        rationale="Same adapter capacity as the failed run, but lower LR to test whether 5e-5 was too aggressive.",
    ),
    Experiment(
        name="lr1e-5_adamw_all_r16",
        lr=1e-5,
        optim="adamw_torch",
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=ALL_TARGETS,
        rationale="Slower LR check. If train loss barely moves, the run is underfitting or needs more steps/data.",
    ),
    Experiment(
        name="lr2e-5_adafactor_all_r16",
        lr=2e-5,
        optim="adafactor",
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=ALL_TARGETS,
        rationale="Optimizer swap requested by TA; compare stability and validation loss against AdamW.",
    ),
    Experiment(
        name="lr2e-5_adamw_attention_r16",
        lr=2e-5,
        optim="adamw_torch",
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=ATTENTION,
        rationale="Attention-only adapter. Less capacity than all projections, useful when the data is small.",
    ),
    Experiment(
        name="lr2e-5_adamw_mlp_r16",
        lr=2e-5,
        optim="adamw_torch",
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=MLP,
        rationale="Feed-forward-only adapter. Separates reasoning/style changes from attention routing changes.",
    ),
    Experiment(
        name="lr2e-5_adamw_all_r8",
        lr=2e-5,
        optim="adamw_torch",
        rank=8,
        alpha=16,
        dropout=0.05,
        target_modules=ALL_TARGETS,
        rationale="Half-rank adapter. Tests whether the 33M-parameter adapter was too large for 371 examples.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable or "python3")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--train-file", default="data/sft_lora_public_v2/train.jsonl")
    parser.add_argument("--eval-file", default="data/sft_lora_public_v2/dev.jsonl")
    parser.add_argument("--train-accuracy-file", default="data/lora_public_v2/train_free_keep.jsonl")
    parser.add_argument("--dev-accuracy-file", default="data/lora_public_v2/public_dev.jsonl")
    parser.add_argument("--holdout-accuracy-file", default="data/lora_public_v2/holdout_free.jsonl")
    parser.add_argument("--output-root", default="checkpoints/lora_sweep_v3")
    parser.add_argument("--merged-root", default="checkpoints/merged_lora_sweep_v3")
    parser.add_argument("--results-root", default="results/lora_sweep_v3")
    parser.add_argument("--summary-csv", default="results/lora_sweep_v3/benchmark_summary.csv")
    parser.add_argument("--plan-dir", default="analysis/lora_experiment_plan")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="Use the same cadence as training logs so train/eval curves have comparable points.",
    )
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--run", action="store_true", help="Execute only the training commands after writing the plan.")
    return parser.parse_args()


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


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
        str(args.epochs),
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
        str(args.eval_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--save-steps",
        str(args.save_steps),
        "--target-modules",
        *exp.target_modules,
    ]


def merge_command(args: argparse.Namespace, exp: Experiment) -> list[str]:
    return [
        args.python,
        "scripts/merge_lora.py",
        "--adapter",
        str(Path(args.output_root) / exp.name),
        "--out",
        str(Path(args.merged_root) / exp.name),
    ]


def eval_command(args: argparse.Namespace, exp: Experiment, split: str, data_path: str) -> list[str]:
    return [
        args.python,
        "scripts/run_math_prompts.py",
        "--mode",
        "submission_response_mode",
        "--data-path",
        data_path,
        "--output-dir",
        str(Path(args.results_root) / exp.name / split),
        "--model",
        str(Path(args.merged_root) / exp.name),
        "--config",
        "greedy_n1",
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--normalize-free-final-answers",
    ]


def score_command(args: argparse.Namespace, exp: Experiment, split: str, data_path: str) -> list[str]:
    analysis_name = f"{Path(args.results_root).name}_{exp.name}_{split}.jsonl"
    return [
        args.python,
        "scripts/score_benchmark.py",
        str(Path(args.results_root) / exp.name / split),
        "--data-path",
        data_path,
        "--name",
        f"{exp.name}_{split}",
        "--summary-csv",
        args.summary_csv,
        "--scored-jsonl",
        str(Path("analysis") / analysis_name),
    ]


def write_outputs(args: argparse.Namespace) -> tuple[Path, Path]:
    plan_dir = Path(args.plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    csv_path = plan_dir / "lora_experiment_commands.csv"
    md_path = plan_dir / "README.md"
    rows: list[dict[str, str]] = []
    split_paths = {
        "train": args.train_accuracy_file,
        "validation": args.dev_accuracy_file,
        "holdout": args.holdout_accuracy_file,
    }

    for exp in EXPERIMENTS:
        rows.append(
            {
                "experiment": exp.name,
                "stage": "train",
                "rationale": exp.rationale,
                "command": shell_join(train_command(args, exp)),
            }
        )
        rows.append(
            {
                "experiment": exp.name,
                "stage": "merge",
                "rationale": "Merge the adapter so vLLM can evaluate the tuned model.",
                "command": shell_join(merge_command(args, exp)),
            }
        )
        for split, data_path in split_paths.items():
            rows.append(
                {
                    "experiment": exp.name,
                    "stage": f"eval_{split}",
                    "rationale": f"Generate {split} predictions for task accuracy and error inspection.",
                    "command": shell_join(eval_command(args, exp, split, data_path)),
                }
            )
            rows.append(
                {
                    "experiment": exp.name,
                    "stage": f"score_{split}",
                    "rationale": f"Score {split} predictions and append diagnostics to the sweep summary.",
                    "command": shell_join(score_command(args, exp, split, data_path)),
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["experiment", "stage", "rationale", "command"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# LoRA Experiment Plan",
        "",
        "All experiments are one epoch and use eval/logging every 10 optimizer steps by default.",
        "Compare training loss against validation loss; keep a run only if train loss decreases without validation loss blowing up and task accuracy improves.",
        "The score commands also write `analysis/*.jsonl`; run `python3 analysis/visualize_wrong.py` afterward to inspect train/validation/holdout error patterns in HTML.",
        "",
        "## Experiments",
        "",
    ]
    for exp in EXPERIMENTS:
        lines.extend(
            [
                f"### {exp.name}",
                "",
                f"- LR/optimizer: `{exp.lr:.8g}` / `{exp.optim}`",
                f"- Rank/alpha/dropout: `{exp.rank}` / `{exp.alpha}` / `{exp.dropout}`",
                f"- Target modules: `{', '.join(exp.target_modules)}`",
                f"- Why: {exp.rationale}",
                "",
                "```bash",
                shell_join(train_command(args, exp)),
                shell_join(merge_command(args, exp)),
                shell_join(eval_command(args, exp, "train", args.train_accuracy_file)),
                shell_join(score_command(args, exp, "train", args.train_accuracy_file)),
                shell_join(eval_command(args, exp, "validation", args.dev_accuracy_file)),
                shell_join(score_command(args, exp, "validation", args.dev_accuracy_file)),
                shell_join(eval_command(args, exp, "holdout", args.holdout_accuracy_file)),
                shell_join(score_command(args, exp, "holdout", args.holdout_accuracy_file)),
                "```",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    args = parse_args()
    csv_path, md_path = write_outputs(args)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")

    if args.run:
        for exp in EXPERIMENTS:
            cmd = train_command(args, exp)
            print(f"running {exp.name}: {shell_join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
