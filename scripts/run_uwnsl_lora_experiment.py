#!/usr/bin/env python3
"""Print or run the UWNSL Mix-Long QLoRA experiment commands."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ALL_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable or "python3")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--dataset", default="UWNSL/Mix-Long_long_0.2_short_0.8")
    parser.add_argument("--output-dir", default="checkpoints/uwnsl_mix_long_lora/mix_long_r16_a32_q4")
    parser.add_argument("--stage", choices=["epoch1", "epoch2"], default="epoch1")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--local-eval-file", default="")
    parser.add_argument("--local-eval-format", choices=["auto", "messages", "public"], default="auto")
    parser.add_argument("--local-eval-prompt", default="compact")
    parser.add_argument("--eval-strategy", choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--attn-implementation", default="fa2", choices=["", "eager", "sdpa", "flash_attention_2", "fa2"])
    parser.add_argument("--run", action="store_true", help="Execute the command instead of only printing it.")
    return parser.parse_args()


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        candidates.append((step, path))
    return max(candidates)[1] if candidates else None


def build_command(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir)
    epochs = "1" if args.stage == "epoch1" else "2"
    resume_checkpoint = args.resume_from_checkpoint
    if args.stage == "epoch2" and not resume_checkpoint:
        checkpoint = latest_checkpoint(output_dir)
        if checkpoint is None:
            raise SystemExit(
                f"No checkpoint found under {output_dir}. Run epoch1 first or pass --resume-from-checkpoint."
            )
        resume_checkpoint = str(checkpoint)

    cmd = [
        args.python,
        "scripts/train_lora_sft.py",
        "--model",
        args.model,
        "--hf-dataset",
        args.dataset,
        "--output-dir",
        str(output_dir),
        "--epochs",
        epochs,
        "--lr",
        f"{args.lr:.8g}",
        "--rank",
        str(args.rank),
        "--alpha",
        str(args.alpha),
        "--dropout",
        str(args.dropout),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--max-seq-length",
        str(args.max_seq_length),
        "--quantization-bit",
        "4",
        "--attn-implementation",
        args.attn_implementation,
        "--gradient-checkpointing",
        "--save-strategy",
        "epoch",
        "--eval-strategy",
        args.eval_strategy,
        "--save-total-limit",
        str(args.save_total_limit),
        "--logging-steps",
        str(args.logging_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--target-modules",
        *ALL_TARGETS,
    ]
    if args.local_eval_file:
        cmd.extend(
            [
                "--hf-local-eval-file",
                args.local_eval_file,
                "--hf-local-eval-format",
                args.local_eval_format,
                "--hf-local-eval-prompt",
                args.local_eval_prompt,
                "--hf-eval-size",
                "0",
            ]
        )
    else:
        cmd.extend(["--hf-eval-size", str(args.eval_size)])
    if resume_checkpoint:
        cmd.extend(["--resume-from-checkpoint", resume_checkpoint])
    return cmd


def main() -> None:
    args = parse_args()
    cmd = build_command(args)
    print(" ".join(cmd), flush=True)
    if args.run:
        raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
