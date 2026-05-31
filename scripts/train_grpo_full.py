#!/usr/bin/env python3
"""Full-parameter GRPO training for free-response math rows.

This script is intended for an 80 GB cloud GPU. It does not use LoRA/PEFT.
The default data files are already free-response only, so the loader does not
filter rows.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompts.grpo_prompt_pack import full_grpo_reward, make_grpo_dataset_rows


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_TRAIN_FILE = "data/public_free_response.jsonl"
DEFAULT_EVAL_FILE = "data/public_dev.jsonl"
DEFAULT_OUTPUT_DIR = "checkpoints/full_grpo_free/qwen3_4b"


def require_cloud_deps() -> tuple[Any, Any, Any, Any, Any]:
    """Import heavy training dependencies with an actionable error."""
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing GRPO training dependency. On the cloud machine, install the "
            "project requirements first, then install a TRL version with "
            "GRPOTrainer support. Example:\n\n"
            "  pip install -r requirements.txt\n"
            "  pip install -U trl transformers accelerate datasets flash-attn\n"
        ) from exc
    return torch, Dataset, AutoModelForCausalLM, AutoTokenizer, (GRPOConfig, GRPOTrainer)


def supported_kwargs(cls: type[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls.__init__).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_dataset(dataset_cls: Any, path: str, limit: int | None) -> Any:
    rows = load_jsonl(Path(path), limit=limit)
    if not rows:
        raise ValueError(f"{path} has no rows")
    return dataset_cls.from_list(make_grpo_dataset_rows(rows))


def build_grpo_config(args: argparse.Namespace, config_cls: type[Any]) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "fp16": False,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "eval_strategy": "steps" if args.eval_file else "no",
        "evaluation_strategy": "steps" if args.eval_file else "no",
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_drop_last": args.dataloader_drop_last,
        "optim": args.optim,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_first_step": True,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "beta": args.beta,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "generation_batch_size": args.generation_batch_size,
        "log_completions": args.log_completions,
        "ddp_find_unused_parameters": False,
    }
    return config_cls(**supported_kwargs(config_cls, kwargs))


def load_model_and_tokenizer(args: argparse.Namespace, torch: Any, model_cls: Any, tokenizer_cls: Any) -> tuple[Any, Any]:
    tokenizer = tokenizer_cls.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = model_cls.from_pretrained(args.model, **model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)

    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=4096)

    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)

    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--log-completions", action="store_true")

    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--dataloader-drop-last", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_grpo_batch(args: argparse.Namespace) -> None:
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_global = world_size * args.per_device_train_batch_size
    effective = micro_global * args.gradient_accumulation_steps
    if args.num_generations < 2:
        raise SystemExit("GRPO requires --num-generations >= 2.")
    if effective % args.num_generations != 0:
        raise SystemExit(
            "Invalid GRPO batch geometry: "
            f"WORLD_SIZE({world_size}) * per_device_train_batch_size({args.per_device_train_batch_size}) "
            f"* gradient_accumulation_steps({args.gradient_accumulation_steps}) = {effective}, "
            f"which is not divisible by num_generations({args.num_generations})."
        )
    if micro_global % args.num_generations != 0:
        print(
            "Warning: micro global batch is not divisible by num_generations. "
            "Recent TRL versions allow this when the effective batch is divisible, "
            "but older TRL versions may require increasing --per-device-train-batch-size.",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    validate_grpo_batch(args)
    torch, dataset_cls, model_cls, tokenizer_cls, grpo_classes = require_cloud_deps()
    grpo_config_cls, grpo_trainer_cls = grpo_classes

    train_dataset = build_dataset(dataset_cls, args.train_file, args.train_limit)
    eval_dataset = build_dataset(dataset_cls, args.eval_file, args.eval_limit) if args.eval_file else None
    model, tokenizer = load_model_and_tokenizer(args, torch, model_cls, tokenizer_cls)
    config = build_grpo_config(args, grpo_config_cls)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": config,
        "reward_funcs": [full_grpo_reward],
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = grpo_trainer_cls(**supported_kwargs(grpo_trainer_cls, trainer_kwargs))

    print("Full-parameter GRPO configuration:")
    print(f"  model={args.model}")
    print(f"  train_file={args.train_file} rows={len(train_dataset)}")
    print(f"  eval_file={args.eval_file} rows={len(eval_dataset) if eval_dataset is not None else 0}")
    print(f"  batch={args.per_device_train_batch_size} grad_accum={args.gradient_accumulation_steps}")
    print(f"  num_generations={args.num_generations}")
    print(f"  dataloader_drop_last={args.dataloader_drop_last}")
    print(f"  max_prompt_length={args.max_prompt_length}")
    print(f"  max_completion_length={args.max_completion_length}")
    print(f"  save_steps={args.save_steps} eval_steps={args.eval_steps} logging_steps={args.logging_steps}")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
