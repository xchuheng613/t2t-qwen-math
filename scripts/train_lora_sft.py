#!/usr/bin/env python3
"""Train a small LoRA SFT adapter on compact-prompt JSONL data."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def supported_kwargs(cls: type[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def build_sft_config(args: argparse.Namespace) -> SFTConfig:
    """Build SFTConfig across TRL versions with small API differences."""
    # Qwen's shipped chat template does not expose assistant token masks, so
    # flattened-message runs apply assistant masking manually before SFTTrainer.
    assistant_only_loss = args.assistant_only_loss and not args.flatten_messages
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "max_seq_length": args.max_seq_length,
        "max_length": args.max_seq_length,
        "dataset_text_field": "text",
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "optim": args.optim,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": True,
        "assistant_only_loss": assistant_only_loss,
        "report_to": "none",
        "packing": False,
    }
    return SFTConfig(**supported_kwargs(SFTConfig, kwargs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--train-file", default="data/sft_free_v1/train.jsonl")
    parser.add_argument("--eval-file", default="data/sft_free_v1/dev.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/qwen3_4b_free_lora_v1")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Projection modules to adapt. Use attention only: q_proj k_proj v_proj o_proj; MLP only: gate_proj up_proj down_proj.",
    )
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--flatten-messages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert messages to a plain text field before TRL sees them. This avoids chat-template generation-marker requirements.",
    )
    return parser.parse_args()


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return encoded["input_ids"], encoded["offset_mapping"]


def tokenize_assistant_only_dataset(dataset: Any, tokenizer: Any, max_length: int) -> Any:
    """Flatten chat rows but mask loss to assistant-response tokens only."""

    def format_row(example: dict[str, Any]) -> dict[str, list[int]]:
        messages = example.get("messages")
        if not messages:
            text = str(example.get("text", ""))
            if tokenizer.eos_token and not text.rstrip().endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            input_ids = _tokenize_text(tokenizer, text)
            return {
                "input_ids": input_ids[:max_length],
                "attention_mask": [1] * min(len(input_ids), max_length),
                "labels": input_ids[:max_length],
            }

        prompt_messages = messages[:-1] if messages[-1].get("role") == "assistant" else messages
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if tokenizer.eos_token and not full_text.rstrip().endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token

        if full_text.startswith(prompt_text):
            answer_start = len(prompt_text)
        else:
            marker = "<|im_start|>assistant\n"
            marker_idx = full_text.rfind(marker)
            if marker_idx < 0:
                raise ValueError("Rendered chat text does not contain an assistant message marker.")
            answer_start = marker_idx + len(marker)
            think_prefix = "<think>\n"
            if full_text.startswith(think_prefix, answer_start):
                answer_start += len(think_prefix)

        input_ids, offsets = _tokenize_with_offsets(tokenizer, full_text)
        input_ids = input_ids[:max_length]
        offsets = offsets[:max_length]
        labels = input_ids.copy()
        labels = [
            label if start >= answer_start else -100
            for label, (start, _end) in zip(labels, offsets)
        ]
        if all(label == -100 for label in labels):
            raise ValueError("Assistant-only label mask removed every token; increase --max-seq-length.")
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    return dataset.map(format_row, remove_columns=dataset.column_names)


def flatten_messages_dataset(dataset: Any, tokenizer: Any) -> Any:
    """Turn conversational rows into a plain text SFT dataset."""

    def format_row(example: dict[str, Any]) -> dict[str, str]:
        messages = example.get("messages")
        if messages:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = str(example.get("text", ""))
        if tokenizer.eos_token and not text.rstrip().endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        return {"text": text}

    return dataset.map(format_row, remove_columns=dataset.column_names)


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")
    if args.limit_train > 0:
        train_dataset = train_dataset.select(range(min(args.limit_train, len(train_dataset))))
    if args.limit_eval > 0:
        eval_dataset = eval_dataset.select(range(min(args.limit_eval, len(eval_dataset))))
    if args.flatten_messages and args.assistant_only_loss:
        train_dataset = tokenize_assistant_only_dataset(train_dataset, tokenizer, args.max_seq_length)
        eval_dataset = tokenize_assistant_only_dataset(eval_dataset, tokenizer, args.max_seq_length)
        print("Using manual assistant-only labels with flattened chat messages.", flush=True)
    elif args.flatten_messages:
        train_dataset = flatten_messages_dataset(train_dataset, tokenizer)
        eval_dataset = flatten_messages_dataset(eval_dataset, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )

    trainer_kwargs = {
        "model": model,
        "args": build_sft_config(args),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "dataset_text_field": "text",
    }
    trainer = SFTTrainer(**supported_kwargs(SFTTrainer, trainer_kwargs))
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
