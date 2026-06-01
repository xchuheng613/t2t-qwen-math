#!/usr/bin/env python3
"""Train a small LoRA SFT adapter on JSONL or Hugging Face SFT data."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_HF_SYSTEM_PROMPT = """Solve the math problem. Reason carefully and end with a final boxed answer."""


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
        "eval_strategy": args.eval_strategy,
        "evaluation_strategy": args.eval_strategy,
        "save_strategy": args.save_strategy,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "assistant_only_loss": assistant_only_loss,
        "report_to": "none",
        "packing": False,
    }
    if args.save_total_limit > 0:
        kwargs["save_total_limit"] = args.save_total_limit
    return SFTConfig(**supported_kwargs(SFTConfig, kwargs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--train-file", default="data/sft_free_v1/train.jsonl")
    parser.add_argument("--eval-file", default="data/sft_free_v1/dev.jsonl")
    parser.add_argument("--hf-dataset", default="", help="Optional Hugging Face dataset ID to load instead of JSONL files.")
    parser.add_argument("--hf-train-split", default="train")
    parser.add_argument("--hf-eval-split", default="", help="Optional explicit HF eval split. If omitted, split eval rows from train.")
    parser.add_argument("--hf-eval-size", type=int, default=256, help="Eval rows held out from the HF train split when --hf-eval-split is omitted.")
    parser.add_argument("--hf-eval-seed", type=int, default=529)
    parser.add_argument(
        "--hf-local-eval-file",
        default="",
        help=(
            "Optional local JSONL eval file to use with --hf-dataset instead of "
            "an HF eval split/holdout. Supports message/text SFT rows or public "
            "benchmark rows with question/answer fields."
        ),
    )
    parser.add_argument(
        "--hf-local-eval-format",
        choices=["auto", "messages", "public"],
        default="auto",
        help="Schema for --hf-local-eval-file. 'auto' uses messages/text if present, else question/answer.",
    )
    parser.add_argument("--hf-local-eval-prompt", default="compact")
    parser.add_argument("--hf-problem-field", default="problem")
    parser.add_argument("--hf-solution-field", default="solution")
    parser.add_argument("--hf-system-prompt", default=DEFAULT_HF_SYSTEM_PROMPT)
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
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-strategy", choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--save-total-limit", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from-checkpoint", default="", help="Resume trainer state from a checkpoint directory.")
    parser.add_argument("--quantization-bit", type=int, choices=[0, 4, 8], default=0)
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn-implementation",
        default="",
        choices=["", "eager", "sdpa", "flash_attention_2", "fa2"],
        help="Attention backend passed to transformers. Use fa2 for flash_attention_2.",
    )
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


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _answer_values(row: dict[str, Any]) -> list[str]:
    answer = row.get("answer", row.get("answers", row.get("gold")))
    values = answer if isinstance(answer, list) else [answer]
    return [_clean_text(value) for value in values]


def problem_solution_to_messages(dataset: Any, args: argparse.Namespace) -> Any:
    """Convert HF rows with problem/solution fields to the chat-message schema."""

    def format_row(example: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        problem = _clean_text(example.get(args.hf_problem_field))
        solution = _clean_text(example.get(args.hf_solution_field))
        if not problem or not solution:
            raise ValueError(
                f"HF row is missing non-empty {args.hf_problem_field!r} or {args.hf_solution_field!r}."
            )
        return {
            "messages": [
                {"role": "system", "content": args.hf_system_prompt},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ]
        }

    return dataset.map(format_row, remove_columns=dataset.column_names)


def public_eval_to_messages(dataset: Any, args: argparse.Namespace) -> Any:
    """Convert public benchmark rows to compact-prompt eval messages."""
    from prompts.compact_prompt_pack import build_routed_prompt, rebuild_final_response

    def format_row(example: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        question = _clean_text(example.get("question", example.get(args.hf_problem_field)))
        answers = _answer_values(example)
        options = example.get("options") or None
        if not question or not answers or any(answer == "" for answer in answers):
            row_id = example.get("id", "<unknown>")
            raise ValueError(f"Local eval row {row_id} is missing a non-empty question or answer.")
        system, user = build_routed_prompt(args.hf_local_eval_prompt, question, options)
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": rebuild_final_response(answers, question)},
            ]
        }

    return dataset.map(format_row, remove_columns=dataset.column_names)


def load_local_eval_dataset(args: argparse.Namespace) -> Any:
    dataset = load_dataset("json", data_files=args.hf_local_eval_file, split="train")
    if args.hf_local_eval_format == "messages":
        return dataset
    if args.hf_local_eval_format == "auto" and (
        "messages" in dataset.column_names or "text" in dataset.column_names
    ):
        return dataset
    return public_eval_to_messages(dataset, args)


def load_train_eval_datasets(args: argparse.Namespace) -> tuple[Any, Any]:
    if args.hf_dataset:
        train_dataset = load_dataset(args.hf_dataset, split=args.hf_train_split)
        eval_is_local = False
        if args.hf_local_eval_file:
            eval_dataset = load_local_eval_dataset(args)
            eval_is_local = True
        elif args.hf_eval_split:
            eval_dataset = load_dataset(args.hf_dataset, split=args.hf_eval_split)
        elif args.hf_eval_size > 0:
            if len(train_dataset) <= args.hf_eval_size:
                raise ValueError("--hf-eval-size must be smaller than the HF train split.")
            split = train_dataset.train_test_split(
                test_size=args.hf_eval_size,
                seed=args.hf_eval_seed,
                shuffle=True,
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            eval_dataset = None
        train_dataset = problem_solution_to_messages(train_dataset, args)
        if eval_dataset is not None and not eval_is_local:
            eval_dataset = problem_solution_to_messages(eval_dataset, args)
        return train_dataset, eval_dataset

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset = None
    if args.eval_strategy != "no":
        eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")
    return train_dataset, eval_dataset


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


def torch_dtype_from_args(args: argparse.Namespace) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def normalized_attn_implementation(value: str) -> str | None:
    if not value:
        return None
    if value == "fa2":
        return "flash_attention_2"
    return value


def model_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch_dtype_from_args(args)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    attn_implementation = normalized_attn_implementation(args.attn_implementation)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if args.quantization_bit == 4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
    elif args.quantization_bit == 8:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    return kwargs


def write_training_manifest(args: argparse.Namespace, train_dataset: Any, eval_dataset: Any | None) -> None:
    manifest = {
        "model": args.model,
        "output_dir": args.output_dir,
        "hf_dataset": args.hf_dataset or None,
        "hf_train_split": args.hf_train_split if args.hf_dataset else None,
        "hf_eval_split": args.hf_eval_split or None,
        "hf_eval_size": (
            args.hf_eval_size
            if args.hf_dataset and not args.hf_eval_split and not args.hf_local_eval_file
            else None
        ),
        "hf_local_eval_file": args.hf_local_eval_file or None,
        "hf_local_eval_format": args.hf_local_eval_format if args.hf_local_eval_file else None,
        "hf_local_eval_prompt": args.hf_local_eval_prompt if args.hf_local_eval_file else None,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "max_seq_length": args.max_seq_length,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": args.target_modules,
        "quantization_bit": args.quantization_bit,
        "attn_implementation": normalized_attn_implementation(args.attn_implementation),
        "resume_from_checkpoint": args.resume_from_checkpoint or None,
    }
    path = Path(args.output_dir) / "training_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = load_train_eval_datasets(args)
    if eval_dataset is None:
        args.eval_strategy = "no"
    if args.limit_train > 0:
        train_dataset = train_dataset.select(range(min(args.limit_train, len(train_dataset))))
    if eval_dataset is not None and args.limit_eval > 0:
        eval_dataset = eval_dataset.select(range(min(args.limit_eval, len(eval_dataset))))
    if args.flatten_messages and args.assistant_only_loss:
        train_dataset = tokenize_assistant_only_dataset(train_dataset, tokenizer, args.max_seq_length)
        if eval_dataset is not None:
            eval_dataset = tokenize_assistant_only_dataset(eval_dataset, tokenizer, args.max_seq_length)
        print("Using manual assistant-only labels with flattened chat messages.", flush=True)
    elif args.flatten_messages:
        train_dataset = flatten_messages_dataset(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = flatten_messages_dataset(eval_dataset, tokenizer)
    write_training_manifest(args, train_dataset, eval_dataset)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_load_kwargs(args),
    )
    model.config.use_cache = False
    if args.quantization_bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

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
        "peft_config": peft_config,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "dataset_text_field": "text",
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer = SFTTrainer(**supported_kwargs(SFTTrainer, trainer_kwargs))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
