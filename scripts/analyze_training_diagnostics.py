#!/usr/bin/env python3
"""Generate LoRA training diagnostics from local checkpoints and benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}
COLORS = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2"]

SIMILAR_DATASETS = [
    {
        "name": "GSM8K",
        "size": "8.5K grade-school math word problems",
        "url": "https://huggingface.co/datasets/openai/gsm8k",
        "why": "Good for concise multi-step arithmetic reasoning and final-answer extraction.",
    },
    {
        "name": "MATH",
        "size": "12.5K competition problems with step-by-step solutions",
        "url": "https://arxiv.org/abs/2103.03874",
        "why": "Useful for harder algebra, counting, number theory, and geometry reasoning.",
    },
    {
        "name": "DeepMind Mathematics Dataset",
        "size": "Generator plus pre-generated modules",
        "url": "https://github.com/google-deepmind/mathematics_dataset",
        "why": "Can synthesize targeted algebra, arithmetic, calculus, comparison, and probability drills.",
    },
    {
        "name": "AQuA-RAT",
        "size": "About 100K algebraic word problems",
        "url": "https://huggingface.co/datasets/deepmind/aqua_rat",
        "why": "Multiple-choice algebra with rationales; useful for MCQ formatting and option selection.",
    },
    {
        "name": "MathQA",
        "size": "29,837 train / 4,475 validation / 2,985 test",
        "url": "https://huggingface.co/datasets/allenai/math_qa",
        "why": "AQuA-derived word problems with options, rationales, and operation programs.",
    },
    {
        "name": "NuminaMath-CoT",
        "size": "About 860K chain-of-thought math problems",
        "url": "https://huggingface.co/datasets/AI-MO/NuminaMath-CoT",
        "why": "Large CoT pool; filter heavily to match the competition answer style.",
    },
    {
        "name": "OpenMathInstruct-2",
        "size": "14M generated problem-solution pairs",
        "url": "https://huggingface.co/datasets/nvidia/OpenMathInstruct-2",
        "why": "Very large synthetic instruction data; best used after strict deduping and format filtering.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/lora_public_v2_B"))
    parser.add_argument("--merged-dir", type=Path, default=Path("checkpoints/merged_lora_public_v2_B"))
    parser.add_argument("--benchmark-summary", type=Path, default=Path("results/lora_bench_v2/benchmark_summary.csv"))
    parser.add_argument("--diff-summary", type=Path, default=Path("analysis/lora_diffs/lora_B_holdout_free_summary.csv"))
    parser.add_argument("--train-file", type=Path, default=Path("data/sft_lora_public_v2/train.jsonl"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/sft_lora_public_v2/dev.jsonl"))
    parser.add_argument("--holdout-file", type=Path, default=Path("data/sft_lora_public_v2/holdout.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/training_diagnostics/lora_public_v2_B"))
    parser.add_argument("--histogram-samples", type=int, default=250_000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8g}"
    return value


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def summarize_jsonl(path: Path, split: str) -> dict[str, Any]:
    total = 0
    mcq_like = 0
    answer_items: list[int] = []
    question_lengths: list[int] = []
    assistant_lengths: list[int] = []
    total_text_lengths: list[int] = []
    if path.exists():
        with path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += 1
                question = question_text(row)
                assistant = assistant_text(row)
                total_text = row_text(row)
                question_lengths.append(len(question))
                assistant_lengths.append(len(assistant))
                total_text_lengths.append(len(total_text))
                values = answer_values(row, assistant)
                answer_items.append(len(values))
                if is_mcq_like(question, values, row.get("options")):
                    mcq_like += 1
    return {
        "split": split,
        "path": str(path),
        "rows": total,
        "mcq_like": mcq_like,
        "free_like": total - mcq_like,
        "avg_question_chars": statistics.mean(question_lengths) if question_lengths else None,
        "avg_assistant_chars": statistics.mean(assistant_lengths) if assistant_lengths else None,
        "avg_total_text_chars": statistics.mean(total_text_lengths) if total_text_lengths else None,
        "avg_answer_items": statistics.mean(answer_items) if answer_items else None,
    }


def question_text(row: dict[str, Any]) -> str:
    if row.get("question") is not None:
        return str(row.get("question", ""))
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return str(row.get("text", ""))


def assistant_text(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages", [])):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def row_text(row: dict[str, Any]) -> str:
    if row.get("text") is not None:
        return str(row.get("text", ""))
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(str(message.get("content", "")) for message in messages)
    return question_text(row)


def answer_values(row: dict[str, Any], assistant: str) -> list[str]:
    answer = row.get("answer")
    if answer is not None:
        values = answer if isinstance(answer, list) else [answer]
        return [str(value).strip() for value in values]
    match = re.search(r"\\boxed\{(.+?)\}", assistant, flags=re.DOTALL)
    if match:
        return [part.strip() for part in match.group(1).split(",") if part.strip()]
    return []


def is_mcq_like(question: str, values: list[Any], options: Any) -> bool:
    if options:
        return True
    answers = [str(value).strip() for value in values]
    all_letters = bool(answers) and all(re.fullmatch(r"[A-Z]+", answer) for answer in answers)
    return all_letters and bool(re.search(r"\bA[\.\)]\s+", question)) and bool(re.search(r"\bB[\.\)]\s+", question))


def load_training_args(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - optional dependency path
        return {"load_error": f"{type(exc).__name__}: {exc}"}
    keys = [
        "output_dir",
        "learning_rate",
        "num_train_epochs",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "max_seq_length",
        "assistant_only_loss",
        "dataset_text_field",
        "packing",
        "bf16",
        "fp16",
        "save_steps",
        "eval_steps",
    ]
    return {key: getattr(obj, key, None) for key in keys}


def trainer_curve_rows(trainer_state: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "step",
        "epoch",
        "loss",
        "eval_loss",
        "mean_token_accuracy",
        "eval_mean_token_accuracy",
        "grad_norm",
        "entropy",
        "eval_entropy",
        "learning_rate",
        "num_tokens",
        "eval_num_tokens",
    ]
    rows: list[dict[str, Any]] = []
    for entry in trainer_state.get("log_history", []):
        rows.append({key: entry.get(key) for key in keys})
    return rows


def series_from_rows(rows: list[dict[str, Any]], y_key: str) -> list[tuple[float, float]]:
    points = []
    for row in rows:
        x = row.get("step")
        y = row.get(y_key)
        if x is not None and y is not None:
            points.append((float(x), float(y)))
    return points


def save_line_chart(
    path: Path,
    series: list[dict[str, Any]],
    title: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.6), constrained_layout=True)
    plotted = False
    all_y: list[float] = []
    for idx, item in enumerate(series):
        points = item.get("points", [])
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        all_y.extend(ys)
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=item.get("color") or COLORS[idx % len(COLORS)],
            label=str(item.get("name", f"series {idx + 1}")),
        )
        plotted = True

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(y_label)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_label == "accuracy":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        if all_y:
            low = max(0.0, min(all_y) - 0.08)
            high = min(1.0, max(all_y) + 0.08)
            if low < high:
                ax.set_ylim(low, high)
    if plotted:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    fig.savefig(path, format="svg")
    plt.close(fig)


def save_bar_chart(path: Path, rows: list[dict[str, Any]], title: str, label_key: str, value_key: str) -> None:
    fig_h = max(3.0, 1.2 + len(rows) * 0.46)
    fig, ax = plt.subplots(figsize=(8.6, fig_h), constrained_layout=True)
    labels = [str(row.get(label_key, "")) for row in rows]
    values = [float(row.get(value_key) or 0.0) for row in rows]
    y_positions = list(range(len(rows)))
    ax.barh(y_positions, values, color=[COLORS[idx % len(COLORS)] for idx in y_positions])
    ax.set_yticks(y_positions, labels=labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    is_percent = value_key == "accuracy" or all(0.0 <= value <= 1.0 for value in values)
    if is_percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(0, max(1.0, max(values or [0.0]) * 1.15))
    else:
        ax.set_xlim(0, max(values or [1.0]) * 1.18 if values else 1.0)
    offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015
    for idx, value in enumerate(values):
        label = f"{value:.1%}" if is_percent else f"{value:,.4g}"
        ax.text(value + offset, idx, label, va="center", fontsize=9)
    fig.savefig(path, format="svg")
    plt.close(fig)


def save_histogram_chart(path: Path, values: list[float], title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.6), constrained_layout=True)
    if values:
        lo = percentile(sorted(values), 1.0)
        hi = percentile(sorted(values), 99.0)
        if lo == hi:
            lo, hi = min(values), max(values)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        ax.hist(
            [value for value in values if lo <= value <= hi],
            bins=60,
            color=COLORS[0],
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_xlabel("sampled adapter weight value")
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "No weights found", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(path, format="svg")
    plt.close(fig)


def summarize_benchmark(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = read_csv(path)
    accuracy_rows: list[dict[str, Any]] = []
    inference_debug_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        name = row.get("run_name", "")
        if "dev" in name:
            split = "validation"
        elif "holdout" in name:
            split = "holdout"
        else:
            split = "other"
        model = "lora" if name.startswith("lora") else "base" if name.startswith("base") else "other"
        accuracy_rows.append(
            {
                "split": split,
                "model": model,
                "run_name": name,
                "total": to_int(row.get("total")),
                "correct": to_int(row.get("correct")),
                "accuracy": to_float(row.get("accuracy")),
                "mcq_accuracy": to_float(row.get("mcq_accuracy")),
                "free_accuracy": to_float(row.get("free_accuracy")),
                "source": row.get("prediction_path", ""),
            }
        )
        inference_debug_rows.append(
            {
                "run_name": name,
                "truncation_count": to_int(row.get("truncation_count")),
                "format_error_count": to_int(row.get("format_error_count")),
                "count_mismatch_count": to_int(row.get("count_mismatch_count")),
                "stage0_repair_used": to_int(row.get("stage0_repair_used")),
                "avg_response_chars": to_float(row.get("avg_response_chars")),
                "avg_raw_response_chars": to_float(row.get("avg_raw_response_chars")),
            }
        )
    return accuracy_rows, inference_debug_rows


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    return None if number is None else int(number)


def summarize_lora_weights(checkpoint_dir: Path, targets: list[str], sample_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    path = checkpoint_dir / "adapter_model.safetensors"
    if not path.exists():
        return [], [], []
    try:
        import torch
        from safetensors.torch import safe_open
    except Exception:
        return [], [], []

    tensor_rows: list[dict[str, Any]] = []
    samples: list[float] = []
    group_acc: dict[str, dict[str, Any]] = defaultdict(lambda: {"params": 0, "tensor_count": 0, "l2_sq": 0.0, "abs_sum": 0.0, "zero_count": 0})
    with safe_open(path, framework="pt", device="cpu") as file:
        keys = list(file.keys())
        per_tensor_samples = max(1, sample_limit // max(len(keys), 1))
        for key in keys:
            tensor = file.get_tensor(key).float()
            numel = tensor.numel()
            module = target_module_from_key(key, targets)
            kind = lora_kind_from_key(key)
            family = family_for_module(module)
            l2_norm = float(torch.linalg.vector_norm(tensor).item())
            abs_sum = float(torch.abs(tensor).sum().item())
            zero_count = int((tensor == 0).sum().item())
            tensor_rows.append(
                {
                    "tensor": key,
                    "module": module,
                    "family": family,
                    "lora_matrix": kind,
                    "shape": "x".join(str(dim) for dim in tensor.shape),
                    "numel": numel,
                    "mean": float(tensor.mean().item()),
                    "std": float(tensor.std(unbiased=False).item()) if numel > 1 else 0.0,
                    "min": float(tensor.min().item()),
                    "max": float(tensor.max().item()),
                    "abs_mean": abs_sum / numel if numel else 0.0,
                    "l2_norm": l2_norm,
                    "zero_fraction": zero_count / numel if numel else 0.0,
                }
            )
            group = group_acc[module]
            group["params"] += numel
            group["tensor_count"] += 1
            group["l2_sq"] += l2_norm * l2_norm
            group["abs_sum"] += abs_sum
            group["zero_count"] += zero_count
            flat = tensor.flatten()
            if numel:
                step = max(1, numel // per_tensor_samples)
                samples.extend(flat[::step][:per_tensor_samples].tolist())

    total_params = sum(group["params"] for group in group_acc.values())
    module_rows: list[dict[str, Any]] = []
    for module in sorted(group_acc):
        group = group_acc[module]
        params = int(group["params"])
        module_rows.append(
            {
                "module": module,
                "family": family_for_module(module),
                "tensor_count": int(group["tensor_count"]),
                "trainable_lora_params": params,
                "percent_of_lora_params": params / total_params if total_params else None,
                "l2_norm": math.sqrt(float(group["l2_sq"])),
                "abs_mean": float(group["abs_sum"]) / params if params else 0.0,
                "zero_fraction": float(group["zero_count"]) / params if params else 0.0,
            }
        )
    return tensor_rows, module_rows, samples[:sample_limit]


def target_module_from_key(key: str, targets: list[str]) -> str:
    for target in targets:
        if f".{target}." in key:
            return target
    return "unknown"


def lora_kind_from_key(key: str) -> str:
    match = re.search(r"\.lora_([AB])\.", key)
    return match.group(1) if match else ""


def family_for_module(module: str) -> str:
    if module in ATTENTION_MODULES:
        return "attention"
    if module in MLP_MODULES:
        return "mlp"
    return "other"


def percentile(values_sorted: list[float], pct: float) -> float:
    if not values_sorted:
        return 0.0
    pos = (len(values_sorted) - 1) * pct / 100.0
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return values_sorted[lower]
    weight = pos - lower
    return values_sorted[lower] * (1 - weight) + values_sorted[upper] * weight


def find_row(rows: list[dict[str, Any]], run_name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("run_name") == run_name:
            return row
    return None


def pct(value: float | None) -> str:
    return "not measured" if value is None else f"{value * 100:.2f}%"


def maybe_delta(new: float | None, old: float | None) -> str:
    if new is None or old is None:
        return ""
    return f"{(new - old) * 100:+.2f} pts"


def write_report(
    path: Path,
    args: argparse.Namespace,
    adapter_config: dict[str, Any],
    model_config: dict[str, Any],
    training_args: dict[str, Any],
    curve_rows: list[dict[str, Any]],
    accuracy_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    module_rows: list[dict[str, Any]],
    diff_rows: list[dict[str, str]],
) -> None:
    train_logs = [row for row in curve_rows if row.get("loss") is not None]
    eval_logs = [row for row in curve_rows if row.get("eval_loss") is not None]
    last_train = train_logs[-1] if train_logs else {}
    last_eval = eval_logs[-1] if eval_logs else {}
    base_dev = find_row(accuracy_rows, "base_dev_post")
    lora_dev = find_row(accuracy_rows, "lora_B_dev_post")
    base_holdout = find_row(accuracy_rows, "base_holdout_free_post_v2fix") or find_row(accuracy_rows, "base_holdout_free_post")
    lora_holdout = find_row(accuracy_rows, "lora_B_holdout_free_post")
    target_modules = adapter_config.get("target_modules", [])
    attention_params = sum(row["trainable_lora_params"] for row in module_rows if row["family"] == "attention")
    mlp_params = sum(row["trainable_lora_params"] for row in module_rows if row["family"] == "mlp")
    total_params = attention_params + mlp_params

    lines = [
        "# LoRA training diagnostics",
        "",
        "## Inputs",
        "",
        f"- Checkpoint: `{args.checkpoint_dir}`",
        f"- Merged model: `{args.merged_dir}`",
        f"- Benchmark summary: `{args.benchmark_summary}`",
        f"- Train/dev/holdout data: `{args.train_file}`, `{args.dev_file}`, `{args.holdout_file}`",
        "",
        "## Training setup",
        "",
        f"- Base model: `{adapter_config.get('base_model_name_or_path', 'unknown')}`",
        f"- Architecture: `{', '.join(model_config.get('architectures', ['unknown']))}`",
        f"- LoRA rank/alpha/dropout: `{adapter_config.get('r')}` / `{adapter_config.get('lora_alpha')}` / `{adapter_config.get('lora_dropout')}`",
        f"- Target modules: `{', '.join(target_modules)}`",
        f"- Trainable adapter params found: `{total_params:,}`",
        f"- Attention LoRA params: `{attention_params:,}`",
        f"- MLP/feed-forward LoRA params: `{mlp_params:,}`",
        "",
        "## Loss function",
        "",
        "The training script uses `SFTTrainer` for causal language modeling, so the objective is next-token cross-entropy over the flattened `text` field.",
        f"In the saved `training_args.bin`, `assistant_only_loss` is `{training_args.get('assistant_only_loss')}`, `packing` is `{training_args.get('packing')}`, and `max_length` is `{training_args.get('max_length')}`.",
        "That means the logged loss is token loss, not exact-answer accuracy.",
        "",
        "## Curves",
        "",
        f"- Last train loss: `{csv_value(last_train.get('loss'))}` at step `{csv_value(last_train.get('step'))}`",
        f"- Last train token accuracy: `{pct(last_train.get('mean_token_accuracy'))}`",
        f"- Eval loss: `{csv_value(last_eval.get('eval_loss'))}` at step `{csv_value(last_eval.get('step'))}`",
        f"- Eval token accuracy: `{pct(last_eval.get('eval_mean_token_accuracy'))}`",
        "",
        "![Training loss](training_loss.svg)",
        "",
        "![Token accuracy](token_accuracy.svg)",
        "",
        "![Debug metrics](debug_metrics.svg)",
        "",
        "## Task accuracy",
        "",
        "| Split | Base | LoRA | Delta | Notes |",
        "|---|---:|---:|---:|---|",
        f"| Training task accuracy | not measured | not measured |  | No generated train-set predictions were found; only token accuracy is logged. |",
        f"| Validation/dev task accuracy | {pct(base_dev.get('accuracy') if base_dev else None)} | {pct(lora_dev.get('accuracy') if lora_dev else None)} | {maybe_delta(lora_dev.get('accuracy') if lora_dev else None, base_dev.get('accuracy') if base_dev else None)} | Public dev split. |",
        f"| Holdout free-response task accuracy | {pct(base_holdout.get('accuracy') if base_holdout else None)} | {pct(lora_holdout.get('accuracy') if lora_holdout else None)} | {maybe_delta(lora_holdout.get('accuracy') if lora_holdout else None, base_holdout.get('accuracy') if base_holdout else None)} | Free-response holdout split. |",
        "",
        "![Task accuracy](accuracy_summary.svg)",
        "",
        "## Weight diagnostics",
        "",
        "![LoRA parameter counts](lora_parameter_counts.svg)",
        "",
        "![LoRA weight norms](lora_weight_norms.svg)",
        "",
        "![LoRA weight histogram](lora_weight_histogram.svg)",
        "",
        "## Why this LoRA probably did not help",
        "",
        f"- The SFT train split has only `{next((row['rows'] for row in dataset_rows if row['split'] == 'train'), 0)}` examples and the run reached only `{csv_value(curve_rows[-1].get('step') if curve_rows else None)}` optimizer steps.",
        f"- Train token accuracy reached `{pct(last_train.get('mean_token_accuracy'))}`, while eval token accuracy was `{pct(last_eval.get('eval_mean_token_accuracy'))}`; that is a clear generalization gap.",
        f"- The LoRA improved free-response dev accuracy from `{pct(base_dev.get('free_accuracy') if base_dev else None)}` to `{pct(lora_dev.get('free_accuracy') if lora_dev else None)}`, but hurt MCQ dev accuracy from `{pct(base_dev.get('mcq_accuracy') if base_dev else None)}` to `{pct(lora_dev.get('mcq_accuracy') if lora_dev else None)}`.",
        f"- Holdout free-response accuracy moved from `{pct(base_holdout.get('accuracy') if base_holdout else None)}` to `{pct(lora_holdout.get('accuracy') if lora_holdout else None)}`, so the small dev free-response gain did not generalize.",
        "- The loss was computed on flattened chat text, so some training signal went into reproducing prompt/template tokens instead of only supervising answer tokens.",
        "- The adapter touches both attention and MLP projections; with little data, that is enough capacity to overfit answer style without improving reasoning.",
        "",
    ]
    if diff_rows:
        lines.extend(["## Holdout diff groups", "", "| Group | Count |", "|---|---:|"])
        for row in diff_rows:
            lines.append(f"| `{row.get('group', '')}` | {row.get('count', '')} |")
        lines.append("")
    lines.extend(
        [
            "## Similar datasets to consider",
            "",
            "Use these as candidates for augmentation, then filter/deduplicate into the exact final-answer format used by this competition.",
            "",
            "| Dataset | Size | Why it helps | Source |",
            "|---|---:|---|---|",
        ]
    )
    for dataset in SIMILAR_DATASETS:
        lines.append(f"| {dataset['name']} | {dataset['size']} | {dataset['why']} | {dataset['url']} |")
    lines.extend(
        [
            "",
            "## Generated files",
            "",
            "- `training_curve.csv`",
            "- `accuracy_summary.csv`",
            "- `inference_debug_metrics.csv`",
            "- `dataset_summary.csv`",
            "- `lora_parameter_summary.csv`",
            "- `lora_weight_stats.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = read_json(args.checkpoint_dir / "adapter_config.json")
    model_config_path = args.merged_dir / "config.json"
    model_config = read_json(model_config_path) if model_config_path.exists() else {}
    trainer_state = read_json(args.checkpoint_dir / "checkpoint-24" / "trainer_state.json")
    training_args = load_training_args(args.checkpoint_dir / "training_args.bin")
    target_modules = list(adapter_config.get("target_modules", []))

    curve_rows = trainer_curve_rows(trainer_state)
    accuracy_rows, inference_debug_rows = summarize_benchmark(args.benchmark_summary)
    dataset_rows = [
        summarize_jsonl(args.train_file, "train"),
        summarize_jsonl(args.dev_file, "dev"),
        summarize_jsonl(args.holdout_file, "holdout"),
    ]
    tensor_rows, module_rows, weight_samples = summarize_lora_weights(
        args.checkpoint_dir,
        target_modules,
        args.histogram_samples,
    )
    diff_rows = read_csv(args.diff_summary)

    write_csv(args.out_dir / "training_curve.csv", curve_rows)
    write_csv(args.out_dir / "accuracy_summary.csv", accuracy_rows)
    write_csv(args.out_dir / "inference_debug_metrics.csv", inference_debug_rows)
    write_csv(args.out_dir / "dataset_summary.csv", dataset_rows)
    write_csv(args.out_dir / "lora_weight_stats.csv", tensor_rows)
    write_csv(args.out_dir / "lora_parameter_summary.csv", module_rows)
    write_csv(args.out_dir / "similar_datasets.csv", SIMILAR_DATASETS)

    save_line_chart(
        args.out_dir / "training_loss.svg",
        [
            {"name": "train loss", "points": series_from_rows(curve_rows, "loss"), "color": COLORS[0]},
            {"name": "eval loss", "points": series_from_rows(curve_rows, "eval_loss"), "color": COLORS[1]},
        ],
        "Training and eval loss",
        "cross-entropy loss",
    )
    save_line_chart(
        args.out_dir / "token_accuracy.svg",
        [
            {"name": "train token accuracy", "points": series_from_rows(curve_rows, "mean_token_accuracy"), "color": COLORS[2]},
            {"name": "eval token accuracy", "points": series_from_rows(curve_rows, "eval_mean_token_accuracy"), "color": COLORS[3]},
        ],
        "Mean token accuracy",
        "accuracy",
    )
    save_line_chart(
        args.out_dir / "debug_metrics.svg",
        [
            {"name": "grad norm", "points": series_from_rows(curve_rows, "grad_norm"), "color": COLORS[4]},
            {"name": "entropy", "points": series_from_rows(curve_rows, "entropy"), "color": COLORS[5]},
        ],
        "Training debug metrics",
        "value",
    )

    chart_acc = [
        {"label": f"{row['split']} {row['model']}", "accuracy": row["accuracy"]}
        for row in accuracy_rows
        if row.get("run_name") in {"base_dev_post", "lora_B_dev_post", "base_holdout_free_post_v2fix", "base_holdout_free_post", "lora_B_holdout_free_post"}
    ]
    save_bar_chart(args.out_dir / "accuracy_summary.svg", chart_acc, "Task accuracy from benchmark runs", "label", "accuracy")
    save_bar_chart(args.out_dir / "lora_parameter_counts.svg", module_rows, "LoRA parameters by target module", "module", "trainable_lora_params")
    save_bar_chart(args.out_dir / "lora_weight_norms.svg", module_rows, "LoRA weight L2 norm by target module", "module", "l2_norm")
    save_histogram_chart(args.out_dir / "lora_weight_histogram.svg", weight_samples, "Sampled LoRA adapter weight distribution")

    write_report(
        args.out_dir / "report.md",
        args,
        adapter_config,
        model_config,
        training_args,
        curve_rows,
        accuracy_rows,
        dataset_rows,
        module_rows,
        diff_rows,
    )
    print(f"wrote {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
