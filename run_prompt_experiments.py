#!/usr/bin/env python3
"""Run repeatable vLLM prompt experiments for the public math set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_DATA_PATH = "data/public.jsonl"
DEFAULT_OUTPUT_DIR = "results/prompt_experiments"


STARTER_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

STARTER_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

STRICT_MATH = """You are a careful mathematical reasoning model.

Solve the problem step by step. Check your arithmetic and logic before giving the final answer.

Important formatting rule:
At the very end, write the final answer on its own line in exactly this format:
\\boxed{answer}

Do not write anything after the boxed answer."""

STRICT_MCQ = """You are a careful mathematical reasoning model.

Solve the problem step by step. Compare all answer choices if needed.

Important formatting rule:
At the very end, write only the final option letter in exactly this format:
\\boxed{A}

Replace A with the correct option letter. Do not write anything after the boxed answer."""

VERIFY_MATH = """You are a careful mathematical reasoning model.

Solve the problem step by step. Before giving the final answer, verify that your result satisfies the original question. Check for arithmetic mistakes, sign errors, and whether the requested format is a number, expression, set, or option letter.

Important formatting rule:
The final line must contain only:
\\boxed{answer}

Do not write anything after the boxed answer."""

VERIFY_MCQ = """You are a careful mathematical reasoning model.

Solve the problem step by step. Compare all answer choices if needed. Before giving the final answer, verify that the selected option answers the original question.

Important formatting rule:
The final line must contain only the option letter in this format:
\\boxed{A}

Replace A with the correct option letter. Do not write anything after the boxed answer."""

DETAILED_MATH = """You are a careful mathematical reasoning model.

Solve the problem carefully using step-by-step reasoning. For algebra or arithmetic, show intermediate steps. Before giving the final answer, check your work.

Important formatting rule:
The final line must contain only:
\\boxed{answer}

Do not write anything after the boxed answer."""

DETAILED_MCQ = """You are a careful mathematical reasoning model.

Solve the problem carefully using step-by-step reasoning. For multiple-choice questions, eliminate wrong choices when useful.

Important formatting rule:
The final line must contain only the option letter in this format:
\\boxed{A}

Replace A with the correct option letter. Do not write anything after the boxed answer."""

CONCISE_MATH = """You are a careful mathematical reasoning model.

Solve the problem efficiently. Avoid unnecessary explanation. Focus on getting the correct final answer.

Important formatting rule:
The final line must contain only:
\\boxed{answer}

Do not write anything after the boxed answer."""

CONCISE_MCQ = """You are a careful mathematical reasoning model.

Solve the problem efficiently. Avoid unnecessary explanation. Select the single best answer choice.

Important formatting rule:
The final line must contain only the option letter in this format:
\\boxed{A}

Replace A with the correct option letter. Do not write anything after the boxed answer."""

VALIDATE_MATH = """You are a careful mathematical reasoning verifier.

A proposed answer is given below. Check whether it is correct for the problem. If the proposed answer is correct, keep it. If it is incorrect, solve the problem and give the correct answer.

Before giving the final answer, briefly verify that the answer satisfies the original question.

Important formatting rule:
The final line must contain only:
\\boxed{answer}

Do not write anything after the boxed answer."""

VALIDATE_MCQ = """You are a careful mathematical reasoning verifier.

A proposed option letter is given below. Check whether it is correct for the problem. If the proposed option is correct, keep it. If it is incorrect, solve the problem and give the correct option letter.

Before giving the final answer, briefly verify that the selected option satisfies the original question.

Important formatting rule:
The final line must contain only the option letter in this format:
\\boxed{A}

Replace A with the correct option letter. Do not write anything after the boxed answer."""


@dataclass(frozen=True)
class PromptSpec:
    name: str
    math_system: str
    mcq_system: str
    notes: str
    validate_candidate: bool = False
    starter_style_user: bool = False


PROMPTS: dict[str, PromptSpec] = {
    "starter": PromptSpec(
        name="starter",
        math_system=STARTER_MATH,
        mcq_system=STARTER_MCQ,
        notes="Starter notebook prompt.",
        starter_style_user=True,
    ),
    "strict_boxed": PromptSpec(
        name="strict_boxed",
        math_system=STRICT_MATH,
        mcq_system=STRICT_MCQ,
        notes="Separate MCQ/free prompts with strict final boxed answer.",
    ),
    "verify": PromptSpec(
        name="verify",
        math_system=VERIFY_MATH,
        mcq_system=VERIFY_MCQ,
        notes="Strict boxed answer plus final answer verification.",
    ),
    "detailed": PromptSpec(
        name="detailed",
        math_system=DETAILED_MATH,
        mcq_system=DETAILED_MCQ,
        notes="Detailed step-by-step reasoning.",
    ),
    "concise": PromptSpec(
        name="concise",
        math_system=CONCISE_MATH,
        mcq_system=CONCISE_MCQ,
        notes="Concise reasoning.",
    ),
    "validate_random": PromptSpec(
        name="validate_random",
        math_system=VALIDATE_MATH,
        mcq_system=VALIDATE_MCQ,
        notes="Seeded random proposed answer; model validates or corrects it.",
        validate_candidate=True,
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def format_options(options: list[str]) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    return "\n".join(f"{label}. {option.strip()}" for label, option in zip(labels, options))


def random_candidate(item: dict[str, Any], seed: int) -> str:
    item_id = item.get("id", "")
    rng = random.Random(f"{seed}:{item_id}:candidate")
    options = item.get("options")
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        return rng.choice(labels)

    simple_pool = [
        "0",
        "1",
        "-1",
        "2",
        "-2",
        "3",
        "4",
        "5",
        "10",
        "12",
        "\\frac{1}{2}",
        "\\frac{3}{2}",
        "\\sqrt{2}",
        "\\pi",
    ]
    if rng.random() < 0.65:
        return str(rng.randint(-20, 50))
    return rng.choice(simple_pool)


def build_messages(
    item: dict[str, Any],
    spec: PromptSpec,
    seed: int,
) -> tuple[list[dict[str, str]], str | None]:
    question = item["question"]
    options = item.get("options")
    is_mcq = bool(options)
    system = spec.mcq_system if is_mcq else spec.math_system
    candidate = random_candidate(item, seed) if spec.validate_candidate else None

    if options:
        opts_text = format_options(options)
        if spec.starter_style_user:
            user = f"{question}\n\nOptions:\n{opts_text}"
        else:
            user = f"Problem:\n{question}\n\nOptions:\n{opts_text}"
    elif spec.starter_style_user:
        user = question
    else:
        user = f"Problem:\n{question}"

    if candidate is not None:
        user += f"\n\nProposed answer:\n\\boxed{{{candidate}}}"

    return [{"role": "system", "content": system}, {"role": "user", "content": user}], candidate


def extract_boxed_values(text: str) -> list[str]:
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end >= 0 else text
    values: list[str] = []
    start = 0
    while True:
        idx = search_text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        i = brace_start
        while i < len(search_text) and depth > 0:
            if search_text[i] == "{":
                depth += 1
            elif search_text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            values.append(search_text[brace_start : i - 1].strip())
        start = max(i, idx + 1)
    return values


def extract_mcq_prediction(text: str, n_options: int) -> str:
    valid = {chr(65 + i) for i in range(n_options)}
    boxed = extract_boxed_values(text)
    candidates = [boxed[-1]] if boxed else []
    candidates.append(text.split("</think>")[-1] if "</think>" in text else text)

    for candidate in candidates:
        upper = candidate.strip().upper()
        exact = re.fullmatch(r"([A-Z])", upper)
        if exact and exact.group(1) in valid:
            return exact.group(1)
        leading = re.match(r"([A-Z])(?:[\.\):\s]|$)", upper)
        if leading and leading.group(1) in valid:
            return leading.group(1)
        option = re.search(r"\bOPTION\s+([A-Z])\b", upper)
        if option and option.group(1) in valid:
            return option.group(1)

    matches = [m for m in re.findall(r"\b([A-Z])\b", text.upper()) if m in valid]
    return matches[-1] if matches else ""


def make_judger():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from judger import Judger

        return Judger(strict_extract=False)
    except ImportError as exc:
        raise SystemExit(
            "Could not initialize judger dependencies. Run the notebook environment setup "
            "or install requirements.txt in this Python environment."
        ) from exc


def extract_free_prediction(text: str, judger: Any) -> str:
    try:
        return judger.extract_ans(text)
    except Exception:
        return ""


def normalize_vote_key(prediction: str, judger: Any) -> str:
    if not prediction:
        return ""
    try:
        return judger.norm_ans_str(prediction)
    except Exception:
        return prediction


def choose_majority(predictions: list[str], vote_keys: list[str] | None = None) -> str:
    if vote_keys is None:
        vote_keys = predictions
    pairs = [
        (prediction, key)
        for prediction, key in zip(predictions, vote_keys)
        if prediction and key
    ]
    if not pairs:
        return ""
    counts = Counter(key for _prediction, key in pairs)
    top_count = counts.most_common(1)[0][1]
    tied = {key for key, count in counts.items() if count == top_count}
    for prediction, key in pairs:
        if key in tied:
            return prediction
    return pairs[0][0]


def score_prediction(item: dict[str, Any], prediction: str, judger: Any) -> bool:
    if not prediction:
        return False

    options = item.get("options")
    gold = item["answer"]
    if options:
        return prediction.strip().upper() == str(gold).strip().upper()

    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return bool(
            judger.auto_judge(
                pred=f"\\boxed{{{prediction}}}",
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        )
    except Exception:
        return False


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    mcq = [result for result in results if result["is_mcq"]]
    free = [result for result in results if not result["is_mcq"]]

    def acc(subset: list[dict[str, Any]]) -> float:
        return 100.0 * sum(bool(row["correct"]) for row in subset) / len(subset) if subset else 0.0

    return {
        "num_problems": len(results),
        "overall_correct": sum(bool(row["correct"]) for row in results),
        "overall_accuracy": acc(results),
        "mcq_count": len(mcq),
        "mcq_correct": sum(bool(row["correct"]) for row in mcq),
        "mcq_accuracy": acc(mcq),
        "free_count": len(free),
        "free_correct": sum(bool(row["correct"]) for row in free),
        "free_accuracy": acc(free),
    }


def rough_error_type(row: dict[str, Any]) -> str:
    if not row.get("chosen_prediction"):
        return "Extraction issue"
    if row["is_mcq"]:
        if len(str(row["chosen_prediction"])) != 1:
            return "Format error"
        return "MCQ mismatch"
    if not any("\\boxed{" in response for response in row.get("responses", [])):
        return "Format error"
    return "Unclassified"


def write_error_analysis(results: list[dict[str, Any]], path: Path, limit: int = 20) -> None:
    wrong = [row for row in results if not row["correct"]][:limit]
    fields = [
        "id",
        "is_mcq",
        "gold",
        "chosen_prediction",
        "suggested_error_type",
        "manual_error_type",
        "notes",
        "response_preview",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in wrong:
            responses = row.get("responses", [])
            preview = responses[0].replace("\n", " ")[:500] if responses else ""
            writer.writerow(
                {
                    "id": row["id"],
                    "is_mcq": row["is_mcq"],
                    "gold": json.dumps(row["gold"]),
                    "chosen_prediction": row.get("chosen_prediction", ""),
                    "suggested_error_type": rough_error_type(row),
                    "manual_error_type": "",
                    "notes": "",
                    "response_preview": preview,
                }
            )


def append_summary_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "timestamp",
        "run_name",
        "experiment",
        "num_problems",
        "samples",
        "mcq_samples",
        "free_samples",
        "temperature",
        "top_p",
        "top_k",
        "overall_correct",
        "overall_accuracy",
        "mcq_correct",
        "mcq_count",
        "mcq_accuracy",
        "free_correct",
        "free_count",
        "free_accuracy",
        "notes",
    ]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def make_sampling_params(args: argparse.Namespace, samples: int):
    from vllm import SamplingParams

    return SamplingParams(
        n=samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=args.repetition_penalty,
    )


def load_vllm(args: argparse.Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
        "seed": args.seed,
    }
    if args.quantization != "none":
        llm_kwargs["quantization"] = args.quantization
    if args.load_format != "auto" and args.quantization != "none":
        llm_kwargs["load_format"] = args.load_format
    if args.dtype != "auto":
        llm_kwargs["dtype"] = args.dtype

    llm = LLM(**llm_kwargs)
    return tokenizer, llm


def effective_sample_count(item: dict[str, Any], args: argparse.Namespace) -> int:
    if item.get("options"):
        return args.mcq_samples if args.mcq_samples is not None else args.samples
    return args.free_samples if args.free_samples is not None else args.samples


def run_experiment(
    experiment: str,
    items: list[dict[str, Any]],
    tokenizer: Any,
    llm: Any,
    judger: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    spec = PROMPTS[experiment]
    built = [build_messages(item, spec, args.seed) for item in items]
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages, _candidate in built
    ]
    candidates = [candidate for _messages, candidate in built]
    sample_counts = [effective_sample_count(item, args) for item in items]
    responses_by_index: list[list[str]] = [[] for _ in items]

    for sample_count in sorted(set(sample_counts)):
        indices = [idx for idx, count in enumerate(sample_counts) if count == sample_count]
        group_prompts = [prompts[idx] for idx in indices]
        sampling_params = make_sampling_params(args, sample_count)
        print(f"Generating {len(group_prompts)} prompts for {experiment} with n={sample_count}...")
        outputs = llm.generate(group_prompts, sampling_params=sampling_params)
        for item_idx, output in zip(indices, outputs):
            responses_by_index[item_idx] = [choice.text.strip() for choice in output.outputs]

    results: list[dict[str, Any]] = []
    for item, candidate, responses in zip(items, candidates, responses_by_index):
        if item.get("options"):
            predictions = [extract_mcq_prediction(response, len(item["options"])) for response in responses]
            vote_keys = predictions
        else:
            predictions = [extract_free_prediction(response, judger) for response in responses]
            vote_keys = [normalize_vote_key(prediction, judger) for prediction in predictions]

        chosen = choose_majority(predictions, vote_keys)
        correct = score_prediction(item, chosen, judger)
        results.append(
            {
                "id": item.get("id"),
                "is_mcq": bool(item.get("options")),
                "gold": item.get("answer"),
                "candidate_answer": candidate,
                "responses": responses,
                "sample_predictions": predictions,
                "chosen_prediction": chosen,
                "correct": correct,
            }
        )
    return results


def select_items(data: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.num_examples < 0:
        selected = data
    else:
        selected = data[: args.num_examples]

    if args.shuffle:
        rng = random.Random(args.seed)
        selected = selected[:]
        rng.shuffle(selected)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--experiments", nargs="+", default=["starter"])
    parser.add_argument("--num-examples", type=int, default=100, help="Use -1 for the full dataset.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before taking num examples.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--mcq-samples", type=int, default=None)
    parser.add_argument("--free-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--quantization", default="bitsandbytes", choices=["bitsandbytes", "none"])
    parser.add_argument("--load-format", default="bitsandbytes")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print sample prompts without loading the model.")
    return parser.parse_args()


def normalize_experiments(raw: list[str]) -> list[str]:
    if raw == ["all"]:
        return ["starter", "strict_boxed", "verify", "concise", "detailed", "validate_random"]
    unknown = [name for name in raw if name not in PROMPTS]
    if unknown:
        valid = ", ".join(sorted(PROMPTS))
        raise SystemExit(f"Unknown experiment(s): {', '.join(unknown)}. Valid: {valid}, all")
    return raw


def main() -> None:
    args = parse_args()
    experiments = normalize_experiments(args.experiments)
    data = load_jsonl(Path(args.data_path))
    items = select_items(data, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(data)} total problems; evaluating {len(items)}.")
    print(f"Experiments: {', '.join(experiments)}")

    if args.dry_run:
        for experiment in experiments:
            spec = PROMPTS[experiment]
            messages, candidate = build_messages(items[0], spec, args.seed)
            print(f"\n=== {experiment} ===")
            if candidate is not None:
                print(f"candidate_answer: {candidate}")
            for message in messages:
                print(f"\n[{message['role']}]\n{message['content'][:2000]}")
        return

    judger = make_judger()
    tokenizer, llm = load_vllm(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows: list[dict[str, Any]] = []

    for experiment in experiments:
        results = run_experiment(experiment, items, tokenizer, llm, judger, args)
        summary = summarize(results)
        samples = args.samples
        mcq_samples = args.mcq_samples if args.mcq_samples is not None else args.samples
        free_samples = args.free_samples if args.free_samples is not None else args.samples
        run_name = (
            f"{timestamp}_{experiment}_n{summary['num_problems']}"
            f"_mcq{mcq_samples}x_free{free_samples}x_t{args.temperature:g}"
        )

        result_path = output_dir / f"{run_name}.jsonl"
        with result_path.open("w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

        error_path = output_dir / f"{run_name}_wrong20.csv"
        write_error_analysis(results, error_path)

        summary_row = {
            "timestamp": timestamp,
            "run_name": run_name,
            "experiment": experiment,
            "samples": samples,
            "mcq_samples": mcq_samples,
            "free_samples": free_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "notes": PROMPTS[experiment].notes,
            **summary,
        }
        append_summary_csv(output_dir / "summary.csv", summary_row)
        summary_rows.append(summary_row)

        print(
            f"{experiment}: {summary['overall_correct']} / {summary['num_problems']} "
            f"({summary['overall_accuracy']:.2f}%) "
            f"| MCQ {summary['mcq_correct']} / {summary['mcq_count']} "
            f"({summary['mcq_accuracy']:.2f}%) "
            f"| Free {summary['free_correct']} / {summary['free_count']} "
            f"({summary['free_accuracy']:.2f}%)"
        )
        print(f"  results: {result_path}")
        print(f"  wrong-example sheet: {error_path}")

    print("\nSummary")
    print("Run\tProblems\tAccuracy\tNotes")
    for row in summary_rows:
        print(
            f"{row['experiment']}\t{row['num_problems']}\t"
            f"{row['overall_accuracy']:.2f}%\t{row['notes']}"
        )


if __name__ == "__main__":
    main()
