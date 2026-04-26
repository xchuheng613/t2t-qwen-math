"""Sweep prompt + sampling variants on a stratified subset of public.jsonl.

Two-stage pipeline:
  Stage 1 (cheap screen): every variant @ temp=0.0, n=1
  Stage 2 (self-consistency): top-K variants from stage 1 @ temp=0.7, n=3, majority vote

Outputs:
  results/sweep/<variant>_<config>.jsonl    per-question records
  results/sweep/summary.csv                 sorted accuracy table

Run:
  python prompt_sweep.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"
DATA_PATH   = "data/public.jsonl"
OUTPUT_DIR  = Path("results/sweep")
MAX_TOKENS  = 32768
SUBSET_SIZE = 40           # stratified: half MCQ, half free-form (when possible)
RNG_SEED    = 42
TOP_K_FOR_VOTING = 3       # how many stage-1 variants advance to self-consistency
SC_NUM_SAMPLES   = 3       # self-consistency sample count

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

sys.path.insert(0, ".")
from judger import Judger
from prompt_variants import VARIANTS, build_prompt


# ── Sampling configs ───────────────────────────────────────────────────────
@dataclass
class SamplingConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int
    n: int                                  # samples per question
    vote: bool = False                      # if True, majority-vote the n samples


STAGE1 = SamplingConfig(name="greedy_n1", temperature=0.0, top_p=1.0, top_k=-1, n=1)
STAGE2 = SamplingConfig(name="sc_n3",     temperature=0.7, top_p=0.95, top_k=20, n=SC_NUM_SAMPLES, vote=True)


# ── Data loading + stratified subset ───────────────────────────────────────
def load_subset(path: str, k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    data = [json.loads(line) for line in open(path)]
    mcq  = [d for d in data if d.get("options")]
    free = [d for d in data if not d.get("options")]
    rng.shuffle(mcq); rng.shuffle(free)
    half = k // 2
    n_mcq  = min(half, len(mcq))
    n_free = min(k - n_mcq, len(free))
    if n_mcq + n_free < k:
        n_mcq = min(k - n_free, len(mcq))
    subset = mcq[:n_mcq] + free[:n_free]
    rng.shuffle(subset)
    print(f"Subset: {len(subset)} total ({n_mcq} MCQ, {n_free} free-form)")
    return subset


# ── Scoring ────────────────────────────────────────────────────────────────
_LETTER_RE = re.compile(r"\\boxed\{([A-Za-z])\}")

def extract_letter(text: str) -> str:
    m = _LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_one(judger: Judger, item: dict, response: str) -> bool:
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]
    if is_mcq:
        return extract_letter(response) == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(pred=response, gold=gold_list, options=[[]] * len(gold_list))
    except Exception:
        return False


def majority_vote_response(responses: list[str], is_mcq: bool, judger: Judger) -> str:
    """Pick a representative response by majority vote on the extracted answer."""
    if is_mcq:
        keys = [extract_letter(r) for r in responses]
    else:
        keys = []
        for r in responses:
            try:
                k = judger.extract_ans(r)
                k = judger.norm_ans_str(k) if k else ""
            except Exception:
                k = ""
            keys.append(k)
    counts = Counter(k for k in keys if k)
    if not counts:
        return responses[0]
    winning_key = counts.most_common(1)[0][0]
    for k, r in zip(keys, responses):
        if k == winning_key:
            return r
    return responses[0]


# ── Generation ─────────────────────────────────────────────────────────────
def render_prompts(tokenizer, variant: str, items: list[dict]) -> list[str]:
    out = []
    for it in items:
        sys_p, usr_p = build_prompt(variant, it["question"], it.get("options"))
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_p},
             {"role": "user",   "content": usr_p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append(text)
    return out


def run_variant(llm, tokenizer, judger: Judger, variant: str, cfg: SamplingConfig,
                items: list[dict]) -> dict:
    prompts = render_prompts(tokenizer, variant, items)

    sampling = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        n=cfg.n,
    )
    print(f"\n[{variant} / {cfg.name}] generating {len(prompts)} prompts × n={cfg.n} ...")
    outputs = llm.generate(prompts, sampling_params=sampling)

    records = []
    for item, out in zip(items, outputs):
        responses = [o.text.strip() for o in out.outputs]
        if cfg.vote and len(responses) > 1:
            chosen = majority_vote_response(responses, bool(item.get("options")), judger)
        else:
            chosen = responses[0]
        correct = score_one(judger, item, chosen)
        records.append({
            "id":         item.get("id"),
            "is_mcq":     bool(item.get("options")),
            "gold":       item["answer"],
            "response":   chosen,
            "all_samples": responses if cfg.n > 1 else None,
            "correct":    correct,
        })

    n_total = len(records)
    n_corr  = sum(r["correct"] for r in records)
    mcq     = [r for r in records if r["is_mcq"]]
    free    = [r for r in records if not r["is_mcq"]]
    summary = {
        "variant":  variant,
        "config":   cfg.name,
        "n":        n_total,
        "correct":  n_corr,
        "acc":      n_corr / n_total if n_total else 0.0,
        "mcq_acc":  sum(r["correct"] for r in mcq)  / len(mcq)  if mcq  else 0.0,
        "free_acc": sum(r["correct"] for r in free) / len(free) if free else 0.0,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{variant}_{cfg.name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> acc {summary['acc']:.3f} (mcq {summary['mcq_acc']:.3f}, free {summary['free_acc']:.3f}) "
          f"[{out_path}]")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    items = load_subset(DATA_PATH, SUBSET_SIZE, RNG_SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.50,
        max_model_len=16384,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
    )
    judger = Judger(strict_extract=False)

    summaries: list[dict] = []

    print("\n" + "=" * 60)
    print("STAGE 1: cheap screen (greedy, n=1)")
    print("=" * 60)
    for name, *_ in VARIANTS:
        summaries.append(run_variant(llm, tokenizer, judger, name, STAGE1, items))

    stage1_sorted = sorted(summaries, key=lambda s: s["acc"], reverse=True)
    top_variants = [s["variant"] for s in stage1_sorted[:TOP_K_FOR_VOTING]]
    print(f"\nTop-{TOP_K_FOR_VOTING} after stage 1: {top_variants}")

    print("\n" + "=" * 60)
    print(f"STAGE 2: self-consistency (n={SC_NUM_SAMPLES}, majority vote)")
    print("=" * 60)
    for name in top_variants:
        summaries.append(run_variant(llm, tokenizer, judger, name, STAGE2, items))

    summaries.sort(key=lambda s: s["acc"], reverse=True)
    csv_path = OUTPUT_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "config", "n", "correct",
                                                "acc", "mcq_acc", "free_acc"])
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print("\n" + "=" * 60)
    print("FINAL RANKING")
    print("=" * 60)
    print(f"{'variant':<24} {'config':<12} {'acc':>7} {'mcq':>7} {'free':>7}")
    for s in summaries:
        print(f"{s['variant']:<24} {s['config']:<12} {s['acc']:>7.3f} "
              f"{s['mcq_acc']:>7.3f} {s['free_acc']:>7.3f}")
    print(f"\nSummary written to {csv_path}")


if __name__ == "__main__":
    main()
