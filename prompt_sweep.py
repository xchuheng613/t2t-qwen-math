"""Independent prompt sweep on a stratified subset of public.jsonl.

Pipeline:
  Single stage: every MCQ prompt on MCQ items, every FREE prompt on FREE
  items, all with self-consistency n=3 (majority vote) on a 100/100 subset.

Outputs:
  results/sweep/<type>__<prompt>__<config>.jsonl    per-question records
  results/sweep/summary.csv                          ranked accuracy table

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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"
DATA_PATH   = "data/public.jsonl"
OUTPUT_DIR  = Path("results/sweep")
MAX_TOKENS  = 32768
SUBSET_SIZE = 200          # stratified: ~100 MCQ + ~100 free-form
RNG_SEED    = 42
SC_NUM_SAMPLES   = 3       # self-consistency sample count (every prompt)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

sys.path.insert(0, ".")
from judger import Judger
from prompt_variants import MCQ_PROMPTS, FREE_PROMPTS, build_mcq_prompt, build_free_prompt


# ── Sampling configs ───────────────────────────────────────────────────────
@dataclass
class SamplingConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int
    n: int
    vote: bool = False


SWEEP_CFG = SamplingConfig(name=f"sc_n{SC_NUM_SAMPLES}",
                           temperature=0.7, top_p=0.95, top_k=20,
                           n=SC_NUM_SAMPLES, vote=True)


# ── Data loading + stratified subset ───────────────────────────────────────
def load_subset(path: str, k: int, seed: int) -> tuple[list[dict], list[dict]]:
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
    mcq_subset  = mcq[:n_mcq]
    free_subset = free[:n_free]
    print(f"Subset: {n_mcq} MCQ + {n_free} free-form = {n_mcq + n_free} total")
    return mcq_subset, free_subset


# ── Answer extraction ──────────────────────────────────────────────────────
_LETTER_RE        = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")
_LETTER_PHRASE_RE = re.compile(
    r"(?:option|choice|answer\s+is)\s*[:\s]*\(?([A-Z])\)?\b", re.IGNORECASE
)


def extract_letter(text: str, options: Optional[list], judger: Judger) -> str:
    """Robust MCQ letter extractor with multiple fallbacks."""
    think_end = text.rfind("</think>")
    tail = text[think_end + len("</think>"):] if think_end >= 0 else text

    m = _LETTER_RE.search(tail) or _LETTER_RE.search(text)
    if m:
        return m.group(1).upper()

    if options:
        try:
            boxed_contents = judger.extract_all_boxed(tail) or judger.extract_all_boxed(text)
        except Exception:
            boxed_contents = []
        if boxed_contents:
            cand = boxed_contents[-1]
            try:
                cand_norm = judger.norm_ans_str(cand)
            except Exception:
                cand_norm = cand
            for i, opt in enumerate(options):
                opt_str = str(opt).strip()
                if cand.strip() == opt_str or cand_norm == opt_str:
                    return chr(65 + i)
                try:
                    if judger.is_equal(cand_norm, judger.norm_ans_str(opt_str)):
                        return chr(65 + i)
                except Exception:
                    pass

    pm = list(_LETTER_PHRASE_RE.finditer(tail))
    if pm:
        return pm[-1].group(1).upper()

    matches = re.findall(r"\b([A-Z])\b", tail.upper())
    return matches[-1] if matches else ""


def score_one(judger: Judger, item: dict, response: str) -> bool:
    is_mcq = bool(item.get("options"))
    gold   = item["answer"]
    if is_mcq:
        return extract_letter(response, item.get("options"), judger) == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(pred=response, gold=gold_list, options=[[]] * len(gold_list))
    except Exception:
        return False


# Error taxonomy. Mutually exclusive; aggregated per (type, prompt, config)
# so we can see WHY a prompt loses accuracy, not just THAT it does.
ERROR_TYPES = ["correct", "truncated", "no_answer", "out_of_range", "wrong", "judge_error"]


def categorize(judger: Judger, item: dict, response: str,
               correct: bool, finish_reason: str | None) -> tuple[str, str]:
    """Return (error_type, extracted_answer_str). Order of checks matters."""
    if correct:
        return "correct", ""
    is_mcq = bool(item.get("options"))
    # Truncation: trust vLLM's finish_reason. The "</think>" heuristic
    # over-counts when the model legitimately skips the thinking block.
    if finish_reason == "length":
        return "truncated", ""
    if is_mcq:
        try:
            extracted = extract_letter(response, item.get("options"), judger)
        except Exception:
            return "judge_error", ""
        if not extracted:
            return "no_answer", ""
        n_opts = len(item.get("options") or [])
        if n_opts and (ord(extracted) - 65) >= n_opts:
            return "out_of_range", extracted
        return "wrong", extracted
    # Free-form
    try:
        ans = judger.extract_ans(response) or ""
    except Exception:
        return "judge_error", ""
    if not ans:
        return "no_answer", ""
    return "wrong", str(ans)


def majority_vote(responses: list[str], item: dict, judger: Judger) -> tuple[str, int]:
    """Return (chosen_text, chosen_idx) so caller can fetch its finish_reason."""
    is_mcq = bool(item.get("options"))
    if is_mcq:
        keys = [extract_letter(r, item.get("options"), judger) for r in responses]
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
        return responses[0], 0
    winning = counts.most_common(1)[0][0]
    for i, (k, r) in enumerate(zip(keys, responses)):
        if k == winning:
            return r, i
    return responses[0], 0


# ── Generation ─────────────────────────────────────────────────────────────
def render_prompts(tokenizer, qtype: str, prompt_name: str, items: list[dict]) -> list[str]:
    out = []
    for it in items:
        if qtype == "mcq":
            sys_p, usr_p = build_mcq_prompt(prompt_name, it["question"], it["options"])
        else:
            sys_p, usr_p = build_free_prompt(prompt_name, it["question"])
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_p},
             {"role": "user",   "content": usr_p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append(text)
    return out


def run_one(llm, tokenizer, judger: Judger, qtype: str, prompt_name: str,
            cfg: SamplingConfig, items: list[dict]) -> dict:
    if not items:
        return {"type": qtype, "prompt": prompt_name, "config": cfg.name,
                "n": 0, "correct": 0, "acc": 0.0}

    prompts = render_prompts(tokenizer, qtype, prompt_name, items)
    sampling = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        n=cfg.n,
    )
    print(f"\n[{qtype} / {prompt_name} / {cfg.name}] generating {len(prompts)} prompts × n={cfg.n} ...")
    outputs = llm.generate(prompts, sampling_params=sampling)

    records = []
    for item, out in zip(items, outputs):
        responses = [o.text.strip() for o in out.outputs]
        finishes  = [getattr(o, "finish_reason", None) for o in out.outputs]
        if cfg.vote and len(responses) > 1:
            chosen, idx = majority_vote(responses, item, judger)
        else:
            chosen, idx = responses[0], 0
        finish_reason = finishes[idx]
        is_correct = score_one(judger, item, chosen)
        err_type, extracted = categorize(judger, item, chosen, is_correct, finish_reason)
        records.append({
            "id":            item.get("id"),
            "is_mcq":        qtype == "mcq",
            "gold":          item["answer"],
            "response":      chosen,
            "all_samples":   responses if cfg.n > 1 else None,
            "finish_reason": finish_reason,
            "extracted":     extracted,
            "correct":       is_correct,
            "error_type":    err_type,
        })

    n_total = len(records)
    n_corr  = sum(r["correct"] for r in records)
    err_counts = Counter(r["error_type"] for r in records)
    summary = {
        "type":    qtype,
        "prompt":  prompt_name,
        "config":  cfg.name,
        "n":       n_total,
        "correct": n_corr,
        "acc":     n_corr / n_total if n_total else 0.0,
    }
    for et in ERROR_TYPES:
        summary[f"err_{et}"] = err_counts.get(et, 0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{qtype}__{prompt_name}__{cfg.name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    err_str = "  ".join(f"{et}={err_counts.get(et, 0)}" for et in ERROR_TYPES if et != "correct")
    print(f"  -> acc {summary['acc']:.3f}  ({n_corr}/{n_total})  [{out_path.name}]")
    print(f"     errors: {err_str}")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────
def write_summary(summaries: list[dict]) -> Path:
    csv_path = OUTPUT_DIR / "summary.csv"
    fields = ["type", "prompt", "config", "n", "correct", "acc"] + [f"err_{et}" for et in ERROR_TYPES]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sorted(summaries, key=lambda x: (x["type"], -x["acc"], x["config"])):
            w.writerow(s)
    return csv_path


def print_table(title: str, rows: list[dict]):
    print(f"\n── {title} ──")
    hdr = f"{'type':<5} {'prompt':<22} {'config':<12} {'acc':>7} {'n':>4} " \
          f"{'trunc':>6} {'noans':>6} {'oor':>5} {'wrong':>6} {'judge':>6}"
    print(hdr)
    for s in sorted(rows, key=lambda x: -x["acc"]):
        print(f"{s['type']:<5} {s['prompt']:<22} {s['config']:<12} {s['acc']:>7.3f} {s['n']:>4} "
              f"{s.get('err_truncated', 0):>6} {s.get('err_no_answer', 0):>6} "
              f"{s.get('err_out_of_range', 0):>5} {s.get('err_wrong', 0):>6} "
              f"{s.get('err_judge_error', 0):>6}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mcq_items, free_items = load_subset(DATA_PATH, SUBSET_SIZE, RNG_SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.50,
        max_model_len=65536,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
    )
    judger = Judger(strict_extract=False)

    summaries: list[dict] = []

    # ── MCQ prompts on MCQ items (sc_n3) ──
    print("\n" + "=" * 60)
    print(f"MCQ prompts on MCQ items (sc_n{SC_NUM_SAMPLES}, majority vote)")
    print("=" * 60)
    mcq_results = []
    for name, *_ in MCQ_PROMPTS:
        s = run_one(llm, tokenizer, judger, "mcq", name, SWEEP_CFG, mcq_items)
        mcq_results.append(s); summaries.append(s)
    print_table("MCQ ranking", mcq_results)

    # ── FREE prompts on FREE items (sc_n3) ──
    print("\n" + "=" * 60)
    print(f"FREE prompts on FREE items (sc_n{SC_NUM_SAMPLES}, majority vote)")
    print("=" * 60)
    free_results = []
    for name, *_ in FREE_PROMPTS:
        s = run_one(llm, tokenizer, judger, "free", name, SWEEP_CFG, free_items)
        free_results.append(s); summaries.append(s)
    print_table("FREE ranking", free_results)

    csv_path = write_summary(summaries)
    print_table("FINAL RANKING", summaries)

    # Combined best — projected accuracy if you used best_mcq for all MCQs and best_free for all free
    def best_for(qtype: str) -> dict:
        rows = [s for s in summaries if s["type"] == qtype]
        return max(rows, key=lambda s: s["acc"]) if rows else {"acc": 0.0, "n": 0, "correct": 0,
                                                                "prompt": "-", "config": "-"}
    bm, bf = best_for("mcq"), best_for("free")
    total_n = bm["n"] + bf["n"]
    total_c = bm["correct"] + bf["correct"]
    print("\n" + "=" * 60)
    print("BEST COMBINED CONFIG")
    print("=" * 60)
    print(f"  MCQ : prompt='{bm['prompt']}'  config={bm['config']}  acc={bm['acc']:.3f}  ({bm['correct']}/{bm['n']})")
    print(f"  FREE: prompt='{bf['prompt']}'  config={bf['config']}  acc={bf['acc']:.3f}  ({bf['correct']}/{bf['n']})")
    if total_n:
        print(f"  Combined acc on subset: {total_c}/{total_n} = {total_c/total_n:.3f}")
    print(f"\nSummary written to {csv_path}")


if __name__ == "__main__":
    main()
