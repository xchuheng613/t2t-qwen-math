"""Generate HTML reports for submission CSVs that have no gold answers.

This is the no-gold-standard counterpart to ``visualize_wrong.py``. Given one
or more submission directories produced by ``run_math_prompts.py`` (or
``create_submission.py``), join each row with its question via ``id`` from
``data/public.jsonl`` (and ``data/private.jsonl`` as a fallback) and render
an HTML page that shows:

  - The original question (with options if present), MathJax-rendered.
  - The model's submitted ``response`` text.
  - The extracted boxed answer (best-effort).
  - Audit fields when ``submission.jsonl`` is present (route, sampling
    config, validation errors, repair flags, all sampled responses, ...).
  - A heuristic flag explaining suspicious rows (missing boxed, truncation,
    hidden-think tags, MCQ letter out-of-range, multi-blank count mismatch,
    repair used, ...).

Usage:

    python analysis/visualize_submission.py
    python analysis/visualize_submission.py results/public_hybrid_routed_v2
    python analysis/visualize_submission.py results/private_submission \
                                                  results/public_hybrid_routed_v2

Default behavior (no args): scan every subdirectory of ``results/`` that
contains a ``submission.csv`` and render one HTML per submission plus an
``index.html``.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DATA_FILE = ROOT / "data" / "public.jsonl"
PRIVATE_DATA_FILE = ROOT / "data" / "private.jsonl"
DATA_FILES = [PUBLIC_DATA_FILE, PRIVATE_DATA_FILE]
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "analysis" / "visualizations"

# Accept BOTH the new (`Final answer: \boxed{...}`) and legacy
# (`FINAL_ANSWERS:\n\boxed{...}`) submission conventions.
FINAL_MARKER_RE = re.compile(
    r"(?:Final\s+answers?(?:\s*,\s*in\s+order)?\s*:\s*\\boxed\{"
    r"|FINAL_ANSWERS\s*:\s*\\boxed\{)",
    re.IGNORECASE,
)
THINK_RE = re.compile(r"</?\s*(think|scratchpad)\b", re.IGNORECASE)
LETTER_RE = re.compile(r"^[A-Z]+$")
TOP_LEVEL_BRACKETS = {"(": ")", "[": "]", "{": "}"}


# ════════════════════════════════════════════════════════════════════════════
# Loading
# ════════════════════════════════════════════════════════════════════════════

def infer_data_files(submission_dir: Path, question_set: str = "auto") -> list[Path]:
    """Choose question-bank precedence for a submission directory.

    Public and private files both use zero-based IDs, so the order matters.
    """
    if question_set == "public":
        return [PUBLIC_DATA_FILE, PRIVATE_DATA_FILE]
    if question_set == "private":
        return [PRIVATE_DATA_FILE, PUBLIC_DATA_FILE]

    name = submission_dir.name.lower()
    if "private" in name:
        return [PRIVATE_DATA_FILE, PUBLIC_DATA_FILE]
    return [PUBLIC_DATA_FILE, PRIVATE_DATA_FILE]


def load_questions(data_files: list[Path] | None = None) -> dict[int, dict]:
    qs: dict[int, dict] = {}
    for path in data_files or DATA_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qs.setdefault(int(row["id"]), row)
    return qs


def load_submission_records(submission_dir: Path) -> list[dict[str, Any]]:
    """Prefer the audit JSONL (richer); fall back to the CSV."""
    jsonl = submission_dir / "submission.jsonl"
    if jsonl.exists():
        records: list[dict[str, Any]] = []
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    csv_path = submission_dir / "submission.csv"
    if csv_path.exists():
        records = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    {"id": int(row["id"]), "response": row.get("response", "")}
                )
        return records

    return []


# ════════════════════════════════════════════════════════════════════════════
# Extraction & diagnosis (no gold)
# ════════════════════════════════════════════════════════════════════════════

def extract_boxed(response: str) -> str:
    if not response:
        return ""
    entries: list[str] = []
    start = 0
    needle = "\\boxed{"
    while True:
        idx = response.find(needle, start)
        if idx < 0:
            break
        brace_start = idx + len(needle)
        depth = 1
        i = brace_start
        while i < len(response) and depth > 0:
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            entries.append(response[brace_start : i - 1].strip())
        start = max(i, idx + 1)
    return repair_sqrt_artifacts(entries[-1]) if entries else ""


def repair_sqrt_artifacts(text: str) -> str:
    return str(text).replace("sqrt{(}", "sqrt(")


def split_top_level_commas(expr: str) -> list[str]:
    parts: list[str] = []
    stack: list[str] = []
    start = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "\\" and i + 1 < len(expr):
            i += 2
            continue
        if ch in TOP_LEVEL_BRACKETS:
            stack.append(TOP_LEVEL_BRACKETS[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
        elif ch == "," and not stack:
            parts.append(expr[start:i].strip())
            start = i + 1
        i += 1
    tail = expr[start:].strip()
    if tail:
        parts.append(tail)
    return parts or [expr.strip()]


def n_blanks(question: str) -> int:
    return question.count("[ANS]")


def has_inline_options(question: str) -> bool:
    return ("A." in question or "A)" in question) and (
        "B." in question or "B)" in question
    )


def diagnose(rec: dict[str, Any], q: dict | None) -> tuple[str, str]:
    """Return (category, advice_html) for a submission record (no gold)."""
    response = rec.get("response", "") or ""
    raw = rec.get("raw_response", "") or response
    boxed = extract_boxed(response)
    has_options = bool(q and q.get("options"))
    inline_opts = bool(q and has_inline_options(str(q.get("question", ""))))
    blanks = n_blanks(str(q.get("question", ""))) if q else 0

    finishes = rec.get("finish_reasons") or []
    if rec.get("finish_reason"):
        finishes = [rec["finish_reason"]]
    truncated = any(str(f).lower() == "length" for f in finishes)

    # Hard blockers (the grader can't read the answer): truncation, no box.
    if not boxed:
        return (
            "missing-boxed",
            "No <code>\\boxed{...}</code> found in the response. The grader "
            "cannot extract an answer. Re-run with the submission-repair "
            "prompt, or raise the token budget if the model was still writing "
            "reasoning when it stopped.",
        )

    if truncated:
        return (
            "truncated",
            "Generation hit the token cap. A boxed answer is still present, "
            "but the response was cut off before the model could finish "
            "verifying. Spot-check the extracted value below; consider "
            "raising <code>max_tokens</code> for borderline rows.",
        )

    if has_options or inline_opts:
        candidate = re.sub(r"[^A-Za-z]", "", boxed).upper()
        if not LETTER_RE.match(candidate):
            return (
                "mcq-not-letter",
                f"Multiple-choice question, but the boxed value is "
                f"<code>{html.escape(boxed)}</code> (not an option letter). "
                "Re-prompt with the multiple-choice format profile and ensure "
                "the box contains the LETTER, not the option text.",
            )
        if has_options:
            n_opts = len(q["options"])
            for letter in candidate:
                if ord(letter) - 65 >= n_opts:
                    return (
                        "mcq-out-of-range",
                        f"Boxed letter <b>{letter}</b> is outside the option "
                        f"range (n={n_opts}). The model picked a letter that "
                        "doesn't exist; raise self-consistency and verify the "
                        "options list in the prompt.",
                    )

    if blanks >= 2:
        items = split_top_level_commas(boxed)
        if len(items) != blanks:
            return (
                "blank-count-mismatch",
                f"Question has <b>{blanks}</b> [ANS] blanks but the boxed "
                f"answer contains <b>{len(items)}</b> comma-separated items. "
                "Either the model merged blanks, split a single answer "
                "incorrectly, or the box lost an item to truncation. Prompt "
                "the model to keep <code>,\\ </code> between blanks and to "
                "leave intervals/ordered pairs intact.",
            )

    if rec.get("repair_used"):
        return (
            "repaired",
            "The runner already ran the submission-repair prompt on this "
            "row. The current text is the repaired version; inspect to make "
            "sure the meaning was preserved.",
        )

    if rec.get("fallback_used"):
        return (
            "fallback",
            "The legacy fallback (short no-reasoning prompt) was used because "
            "the primary generation was truncated or had no extractable "
            "answer. The boxed answer below comes from the fallback model "
            "call.",
        )

    if rec.get("validation_error"):
        return (
            "validation-error",
            f"Runner flagged a validation error: "
            f"<code>{html.escape(str(rec['validation_error']))}</code>. "
            "Inspect the response and re-run with repair if needed.",
        )

    # Soft / informational flags — the boxed answer is still there, but the
    # response has cosmetic issues. Reported separately so they don't hide
    # actionable rows.
    if THINK_RE.search(response):
        return (
            "info-think",
            "Response contains <code>&lt;think&gt;</code> / "
            "<code>&lt;scratchpad&gt;</code> tags. The boxed answer is still "
            "extractable, but the CSV cell will include the reasoning text. "
            "Run the submission-repair prompt if you want a cleaner cell.",
        )

    if not FINAL_MARKER_RE.search(response):
        return (
            "info-no-marker",
            "A boxed value exists but the canonical "
            "<code>Final answer: \\boxed{...}</code> / "
            "<code>FINAL_ANSWERS:</code> marker is absent. The grader can "
            "still read the box; the marker is only useful for downstream "
            "parsers.",
        )

    return (
        "clean",
        "No automated red flag. Without a gold answer the correctness of the "
        "boxed value cannot be confirmed — spot-check manually.",
    )


CATEGORY_LABELS = {
    "missing-boxed": "No \\boxed{}",
    "truncated": "Truncated output",
    "mcq-not-letter": "MCQ: boxed isn't a letter",
    "mcq-out-of-range": "MCQ: letter out of range",
    "blank-count-mismatch": "Multi-blank count mismatch",
    "validation-error": "Validation flagged",
    "fallback": "Fallback prompt used",
    "repaired": "Repair pass applied",
    "clean": "No flag",
    "info-think": "Info: <think> tags in response",
    "info-no-marker": "Info: no Final-answer marker",
}

CATEGORY_COLORS = {
    "missing-boxed": "#fde9e9",
    "truncated": "#fff3cd",
    "mcq-not-letter": "#fde9e9",
    "mcq-out-of-range": "#fde9e9",
    "blank-count-mismatch": "#fff3cd",
    "validation-error": "#fff3cd",
    "fallback": "#d1ecf1",
    "repaired": "#d1ecf1",
    "clean": "#e6f4ea",
    "info-think": "#eef0f4",
    "info-no-marker": "#eef0f4",
}

# Categories listed AFTER "clean" are soft/info — they do NOT count toward
# the "flagged" total in the index page.
CATEGORY_ORDER = [
    "missing-boxed", "truncated",
    "mcq-not-letter", "mcq-out-of-range", "blank-count-mismatch",
    "validation-error", "fallback", "repaired",
    "clean",
    "info-think", "info-no-marker",
]
SOFT_CATEGORIES = {"clean", "info-think", "info-no-marker", "repaired", "fallback"}


# ════════════════════════════════════════════════════════════════════════════
# HTML rendering
# ════════════════════════════════════════════════════════════════════════════

CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #222; }
h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
.summary { background: #f4f6f8; padding: 12px 16px; border-radius: 6px; margin-bottom: 24px; }
.case { border: 1px solid #ddd; border-radius: 8px; margin-bottom: 24px;
        padding: 16px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.case h2 { margin: 0 0 8px 0; font-size: 1.05em; color: #1a4d8c; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: .8em;
       margin-right: 6px; }
.tag-mcq { background: #e0f0ff; color: #1a4d8c; }
.tag-free { background: #fff1cc; color: #8c5a00; }
.tag-route { background: #eef0f4; color: #555; }
.question { margin: 8px 0 12px 0; padding: 10px 12px; background: #fafafa; border-left: 3px solid #1a4d8c; }
.options { padding-left: 20px; }
.opt-letter { font-weight: bold; color: #555; margin-right: 4px; }
.answers { display: grid; grid-template-columns: 1fr; gap: 12px; margin: 12px 0; }
.ans-box { padding: 10px 12px; border-radius: 6px; }
.ans-pred { background: #eef6ff; border: 1px solid #b6d2ee; }
.ans-label { font-weight: bold; font-size: .85em; text-transform: uppercase; color: #555; }
details { margin-top: 8px; }
summary { cursor: pointer; color: #1a4d8c; font-weight: 500; user-select: none; }
pre.response { white-space: pre-wrap; word-wrap: break-word; background: #fafafa;
               padding: 12px; border-radius: 4px; max-height: 600px; overflow-y: auto;
               font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 13px; }
.sample { border-top: 1px dashed #ccc; padding-top: 8px; margin-top: 8px; }
.toc a { display: block; padding: 2px 0; }
nav { position: sticky; top: 0; background: white; padding: 8px 0; border-bottom: 1px solid #eee; }
.diag { margin: 12px 0; padding: 12px 14px; border-radius: 6px; border-left: 4px solid #888; }
.diag-label { font-size: .75em; text-transform: uppercase; letter-spacing: .03em; color: #555; font-weight: 700; }
.diag-cat { font-weight: 700; margin: 2px 0 6px 0; font-size: 1.02em; }
.diag-advice { font-size: .95em; line-height: 1.45; }
.diag-advice code { background: rgba(0,0,0,.06); padding: 1px 5px; border-radius: 3px; font-size: .9em; }
.cat-pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78em;
            background: #eee; color: #333; margin-left: 6px; }
.cat-summary { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.cat-summary span { padding: 4px 10px; border-radius: 999px; font-size: .82em; }
.filter { margin: 8px 0 18px 0; }
.filter button { font-size: .82em; padding: 4px 10px; margin: 2px 4px 2px 0;
                 border: 1px solid #ccc; border-radius: 999px; background: #fafafa; cursor: pointer; }
.filter button.active { background: #1a4d8c; color: white; border-color: #1a4d8c; }
.case.hidden { display: none; }
.audit-table { font-size: .85em; border-collapse: collapse; margin-top: 6px; }
.audit-table td { padding: 2px 8px; border-bottom: 1px solid #eee; vertical-align: top; }
.audit-table td.k { color: #666; font-weight: 600; white-space: nowrap; }
"""

MATHJAX = """
<script>
window.MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\\\(','\\\\)']],
         displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
         processEscapes: true },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre'] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
"""

FILTER_JS = """
<script>
function applyFilter(cat) {
  document.querySelectorAll('.filter button').forEach(b => {
    b.classList.toggle('active', b.dataset.cat === cat);
  });
  document.querySelectorAll('section.case').forEach(c => {
    if (cat === 'all' || c.dataset.cat === cat) c.classList.remove('hidden');
    else c.classList.add('hidden');
  });
}
</script>
"""


def render_options(opts: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rows = []
    for i, opt in enumerate(opts):
        letter = letters[i] if i < len(letters) else str(i)
        rows.append(
            f'<li><span class="opt-letter">{letter}.</span> '
            f'<span class="opt-text">{html.escape(str(opt))}</span></li>'
        )
    return '<ol class="options">' + "".join(rows) + "</ol>"


def render_audit_table(rec: dict[str, Any]) -> str:
    keys_to_show = [
        "format_type", "prompt", "config", "routing_mode", "hybrid_source",
        "prompt_module", "chosen_idx", "vote_key", "fallback_used",
        "free_postprocess_used", "repair_used", "valid", "validation_error",
    ]
    rows = []
    for k in keys_to_show:
        if k in rec and rec[k] not in (None, "", False):
            rows.append(
                f'<tr><td class="k">{html.escape(k)}</td>'
                f'<td>{html.escape(str(rec[k]))}</td></tr>'
            )
    if "route" in rec and isinstance(rec["route"], dict):
        for k, v in rec["route"].items():
            if v in (None, ""):
                continue
            rows.append(
                f'<tr><td class="k">route.{html.escape(str(k))}</td>'
                f'<td>{html.escape(str(v))}</td></tr>'
            )
    if "candidate_scores" in rec and rec["candidate_scores"]:
        rows.append(
            f'<tr><td class="k">candidate_scores</td>'
            f'<td>{html.escape(json.dumps(rec["candidate_scores"], ensure_ascii=False))}</td></tr>'
        )
    if not rows:
        return ""
    body = "".join(rows)
    return (
        '<details><summary>Audit fields</summary>'
        f'<table class="audit-table">{body}</table></details>'
    )


def render_case(rec: dict[str, Any], q: dict | None) -> tuple[str, str]:
    rid = rec["id"]
    response = rec.get("response", "") or ""
    raw_response = rec.get("raw_response") or ""
    boxed = extract_boxed(response) or "(no \\boxed{} found)"

    has_options = bool(q and q.get("options"))
    inline_opts = bool(q and has_inline_options(str(q.get("question", ""))))
    is_mcq = has_options or inline_opts

    category, advice = diagnose(rec, q)
    cat_label = CATEGORY_LABELS.get(category, category)
    cat_color = CATEGORY_COLORS.get(category, "#eee")

    tag_kind = (
        f'<span class="tag tag-mcq">multiple choice</span>'
        if is_mcq
        else f'<span class="tag tag-free">free response</span>'
    )
    fmt_type = rec.get("format_type") or (rec.get("route") or {}).get("format_type")
    tag_route = (
        f'<span class="tag tag-route">{html.escape(str(fmt_type))}</span>'
        if fmt_type else ""
    )
    tag_prompt = (
        f'<span class="tag">prompt: {html.escape(str(rec["prompt"]))}</span>'
        if rec.get("prompt") else ""
    )
    tag_config = (
        f'<span class="tag">config: {html.escape(str(rec["config"]))}</span>'
        if rec.get("config") else ""
    )
    tag_repair = (
        '<span class="tag tag-route">repaired</span>'
        if rec.get("repair_used") else ""
    )
    tag_fallback = (
        '<span class="tag tag-route">fallback</span>'
        if rec.get("fallback_used") else ""
    )
    tag_cat = (
        f'<span class="cat-pill" style="background:{cat_color}">'
        f'{html.escape(cat_label)}</span>'
    )

    if q is None:
        q_html = '<em>(question id not found in data/public.jsonl or data/private.jsonl)</em>'
        opts_html = ""
    else:
        q_html = html.escape(q.get("question", ""))
        opts = q.get("options")
        opts_html = render_options(opts) if opts else ""

    response_html = (
        '<details open><summary>Model response</summary>'
        f'<pre class="response">{html.escape(response)}</pre></details>'
    )

    raw_html = ""
    if raw_response and raw_response != response:
        raw_html = (
            '<details><summary>Raw response (pre-cleanup)</summary>'
            f'<pre class="response">{html.escape(raw_response)}</pre></details>'
        )

    samples_html = ""
    samples = rec.get("all_samples") or []
    if samples and len(samples) > 1:
        chunks = []
        chosen_idx = rec.get("chosen_idx")
        for i, s in enumerate(samples):
            mark = " (chosen)" if chosen_idx == i else ""
            chunks.append(
                f'<div class="sample"><b>Sample {i + 1}{mark}:</b>'
                f'<pre class="response">{html.escape(s)}</pre></div>'
            )
        samples_html = (
            f'<details><summary>All sampled responses ({len(samples)})</summary>'
            f'{"".join(chunks)}</details>'
        )

    audit_html = render_audit_table(rec)

    diag_html = (
        f'<div class="diag" style="background:{cat_color}; border-left-color:{cat_color}">'
        f'<div class="diag-label">Heuristic flag</div>'
        f'<div class="diag-cat">{html.escape(cat_label)}</div>'
        f'<div class="diag-advice">{advice}</div>'
        f'</div>'
    )

    section = f"""
<section class="case" id="q{rid}" data-cat="{category}">
  <h2>#{rid} {tag_kind}{tag_route}{tag_prompt}{tag_config}{tag_repair}{tag_fallback}{tag_cat}</h2>
  <div class="question">{q_html}</div>
  {opts_html}
  <div class="answers">
    <div class="ans-box ans-pred">
      <div class="ans-label">Extracted boxed answer</div>
      <div>{html.escape(boxed)}</div>
    </div>
  </div>
  {diag_html}
  {response_html}
  {raw_html}
  {samples_html}
  {audit_html}
</section>
"""
    return section, category


def render_file(name: str, records: list[dict], questions: dict[int, dict]) -> str:
    rendered: list[str] = []
    cat_counts: dict[str, int] = {}
    for r in records:
        section, category = render_case(r, questions.get(int(r["id"])))
        rendered.append(section)
        cat_counts[category] = cat_counts.get(category, 0) + 1

    body_cases = "\n".join(rendered)

    def cat_sort_key(c: str) -> tuple[int, str]:
        try:
            return (CATEGORY_ORDER.index(c), c)
        except ValueError:
            return (len(CATEGORY_ORDER), c)

    cat_pills = "".join(
        f'<span style="background:{CATEGORY_COLORS.get(c, "#eee")}">'
        f'{html.escape(CATEGORY_LABELS.get(c, c))}: <b>{n}</b></span>'
        for c, n in sorted(cat_counts.items(), key=lambda kv: cat_sort_key(kv[0]))
    )

    filter_buttons = ['<button class="active" data-cat="all" onclick="applyFilter(\'all\')">All</button>']
    for c, n in sorted(cat_counts.items(), key=lambda kv: cat_sort_key(kv[0])):
        filter_buttons.append(
            f'<button data-cat="{c}" onclick="applyFilter(\'{c}\')">'
            f'{html.escape(CATEGORY_LABELS.get(c, c))} ({n})</button>'
        )

    toc = "\n".join(
        f'<a href="#q{r["id"]}">#{r["id"]} — {html.escape(extract_boxed(r.get("response", "")) or "(no box)")}</a>'
        for r in records
    )

    n_clean = cat_counts.get("clean", 0)
    n_total = len(records)
    n_flagged = sum(n for c, n in cat_counts.items() if c not in SOFT_CATEGORIES)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Submission view — {html.escape(name)}</title>
  <style>{CSS}</style>
  {MATHJAX}
  {FILTER_JS}
</head>
<body>
  <h1>{html.escape(name)}</h1>
  <div class="summary">
    <div><b>Total rows:</b> {n_total} &middot; <b>Hard-flagged:</b> {n_flagged}
         &middot; <b>No flag:</b> {n_clean} ({100 * n_clean / n_total:.1f}%)</div>
    <div style="margin-top:6px; color:#555; font-size:.9em">
      No gold answers available; flags below are heuristic checks on the
      response text only. Hard flags (truncation, missing boxed marker, MCQ
      letter validity, multi-blank count) are likely to hurt the grader. Soft
      info flags (think tags, missing &quot;Final answer:&quot; marker) are
      cosmetic — the boxed value is still extractable.
    </div>
    <div class="cat-summary">{cat_pills}</div>
  </div>
  <div class="filter">{"".join(filter_buttons)}</div>
  <details><summary>Jump to question</summary><nav class="toc">{toc}</nav></details>
  {body_cases}
</body>
</html>
"""


def render_index(entries: list[tuple[str, dict[str, int], int]]) -> str:
    rows = []
    for name, cat_counts, total in entries:
        flagged = sum(n for c, n in cat_counts.items() if c not in SOFT_CATEGORIES)
        clean = cat_counts.get("clean", 0)
        rows.append(
            f'<tr><td><a href="{html.escape(name)}.html">{html.escape(name)}</a></td>'
            f'<td>{total}</td><td>{clean}</td><td>{flagged}</td>'
            f'<td>{100 * clean / total:.1f}%</td></tr>'
        )
    rows_html = "".join(rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Submission visualizations (no gold)</title>
  <style>{CSS}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #f4f6f8; }}
  </style>
</head>
<body>
  <h1>Submission visualizations (no gold)</h1>
  <table>
    <thead><tr><th>Submission</th><th>Total</th><th>No-flag</th><th>Flagged</th><th>No-flag %</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</body>
</html>
"""


# ════════════════════════════════════════════════════════════════════════════
# Discovery
# ════════════════════════════════════════════════════════════════════════════

def discover_submission_dirs() -> list[Path]:
    """Find every subdirectory of ``results/`` containing a submission.csv."""
    if not RESULTS_DIR.exists():
        return []
    dirs = [
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and (d / "submission.csv").exists()
    ]
    return sorted(dirs, key=lambda p: p.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submissions", nargs="*",
        help="Submission directories (each containing submission.csv "
             "and optionally submission.jsonl). Defaults to every directory "
             "in results/ that has a submission.csv.",
    )
    parser.add_argument(
        "--out-dir", default=str(OUT_DIR),
        help="Where to write HTML files. Default: analysis/visualizations/",
    )
    parser.add_argument(
        "--question-set",
        choices=["auto", "public", "private"],
        default="auto",
        help="Question-bank precedence. Default: auto (private dirs use private.jsonl first).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.submissions:
        sub_dirs = [Path(s) for s in args.submissions]
    else:
        sub_dirs = discover_submission_dirs()

    if not sub_dirs:
        sys.exit("No submission directories found. Pass them on the CLI.")

    entries: list[tuple[str, dict[str, int], int]] = []

    for sub_dir in sub_dirs:
        if not sub_dir.exists():
            print(f"  skip: {sub_dir} (does not exist)", file=sys.stderr)
            continue
        records = load_submission_records(sub_dir)
        if not records:
            print(f"  skip: {sub_dir} (no submission.csv / submission.jsonl)", file=sys.stderr)
            continue
        questions = load_questions(infer_data_files(sub_dir, args.question_set))
        name = f"submission_{sub_dir.name}"
        out_html = out_dir / f"{name}.html"
        out_html.write_text(
            render_file(name, records, questions), encoding="utf-8"
        )
        cat_counts: dict[str, int] = {}
        for r in records:
            c, _ = diagnose(r, questions.get(int(r["id"])))
            cat_counts[c] = cat_counts.get(c, 0) + 1
        entries.append((name, cat_counts, len(records)))
        hard = sum(n for c, n in cat_counts.items() if c not in SOFT_CATEGORIES)
        clean = cat_counts.get("clean", 0)
        info = sum(n for c, n in cat_counts.items() if c.startswith("info-"))
        print(
            f"  {sub_dir.name}: {len(records)} rows  "
            f"hard={hard}  clean={clean}  info={info}  "
            f"-> {out_html.relative_to(ROOT)}"
        )

    if entries:
        index_path = out_dir / "submissions_index.html"
        index_path.write_text(render_index(entries), encoding="utf-8")
        print(f"\nWrote {len(entries)} reports to {out_dir}")
        print(f"Open: {index_path}")


if __name__ == "__main__":
    main()
