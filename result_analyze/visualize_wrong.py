"""Generate HTML reports for wrong answers in result_analyze/*.jsonl.

Joins each wrong record (correct == false) with the original question from
data/public.jsonl by id, then writes one HTML file per result file plus an
index. LaTeX in questions and responses is rendered via MathJax.

Also includes public format-router outputs from results/public_format_router/.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "public.jsonl"
RESULT_DIR = ROOT / "result_analyze"
OUT_DIR = RESULT_DIR / "visualizations"
EXTRA_RESULT_DIRS = [
    ROOT / "results" / "public_format_router",
]

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
ANS_RE = re.compile(r"\[ANS\]\s*([^\n]+)")
NUMBER_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")


def _is_number(s: str) -> bool:
    return bool(NUMBER_RE.match(s.strip()))


def _normalize_str(s: str) -> str:
    """Loose normalization for equivalence comparisons."""
    if s is None:
        return ""
    s = s.strip().lower()
    # strip common LaTeX wrappers
    s = re.sub(r"\\(left|right|displaystyle|mathrm|text)\b", "", s)
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\frac", "frac").replace("\\sqrt", "sqrt")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\infty", "infinity").replace("\\inf", "infinity").replace("∞", "infinity")
    s = s.replace("^{", "^(").replace("e^{", "exp(").replace("{", "(").replace("}", ")")
    s = s.replace("$", "").replace("\\$", "").replace("\\(", "").replace("\\)", "")
    s = re.sub(r"\s+", "", s)
    return s


def _split_multi(s: str) -> list[str]:
    if s is None:
        return []
    parts = re.split(r"[,;]", s)
    return [p.strip() for p in parts if p.strip()]


def _numbers_close(a: str, b: str, rel: float = 5e-3) -> bool:
    try:
        x, y = float(a), float(b)
    except (ValueError, TypeError):
        return False
    if x == y:
        return True
    if y == 0:
        return abs(x) < rel
    return abs(x - y) / max(abs(x), abs(y)) < rel


def diagnose(rec: dict, q: dict | None) -> tuple[str, str]:
    """Return (category, advice_html) for a wrong record.

    Categories:
      - truncated: model ran out of tokens
      - empty-extracted: predicted field blank (likely extraction bug)
      - rounding: numeric answers with precision-only mismatch
      - notation: same value, different LaTeX/ASCII formatting
      - case-only: word answer differs by case
      - multi-blank-misaligned: multi-answer ordering or split mismatch
      - mcq-letter-disagree: MCQ where the model committed to a letter
      - real-error: nothing else matched (likely a real reasoning error)
    """
    error_type = rec.get("error_type", "")
    finish_reason = rec.get("finish_reason", "")
    is_mcq = rec.get("is_mcq", False)
    gold = rec.get("gold")
    pred = extract_predicted(rec)
    pred_stripped = pred.strip()

    if error_type == "truncated" or finish_reason == "length":
        return (
            "truncated",
            "Model hit the token cap before emitting <code>\\boxed{}</code>. "
            "Try stop sequences on <code>\\boxed{</code>, instruct the model not to "
            "re-verify, or raise <code>max_new_tokens</code>. Answer-first prompting "
            "(letter first, justification after) also helps.",
        )

    if not pred_stripped or pred_stripped == "(none)":
        return (
            "empty-extracted",
            "No <code>extracted</code> field and no <code>\\boxed{}</code>/[ANS] in "
            "the response. Producer pipeline should backfill <code>extracted</code> "
            "for this variant; tighten the regex to also accept "
            "<code>final answer:</code> patterns.",
        )

    if is_mcq:
        gold_letter = (gold or "").strip().upper() if isinstance(gold, str) else ""
        # Try to find the model's chosen letter from the response.
        boxed = BOXED_RE.findall(rec.get("response", ""))
        chosen = (boxed[-1] if boxed else pred_stripped).strip().upper()
        chosen = re.sub(r"[^A-Z]", "", chosen)[:1]
        if gold_letter and chosen and chosen != gold_letter:
            return (
                "mcq-letter-disagree",
                f"Model committed to <b>{chosen}</b>, gold is <b>{gold_letter}</b>. "
                "Bump self-consistency to n=8–16 with majority vote (n=3 is too few "
                "to escape correlated mistakes), and try a verifier pass that argues "
                "against the chosen letter before re-voting. For OEIS/algorithm-"
                "definition questions, give the model code execution.",
            )
        return (
            "real-error",
            "Hard reasoning failure on an MCQ. Try a verifier prompt and "
            "stronger self-consistency.",
        )

    # Free-form analysis.
    gold_list = gold if isinstance(gold, list) else [str(gold)]
    pred_list = _split_multi(pred_stripped)

    if len(gold_list) > 1 or len(pred_list) > 1:
        # Per-blank comparison
        if len(gold_list) == len(pred_list):
            mismatches = []
            for i, (g, p) in enumerate(zip(gold_list, pred_list)):
                gs, ps = str(g).strip(), str(p).strip()
                if gs == ps:
                    continue
                if _is_number(gs) and _is_number(ps) and _numbers_close(gs, ps):
                    mismatches.append(("rounding", i, gs, ps))
                elif _normalize_str(gs) == _normalize_str(ps):
                    mismatches.append(("notation", i, gs, ps))
                elif gs.lower() == ps.lower():
                    mismatches.append(("case", i, gs, ps))
                else:
                    mismatches.append(("real", i, gs, ps))
            kinds = {m[0] for m in mismatches}
            if kinds and kinds.issubset({"rounding"}):
                return (
                    "rounding",
                    "Per-blank values match within tolerance; failure is precision-"
                    "only. Instruct the model to emit the exact closed form "
                    "(fraction / log / exp) in <code>\\boxed{}</code>, and add "
                    "numeric tolerance to the judger.",
                )
            if kinds and kinds.issubset({"rounding", "notation", "case"}):
                return (
                    "notation",
                    "Per-blank values are equivalent up to formatting (notation/"
                    "case/precision). Harden the judger: SymPy "
                    "<code>simplify(a-b)==0</code>, normalize "
                    "<code>\\frac</code>/<code>^{}</code>/<code>\\infty</code>, and "
                    "compare case-insensitively for word answers.",
                )
            return (
                "real-error",
                "At least one blank is genuinely different from the gold value. "
                "Inspect the response for the failing blank.",
            )
        else:
            return (
                "multi-blank-misaligned",
                "Different number of answers than gold expects "
                f"(gold={len(gold_list)}, pred={len(pred_list)}). Either the model "
                "merged / split blanks, or the extraction split on the wrong "
                "delimiter. Prompt for one <code>\\boxed{}</code> per blank in "
                "question order, and split judger inputs on <code>[ANS]</code> "
                "boundaries rather than commas.",
            )

    # Single-answer free-form
    g0, p0 = str(gold_list[0]).strip(), pred_stripped
    if _is_number(g0) and _is_number(p0):
        if _numbers_close(g0, p0):
            return (
                "rounding",
                "Numeric answers agree within tolerance — judger is doing exact "
                "string compare. Add numeric tolerance, or prompt the model to "
                "match the gold's significant figures (read the question's "
                "rounding hint).",
            )
        return (
            "real-error",
            "Numeric values genuinely differ. Likely a real arithmetic / setup "
            "error in the model's reasoning.",
        )

    if _normalize_str(g0) == _normalize_str(p0):
        return (
            "notation",
            "Same value, different formatting. Harden the judger: strip LaTeX "
            "wrappers, normalize <code>\\frac</code>/<code>^{}</code>/"
            "<code>\\infty</code>, then compare. Use SymPy "
            "<code>simplify(a-b)==0</code> for symbolic equivalence.",
        )

    if g0.lower() == p0.lower():
        return (
            "case-only",
            "Differs only in case. Make the judger case-insensitive for "
            "word/text answers.",
        )

    # Try numeric-vs-symbolic: gold is symbolic, pred is decimal (or vice versa)
    if _is_number(p0) and not _is_number(g0):
        return (
            "symbolic-vs-decimal",
            "Gold is a closed form (e.g. <code>10^(-2)</code>, <code>132/13</code>) "
            "and the model returned a decimal. Either evaluate the gold "
            "symbolically and compare numerically in the judger, or instruct the "
            "model to return the exact closed form when one is asked for.",
        )

    return (
        "real-error",
        "No equivalence pattern matched — likely a real reasoning error. "
        "Inspect the response.",
    )


CATEGORY_LABELS = {
    "truncated": "Truncated output",
    "empty-extracted": "Empty extraction",
    "rounding": "Rounding / precision",
    "notation": "Notation only",
    "case-only": "Case-only mismatch",
    "symbolic-vs-decimal": "Symbolic vs decimal",
    "multi-blank-misaligned": "Multi-blank misaligned",
    "mcq-letter-disagree": "MCQ letter disagreement",
    "real-error": "Real reasoning error",
}

CATEGORY_COLORS = {
    "truncated": "#fff3cd",
    "empty-extracted": "#e2e3e5",
    "rounding": "#d1ecf1",
    "notation": "#d1ecf1",
    "case-only": "#d1ecf1",
    "symbolic-vs-decimal": "#d1ecf1",
    "multi-blank-misaligned": "#fde9e9",
    "mcq-letter-disagree": "#fde9e9",
    "real-error": "#f8d7da",
}


def load_questions() -> dict[int, dict]:
    qs: dict[int, dict] = {}
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qs[r["id"]] = r
    return qs


def extract_predicted(rec: dict) -> str:
    """Best-effort pull of the model's final answer."""
    extracted = (rec.get("extracted") or "").strip()
    if extracted:
        return extracted
    resp = rec.get("response", "")
    boxed = BOXED_RE.findall(resp)
    if boxed:
        return boxed[-1].strip()
    ans = ANS_RE.findall(resp)
    if ans:
        return ans[-1].strip()
    return "(none)"


def render_options(opts: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rows = []
    for i, opt in enumerate(opts):
        letter = letters[i] if i < len(letters) else str(i)
        rows.append(
            f'<li><span class="opt-letter">{letter}.</span> <span class="opt-text">{html.escape(opt)}</span></li>'
        )
    return '<ol class="options">' + "".join(rows) + "</ol>"


def render_gold(gold) -> str:
    if isinstance(gold, list):
        return ", ".join(html.escape(str(g)) for g in gold)
    return html.escape(str(gold))


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
.tag-err { background: #fde2e2; color: #a40000; }
.question { margin: 8px 0 12px 0; padding: 10px 12px; background: #fafafa; border-left: 3px solid #1a4d8c; }
.options { padding-left: 20px; }
.opt-letter { font-weight: bold; color: #555; margin-right: 4px; }
.answers { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0; }
.ans-box { padding: 10px 12px; border-radius: 6px; }
.ans-gold { background: #e8f7e8; border: 1px solid #b0d8b0; }
.ans-pred { background: #fde9e9; border: 1px solid #e8b0b0; }
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


def record_kind(rec: dict) -> str:
    route = rec.get("route") or {}
    if rec.get("format_type"):
        return str(rec["format_type"])
    if route.get("format_type"):
        return str(route["format_type"])
    return "mcq" if rec.get("is_mcq", False) else "free_response"


def render_case(rec: dict, q: dict | None) -> tuple[str, str]:
    rid = rec["id"]
    kind = record_kind(rec)
    is_mcq = rec.get("is_mcq", False) or kind in {"mcq", "multi_select", "true_false"}
    error_type = rec.get("error_type", "")
    if not error_type and rec.get("correct") is False:
        error_type = "wrong"
    finish_reason = rec.get("finish_reason", "")
    if not finish_reason:
        finish_reasons = rec.get("finish_reasons") or []
        if finish_reasons:
            finish_reason = ",".join(str(x) for x in finish_reasons if x)
    gold = render_gold(rec.get("gold"))
    pred = html.escape(extract_predicted(rec))

    category, advice = diagnose(rec, q)
    cat_label = CATEGORY_LABELS.get(category, category)
    cat_color = CATEGORY_COLORS.get(category, "#eee")

    tag_class = "tag-mcq" if is_mcq else "tag-free"
    tag_kind = f'<span class="tag {tag_class}">{html.escape(kind)}</span>'
    tag_err = f'<span class="tag tag-err">{html.escape(error_type)}</span>' if error_type else ""
    tag_finish = f'<span class="tag">finish: {html.escape(finish_reason)}</span>' if finish_reason and finish_reason != "stop" else ""
    tag_cat = f'<span class="cat-pill" style="background:{cat_color}">{html.escape(cat_label)}</span>'
    tag_prompt = ""
    if rec.get("prompt"):
        tag_prompt = f'<span class="tag">prompt: {html.escape(str(rec["prompt"]))}</span>'
    tag_config = ""
    if rec.get("config"):
        tag_config = f'<span class="tag">config: {html.escape(str(rec["config"]))}</span>'
    tag_fallback = ""
    if rec.get("fallback_used"):
        tag_fallback = '<span class="tag tag-err">fallback</span>'

    if q is None:
        q_html = '<em>(question id not found in data/public.jsonl)</em>'
        opts_html = ""
    else:
        q_html = html.escape(q.get("question", ""))
        opts = q.get("options")
        opts_html = render_options(opts) if opts else ""

    samples_html = ""
    all_samples = rec.get("all_samples")
    if all_samples and len(all_samples) > 1:
        chunks = []
        for i, s in enumerate(all_samples):
            chunks.append(
                f'<div class="sample"><b>Sample {i+1}:</b><pre class="response">{html.escape(s)}</pre></div>'
            )
        samples_html = (
            '<details><summary>All sampled responses ('
            f'{len(all_samples)})</summary>{"".join(chunks)}</details>'
        )

    response = rec.get("response", "")
    response_html = (
        '<details open><summary>Model response</summary>'
        f'<pre class="response">{html.escape(response)}</pre></details>'
    )

    raw_response = rec.get("raw_response")
    raw_response_html = ""
    if raw_response and raw_response != response:
        raw_response_html = (
            '<details><summary>Raw response before formatting/router cleanup</summary>'
            f'<pre class="response">{html.escape(raw_response)}</pre></details>'
        )

    fallback_raw = rec.get("fallback_raw_response")
    fallback_html = ""
    if fallback_raw and fallback_raw != raw_response:
        fallback_html = (
            '<details><summary>Fallback raw response</summary>'
            f'<pre class="response">{html.escape(fallback_raw)}</pre></details>'
        )

    diag_html = f"""
  <div class="diag" style="background:{cat_color}; border-left-color:{cat_color}">
    <div class="diag-label">Likely cause &middot; advice</div>
    <div class="diag-cat">{html.escape(cat_label)}</div>
    <div class="diag-advice">{advice}</div>
  </div>
"""

    section = f"""
<section class="case" id="q{rid}">
  <h2>#{rid} {tag_kind}{tag_err}{tag_prompt}{tag_config}{tag_fallback}{tag_finish}{tag_cat}</h2>
  <div class="question">{q_html}</div>
  {opts_html}
  <div class="answers">
    <div class="ans-box ans-gold"><div class="ans-label">Gold</div><div>{gold}</div></div>
    <div class="ans-box ans-pred"><div class="ans-label">Model</div><div>{pred}</div></div>
  </div>
  {diag_html}
  {response_html}
  {raw_response_html}
  {fallback_html}
  {samples_html}
</section>
"""
    return section, category


def render_file(name: str, records: list[dict], questions: dict[int, dict]) -> str:
    wrong = [r for r in records if r.get("correct") is False]
    total = len(records)

    rendered: list[str] = []
    cat_counts: dict[str, int] = {}
    for r in wrong:
        section, category = render_case(r, questions.get(r["id"]))
        rendered.append(section)
        cat_counts[category] = cat_counts.get(category, 0) + 1
    body_cases = "\n".join(rendered)

    cat_pills = "".join(
        f'<span style="background:{CATEGORY_COLORS.get(c, "#eee")}">'
        f'{html.escape(CATEGORY_LABELS.get(c, c))}: <b>{n}</b></span>'
        for c, n in sorted(cat_counts.items(), key=lambda kv: -kv[1])
    )

    toc = "\n".join(
        f'<a href="#q{r["id"]}">#{r["id"]} — gold {render_gold(r.get("gold"))}</a>'
        for r in wrong
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Wrong answers — {html.escape(name)}</title>
  <style>{CSS}</style>
  {MATHJAX}
</head>
<body>
  <h1>{html.escape(name)}</h1>
  <div class="summary">
    <div><b>Wrong:</b> {len(wrong)} / {total}
         (accuracy {100 * (total - len(wrong)) / total:.1f}%)</div>
    <div class="cat-summary">{cat_pills}</div>
  </div>
  <details><summary>Jump to question</summary><nav class="toc">{toc}</nav></details>
  {body_cases}
</body>
</html>
"""


def render_index(files: list[tuple[str, int, int]]) -> str:
    rows = "".join(
        f'<tr><td><a href="{html.escape(name)}.html">{html.escape(name)}</a></td>'
        f'<td>{wrong}</td><td>{total}</td>'
        f'<td>{100 * (total - wrong) / total:.1f}%</td></tr>'
        for name, wrong, total in files
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Wrong-answer reports</title>
  <style>{CSS}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #f4f6f8; }}
  </style>
</head>
<body>
  <h1>Wrong-answer reports</h1>
  <table>
    <thead><tr><th>File</th><th>Wrong</th><th>Total</th><th>Accuracy</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def result_files() -> list[Path]:
    files = {jf.resolve(): jf for jf in RESULT_DIR.glob("*.jsonl")}
    for extra_dir in EXTRA_RESULT_DIRS:
        if extra_dir.exists():
            files.update({jf.resolve(): jf for jf in extra_dir.glob("*.jsonl")})
    return sorted(files.values(), key=lambda p: p.stem)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    questions = load_questions()
    summary: list[tuple[str, int, int]] = []

    for jf in result_files():
        name = jf.stem
        records: list[dict] = []
        with jf.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        wrong_count = sum(1 for r in records if r.get("correct") is False)
        out_html = OUT_DIR / f"{name}.html"
        out_html.write_text(render_file(name, records, questions), encoding="utf-8")
        summary.append((name, wrong_count, len(records)))
        print(f"  {name}: {wrong_count}/{len(records)} wrong -> {out_html.name}")

    (OUT_DIR / "index.html").write_text(render_index(summary), encoding="utf-8")
    print(f"\nWrote {len(summary)} reports to {OUT_DIR}")
    print(f"Open: {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
