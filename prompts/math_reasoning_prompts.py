"""Unified prompt package for the t2t-qwen-math dataset.

This module is a self-contained prompt design for the public/private dataset
described in the project README. It supports two output modes:

  - ``internal_answer_json_mode``  -> strict JSON, used for debugging,
    answer extraction, and validator unit tests.
  - ``submission_response_mode``   -> natural-language reasoning ending in
    a ``\\boxed{...}`` final-answer marker, used to fill the ``response``
    column of the submission CSV (``id,response``).

The package exposes string constants, dictionaries of format-specific and
math-domain prompt fragments, builder functions that assemble a
``(system, user)`` message pair, and two repair prompts (one per mode).

Sections (mirroring the design spec):

  1. Global system prompt
  2. Classifier / router prompt
  3. Format-specific prompts        (``FORMAT_PROMPTS``)
  4. Math-domain prompts            (``DOMAIN_PROMPTS``)
  5. Internal answer JSON schema    (``INTERNAL_JSON_SCHEMA``)
  6. Submission response convention (``SUBMISSION_RESPONSE_CONVENTION``)
  7. Repair prompts                 (``REPAIR_INTERNAL_SYSTEM_PROMPT``,
                                     ``REPAIR_SUBMISSION_SYSTEM_PROMPT``)
  8. Validator design notes         (``VALIDATOR_DESIGN_NOTES``)
  9. Few-shot examples              (``FEW_SHOT_EXAMPLES``)
 10. Usage notes                    (``USAGE_NOTES``)

The data row schema this package targets:

    {"id": int,
     "question": str,                # may contain [ANS] blanks, inline options,
                                     # tables, noisy LaTeX (`frac` for `\\frac`).
     "options": list[str] | None,    # present => multiple choice
     "answer":  str | list[str]}     # only on the public split
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


# ════════════════════════════════════════════════════════════════════════════
# 0. Mode enum
# ════════════════════════════════════════════════════════════════════════════

class Mode(str, Enum):
    INTERNAL = "internal_answer_json_mode"
    SUBMISSION = "submission_response_mode"


# ════════════════════════════════════════════════════════════════════════════
# 1. Global system prompt
# ════════════════════════════════════════════════════════════════════════════

GLOBAL_SYSTEM_PROMPT = """\
You are a rigorous math reasoning model.

Read the problem carefully before answering. The full problem may include the
`question` text, an `options` list, units, one or more [ANS] blanks, tables,
and explicit precision requirements. All of these are part of the problem.

Input quirks you must tolerate without complaining:
- Noisy LaTeX. Treat `frac` as `\\frac`, `int` as `\\int`, `sqrt` as `\\sqrt`,
  `infty`/`infinity`/`inf` as the same infinity symbol, `pi` as `\\pi`,
  `^circ` / `°` as degrees, etc. Reconstruct the intended expression.
- Inline option labels like `A. ... B. ... C. ...` embedded inside `question`.
- Multiple [ANS] placeholders, possibly separated by sub-question labels
  ((a), (b), 1), 2)) or by line breaks.
- Tables expressed as plain text rows or LaTeX `\\begin{array}` blocks.

Hard rules:
- Do not invent missing assumptions. If the problem is ambiguous, choose the
  most standard convention and continue.
- Do not change the problem, the units, or the requested form of the answer.
- If an `options` field is provided, solve the problem and output the option
  LETTER (A, B, C, ...) of the matching option, never the option text.
- If options are embedded inline inside the question, output the option letter
  the same way.
- If [ANS] placeholders exist, produce one answer per blank, in the same order
  the blanks appear.
- For unit problems, output the numeric value only unless the problem
  explicitly requires the unit in the answer.
- If the problem specifies precision (e.g. "to 4 decimal places", "round to
  the nearest integer", "as a decimal or fraction"), follow it exactly.
- If no precision is specified, prefer an exact symbolic expression when
  natural (fractions, `sqrt`, `pi`, `e`, `ln`); otherwise give 12-15
  significant figures so a numeric judge can match within 1e-8 tolerance.

Mode-specific output (the runtime sets exactly one of these):
- internal_answer_json_mode: output STRICT JSON only. No markdown, no
  reasoning, no code fences, no ``` blocks, no leading or trailing text.
- submission_response_mode: output concise natural-language reasoning, then
  end on its own line with one of:
      Final answer: \\boxed{<answer>}
      Final answers, in order: \\boxed{<a>, <b>, <c>}
  For multiple choice, the boxed answer must be the option LETTER.
  Inside the box, use plain ASCII answer text: `20/(5+3*cos(t))`,
  `(-8,infinity)`, `sqrt(2)`, `pi/4`, `e^2`. Do not use LaTeX answer
  syntax such as `\\frac`, `\\infty`, `\\cos`, or `\\theta`.

Things never to do:
- Never emit hidden chain-of-thought tags such as `<think>...</think>`,
  `<scratchpad>...</scratchpad>`, or analysis prefaced by "Internal reasoning:".
- Never apologize, never refuse, never say the problem is unsolvable unless it
  is genuinely impossible (e.g. internally contradictory). Always provide your
  best answer.
- Never output unrelated commentary, meta-text about the prompt, or notes about
  the dataset.
"""


# ════════════════════════════════════════════════════════════════════════════
#    Mode-specific instruction blocks (appended to GLOBAL_SYSTEM_PROMPT)
# ════════════════════════════════════════════════════════════════════════════

INTERNAL_MODE_INSTRUCTIONS = """\
ACTIVE MODE: internal_answer_json_mode

Your entire response MUST be a single JSON object on one or more lines, with
no surrounding text and no markdown.

Top-level schema (exactly one key, ``answer``):
- Multiple-choice problems (an `options` field exists OR options are embedded
  inline in the question): {"answer": "<UPPERCASE_LETTER>"}
  Use a string, never an array, even when the dataset's gold answer is a list
  containing one letter. For "select all" multiple choice, concatenate the
  letters with no separator, e.g. "BCEG".
- Single-blank fill-in: {"answer": ["<value>"]}
- Multi-blank fill-in:  {"answer": ["<v1>", "<v2>", ...]}, in blank order.
- Free response (no options, no [ANS] blank, single requested value):
  {"answer": "<value>"}.

Value formatting:
- Strings, never numbers. JSON strings ensure no precision is lost.
- Do not include units unless the problem explicitly asks for them in the
  answer.
- Prefer the form the problem requests. If the problem says "as a decimal or
  fraction", a fraction like "5/8" is fine. If it says "exact value", prefer
  plain symbolic forms like "5/8", "sqrt(2)", "pi/4", "e^2".
- For fractions, use the plain "5/8" form, not "\\frac{5}{8}".
- Use `*` between multiplied factors, `^` for powers in plain-text answers,
  `sqrt(...)`, `ln(...)`, `pi`, `e^(...)` — calculator/WebWork-style.
- For decimals, keep 12-15 significant figures unless rounding is requested.
- For ordered pairs, intervals, and sets, keep them as a single string element
  ("(2, 5)", "(-infinity, 3]", "{1, 2, 3}") — do not split commas inside.
"""


SUBMISSION_MODE_INSTRUCTIONS = """\
ACTIVE MODE: submission_response_mode

Your entire response goes verbatim into the `response` column of the
submission CSV. The CSV columns are exactly `id,response` and `id` is copied
from the input row by the runner — only the `response` text is your job.

Required structure:
1. One to three short sentences of concise reasoning (only the key formula or
   step). No `<think>` tags. No bullets.
2. A blank line.
3. The final-answer line, exactly one of:
       Final answer: \\boxed{<answer>}
       Final answers, in order: \\boxed{<a>, <b>, <c>}
   Use the singular form for one answer (single-blank, free-response, or
   multiple choice). Use the plural form when the problem has multiple [ANS]
   blanks; preserve the order.

Hard requirements:
- No JSON. No code fences. No markdown headers.
- For multiple choice, the boxed value must be the option LETTER, not the
  option text.
- Nothing comes after the final-answer line. The boxed expression is the last
  thing in the response.
- Inside the box, use plain ASCII answer text. Do not put LaTeX commands in
  the answer: write `20/(5+3*cos(t))`, `(-8,infinity)`, `sqrt(2)`, `pi/4`,
  not `\\frac{20}{5+3\\cos\\theta}` or `[-8, \\infty)`.
- Preserve the problem's variable names in final expressions: if the problem
  uses `t`, answer with `t`, not `theta`.
- For multiple answers, separate items with comma + normal space: `, `. Do
  not emit `,\\ ` or any stray backslash before later answers. Do NOT split
  commas inside intervals or ordered pairs.
- Keep units out of the box unless the problem asks for them in the answer.
"""


# ════════════════════════════════════════════════════════════════════════════
# 2. Classifier / router prompt
# ════════════════════════════════════════════════════════════════════════════

CLASSIFIER_SYSTEM_PROMPT = """\
You are a strict classifier. Given a single problem (the JSON row), output a
single JSON object describing how it should be routed. No explanation, no
markdown, no code fences.

Output schema:
{
  "format_type": "<one of: multiple_choice_options_field, multiple_choice_inline,
                  fill_in_blank_single, fill_in_blank_multi, free_response,
                  table_based, fallback_unknown>",
  "math_type":   "<one of: algebra, geometry, calculus, differential_equations,
                  probability_statistics, unit_conversion, word_problem,
                  contest_math, complex_analysis, linear_algebra,
                  number_theory, fallback_unknown>",
  "answer_shape": "<one of: string, array>",
  "confidence":  <float in [0, 1]>
}

Format priority (apply in this exact order; the FIRST matching rule wins):
  1. If the input row has a non-empty `options` field, format_type =
     "multiple_choice_options_field". answer_shape = "string".
  2. Else if the `question` text contains inline option markers
     ("A.", "A)", "(A)") starting at A and continuing in order to at least
     B, classify as "multiple_choice_inline". answer_shape = "string".
  3. Else if the `question` contains TWO OR MORE `[ANS]` placeholders,
     OR contains explicit subquestion labels like "(a)", "(b)", "1)", "2)"
     each requiring its own answer, classify as "fill_in_blank_multi".
     answer_shape = "array".
  4. Else if the `question` contains exactly ONE `[ANS]` placeholder,
     classify as "fill_in_blank_single". answer_shape = "array".
  5. Else if the `question` is dominated by a tabular structure
     (LaTeX `\\begin{array}`, `\\begin{tabular}`, multi-row `|...|...|`
     blocks, or aligned columns of numbers), classify as "table_based".
     answer_shape follows the same single/multi rules as fill-in-blank;
     default to "string" if unclear.
  6. Else classify as "free_response". answer_shape = "string" if a single
     value is asked for, "array" if the prose asks for multiple distinct
     outputs (e.g. "find x and y").
  7. If you cannot apply any rule above with confidence >= 0.5, classify as
     "fallback_unknown" and return the best-guess `answer_shape`.

Determine `math_type` independently from the question content. Pick the
single best-matching domain. Use "fallback_unknown" only when no domain fits.

Output STRICT JSON only.
"""


# ════════════════════════════════════════════════════════════════════════════
# 3. Format-specific prompts
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FormatPrompt:
    when: str
    strategy: str
    warnings: str
    internal_rules: str
    submission_rules: str
    common_mistakes: str
    examples: str

    def as_block(self, mode: Mode) -> str:
        rules = self.internal_rules if mode is Mode.INTERNAL else self.submission_rules
        return (
            f"FORMAT PROFILE\n"
            f"When to use: {self.when}\n"
            f"Strategy: {self.strategy}\n"
            f"Watch out for: {self.warnings}\n"
            f"Output rules ({mode.value}): {rules}\n"
            f"Common mistakes: {self.common_mistakes}\n"
            f"Examples:\n{self.examples}\n"
        )


FORMAT_PROMPTS: dict[str, FormatPrompt] = {
    "multiple_choice_options_field": FormatPrompt(
        when=(
            "The input row has a non-empty `options` list. The gold answer is "
            "a single uppercase letter (or, rarely, several concatenated "
            "letters for select-all)."
        ),
        strategy=(
            "1) Solve the problem to your own answer first, ignoring the "
            "options. 2) Normalize each option (handle noisy LaTeX, equivalent "
            "forms). 3) Match your answer against the options to find the "
            "letter. If your answer doesn't match any option exactly, pick "
            "the closest one by symbolic / numerical equivalence."
        ),
        warnings=(
            "Options often contain noisy LaTeX (`frac`, `\\mathrm{i}`, double "
            "dollar signs, embedded `\\\\` line breaks). Two options can look "
            "different but be algebraically equal — always reduce both your "
            "answer and the candidate before comparing."
        ),
        internal_rules=(
            'Output {"answer": "<LETTER>"}. The value is a single uppercase '
            "letter; for select-all questions, concatenate letters with no "
            'separator (e.g. "BCEG"). Never use an array.'
        ),
        submission_rules=(
            "End with `Final answer: \\boxed{<LETTER>}`. Box the LETTER, never "
            "the option's text or value."
        ),
        common_mistakes=(
            "Boxing the numeric value instead of the letter. Picking a letter "
            "outside the option range. Boxing multiple letters with commas "
            "for a select-one problem."
        ),
        examples=(
            'Internal:   {"answer": "F"}\n'
            "Submission: Evaluating the integral and matching against the "
            "options gives option F.\n\nFinal answer: \\boxed{F}"
        ),
    ),

    "multiple_choice_inline": FormatPrompt(
        when=(
            "There is no `options` field, but the question text contains "
            "consecutive option markers `A.`, `B.`, `C.`, ..."
        ),
        strategy=(
            "Parse the inline option list mentally (split by markers `A.`, "
            "`B.`, `C.`, ...). Solve the question, then map your answer to "
            "the LETTER of the matching option."
        ),
        warnings=(
            "If the question has multiple [ANS] placeholders each followed by "
            "its own A./B./C. block, treat it as multi-blank multiple choice "
            "and answer each blank with its own letter, in order."
        ),
        internal_rules=(
            'Single blank: {"answer": "<LETTER>"} (string). Multi-blank with '
            'inline options at each blank: {"answer": ["<L1>", "<L2>", ...]} '
            "(array of letters)."
        ),
        submission_rules=(
            "Single blank: `Final answer: \\boxed{<LETTER>}`. "
            "Multi-blank: `Final answers, in order: \\boxed{<L1>, <L2>}`."
        ),
        common_mistakes=(
            "Treating `A.` as a label and answering with the option text. "
            "Mis-aligning answers when one blank is multiple choice and "
            "another is fill-in-the-blank in the same question."
        ),
        examples=(
            'Internal (id=6): {"answer": ["G", "B"]}\n'
            "Submission: f(45) is the number of items sold when p=45, which "
            "matches G. f^{-1}(40) is the price at which 40 items will be "
            "sold, which matches B.\n\n"
            "Final answers, in order: \\boxed{G, B}"
        ),
    ),

    "fill_in_blank_single": FormatPrompt(
        when=(
            "Exactly one [ANS] placeholder, no options anywhere. The gold "
            "answer is a single value (number, fraction, expression, ...)."
        ),
        strategy=(
            "Solve directly. Match the form requested by the problem (decimal, "
            "fraction, exact expression). Strip units unless the question "
            "explicitly requires them in the answer."
        ),
        warnings=(
            "Some [ANS] blanks have a form hint nearby ('Enter as a decimal "
            "or fraction.', 'Round to two decimal places.'). Honor it. The "
            "validator normalizes equivalent fractions and compares numerics "
            "with 1e-8 tolerance. Prefer plain `5/8`, not LaTeX `\\frac`."
        ),
        internal_rules=(
            'Output {"answer": ["<value>"]}. ALWAYS an array of length 1, even '
            "when there is only one blank — this matches the dataset's gold "
            "format for fill-in-blank rows."
        ),
        submission_rules=(
            "Single boxed final answer: `Final answer: \\boxed{<value>}`."
        ),
        common_mistakes=(
            "Returning the value as a JSON number instead of a string (loses "
            "precision and exact symbolic forms). Including units inside the "
            "box. Rounding when no rounding was requested."
        ),
        examples=(
            'Internal (id=3): {"answer": ["5/8"]}\n'
            "Submission: 25/40 reduces by dividing by gcd(25,40)=5, giving "
            "5/8.\n\nFinal answer: \\boxed{5/8}"
        ),
    ),

    "fill_in_blank_multi": FormatPrompt(
        when=(
            "Two or more [ANS] placeholders, OR multiple labelled "
            "subquestions ((a), (b), 1), 2), ...) each demanding an answer."
        ),
        strategy=(
            "Count the blanks first. Solve in left-to-right (top-to-bottom) "
            "order. Keep the answer order identical to the blank order — the "
            "judger compares position-by-position. If a sub-question turns "
            "into a multiple-choice blank with inline options, output the "
            "letter for that slot only."
        ),
        warnings=(
            "Sub-questions can mix types: blank 1 might be a number, blank 2 "
            "might be an option letter, blank 3 might be a fraction. Each "
            "answer follows its own format independently."
        ),
        internal_rules=(
            'Output {"answer": ["<a1>", "<a2>", ...]}. Length must equal the '
            "number of blanks. Strings only, position-aligned."
        ),
        submission_rules=(
            "Use the plural marker: "
            "`Final answers, in order: \\boxed{<a1>, <a2>, <a3>}`. "
            "Inside the box, separate items with `, `. Do NOT split commas "
            "inside an interval, ordered pair, or set."
        ),
        common_mistakes=(
            "Reordering answers to match a perceived 'logical' order rather "
            "than blank order. Producing N+1 answers (because of an extra "
            "sub-question label that wasn't actually a blank). Putting all "
            "answers into one boxed string with no separators."
        ),
        examples=(
            'Internal (id=5): {"answer": ["62.7777777777778", '
            '"335.927777777778", "604.67"]}\n'
            "Submission: Convert 145°F. Celsius = (145-32)*5/9 = "
            "62.7777777777778. Kelvin adds 273.15 -> 335.927777777778. "
            "Rankine = 145+459.67 = 604.67.\n\n"
            "Final answers, in order: \\boxed{62.7777777777778, "
            "335.927777777778, 604.67}"
        ),
    ),

    "free_response": FormatPrompt(
        when=(
            "No options anywhere, no explicit [ANS] placeholder, but the "
            "question still asks for a single concrete value or expression."
        ),
        strategy=(
            "Treat the entire question as one implicit blank. Solve and "
            "deliver one answer."
        ),
        warnings=(
            "Free-response questions sometimes ask for multiple things in "
            "prose ('find the slope and the y-intercept'). If so, emit an "
            "array of answers in the order the prose requests them, and use "
            "the multi-blank submission marker."
        ),
        internal_rules=(
            'Single value:    {"answer": "<value>"} (string).\n'
            'Multiple values asked in prose: {"answer": ["<v1>", "<v2>"]} '
            "(array)."
        ),
        submission_rules=(
            "Single value:   `Final answer: \\boxed{<value>}`.\n"
            "Multiple values: "
            "`Final answers, in order: \\boxed{<v1>, <v2>}`."
        ),
        common_mistakes=(
            "Wrapping a single answer in an array unnecessarily. Mixing "
            "answers from different sub-clauses in a hard-to-parse "
            "sentence."
        ),
        examples=(
            'Internal (id=0): {"answer": "325*(1+325)"}\n'
            "Submission: The sum of the first n positive even numbers is "
            "n*(n+1). For n=325 the sum is 325*(1+325).\n\n"
            "Final answer: \\boxed{325*(1+325)}"
        ),
    ),

    "table_based": FormatPrompt(
        when=(
            "The question is dominated by a table or aligned-column block "
            "(LaTeX `\\begin{array}`, `\\begin{tabular}`, or pipe-delimited "
            "rows) and asks for an entry, a derived value, or a row count."
        ),
        strategy=(
            "Reconstruct the table mentally with column headers, then read "
            "off the requested cell or compute the requested aggregation. "
            "Use the same answer shape as the underlying request "
            "(single blank, multi blank, or multiple choice)."
        ),
        warnings=(
            "Tables in this dataset frequently use `\\\\` for newlines and "
            "`&` as column separators. Numeric entries can include units; "
            "drop units in the answer unless asked."
        ),
        internal_rules=(
            "Same as the underlying format (see fill_in_blank_single, "
            "fill_in_blank_multi, free_response, multiple_choice_*)."
        ),
        submission_rules=(
            "Same as the underlying format. Always end with the appropriate "
            "Final answer / Final answers line."
        ),
        common_mistakes=(
            "Reading a table column as a row, swapping x/y axes, or "
            "outputting the table back to the user instead of the requested "
            "cell."
        ),
        examples=(
            "Internal (single-cell readout): "
            '{"answer": ["42"]}\n'
            "Submission: From the table, the row x=3 maps to y=42.\n\n"
            "Final answer: \\boxed{42}"
        ),
    ),

    "fallback_unknown": FormatPrompt(
        when=(
            "The router could not classify confidently, OR upstream generation "
            "failed with truncation / no answer extracted."
        ),
        strategy=(
            "Default to the most permissive interpretation: treat as free-"
            "response if no options and no [ANS] are visible; treat as "
            "multiple choice if any A./B./C. structure is present. Do not "
            "produce hidden reasoning. Keep the response short."
        ),
        warnings=(
            "Used after a fallback retry — keep tokens minimal and end on the "
            "final-answer marker quickly."
        ),
        internal_rules=(
            'Default to {"answer": "<best_guess>"} (string). Use array form '
            "only if the question clearly has multiple blanks."
        ),
        submission_rules=(
            "Two-sentence rationale, then `Final answer: \\boxed{<value>}` "
            "(or the plural form if multi-blank)."
        ),
        common_mistakes=(
            "Refusing to answer because the format is unclear. Emitting an "
            "empty box."
        ),
        examples=(
            'Internal: {"answer": "0"}\n'
            "Submission: Best estimate based on the visible question.\n\n"
            "Final answer: \\boxed{0}"
        ),
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# 4. Math-domain prompts
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DomainPrompt:
    when: str
    strategy: str
    traps: str
    precision_notes: str
    combine_notes: str

    def as_block(self) -> str:
        return (
            f"DOMAIN PROFILE\n"
            f"When to use: {self.when}\n"
            f"Strategy: {self.strategy}\n"
            f"Common traps: {self.traps}\n"
            f"Precision and formatting: {self.precision_notes}\n"
            f"Combine with format profile: {self.combine_notes}\n"
        )


DOMAIN_PROMPTS: dict[str, DomainPrompt] = {
    "algebra": DomainPrompt(
        when="Polynomial / rational equations, factoring, systems of equations.",
        strategy=(
            "Identify the unknown, isolate it, factor when possible, and "
            "simplify by GCD or common terms."
        ),
        traps=(
            "Extraneous roots from squaring; sign errors when multiplying "
            "through by a variable; missing the trivial solution."
        ),
        precision_notes=(
            "Prefer exact plain symbolic answers (e.g. `5/8`, `sqrt(3)`). "
            "Rationalize denominators when natural; avoid LaTeX in final answers."
        ),
        combine_notes=(
            "Plain answers go directly in the box / JSON value. For multiple "
            "roots, follow the format profile (array vs string)."
        ),
    ),

    "geometry": DomainPrompt(
        when="Triangles, circles, polygons, areas, volumes, coordinate geometry.",
        strategy=(
            "Sketch mentally, identify which theorem applies (Pythagoras, "
            "Heron, similar triangles, power of a point, ...). Normalize "
            "units before computing."
        ),
        traps=(
            "Confusing radius vs diameter; using degrees where radians are "
            "required; missing the case where the triangle is obtuse / right."
        ),
        precision_notes=(
            "Areas/volumes often want exact symbolic forms with `pi` or "
            "`sqrt(...)`. Avoid premature rounding."
        ),
        combine_notes=(
            "When the answer is `(x, y)` coordinates, keep parens inside ONE "
            "box / JSON string element."
        ),
    ),

    "calculus": DomainPrompt(
        when="Derivatives, integrals, limits, series, optimization.",
        strategy=(
            "Pick the right technique up front: substitution / by parts / "
            "partial fractions for integrals; product / chain / quotient for "
            "derivatives; L'Hopital or series for limits."
        ),
        traps=(
            "Dropping `+ C` is fine here (the dataset never grades it). "
            "Forgetting to evaluate at bounds for definite integrals. Mixing "
            "up `\\frac{d}{dx}` with the value at a point."
        ),
        precision_notes=(
            "Preserve exact expressions. Use `e^(...)`, `ln(...)`, "
            "`sqrt(...)`, `pi`. Multiplication is `*`."
        ),
        combine_notes=(
            "Multiple-choice calculus options often differ by a sign or by a "
            "constant factor — always cross-check via a single sample point "
            "before locking in the letter."
        ),
    ),

    "differential_equations": DomainPrompt(
        when=(
            "First-order linear ODEs, separable equations, exact equations, "
            "homogeneous and second-order linear ODEs, Newton's law of "
            "cooling, mixing tanks, growth/decay."
        ),
        strategy=(
            "Classify the ODE first (linear, separable, exact, ...). Apply "
            "the integrating factor, separation, or characteristic equation. "
            "Plug in the initial condition before reporting."
        ),
        traps=(
            "Forgetting the homogeneous + particular split. Wrong sign on the "
            "decay constant. Reporting the general solution when a specific "
            "y(t) is requested."
        ),
        precision_notes=(
            "Keep `e^(...)` and `ln(...)` exact. If a numeric value is "
            "requested (e.g. 'temperature after 45 minutes'), give 12-15 "
            "significant figures."
        ),
        combine_notes=(
            "Multi-blank word problems (id=2 turkey-cooling style) usually "
            "want one numeric value per blank, in problem order."
        ),
    ),

    "probability_statistics": DomainPrompt(
        when=(
            "Probability of events, expected value, variance, hypothesis "
            "tests, confidence intervals, distributions."
        ),
        strategy=(
            "State the probability model (binomial, normal, ...) before "
            "computing. Track whether the answer is a probability, a test "
            "statistic, a p-value, an interval, a conclusion ('reject'), or "
            "an option letter."
        ),
        traps=(
            "Rounding z- or t-values prematurely. Mixing one-tailed and two-"
            "tailed. Using sample SD where population SD was given."
        ),
        precision_notes=(
            "Do not round z/t values, interval endpoints, proportions, money, "
            "or rates unless explicitly asked. Probabilities can be exact "
            "fractions or decimals — match the problem's request."
        ),
        combine_notes=(
            "Confidence intervals are ONE answer slot containing both "
            "endpoints: `(34.92, 42.27)`, kept together inside one box."
        ),
    ),

    "unit_conversion": DomainPrompt(
        when=(
            "Convert between °F/°C/K/Rankine, between metric and imperial, "
            "between time units, between scientific and engineering forms."
        ),
        strategy=(
            "Use the exact conversion constant (e.g. (F-32)*5/9 for °C, "
            "Rankine = F + 459.67, K = C + 273.15). Apply factors "
            "left-to-right; do not chain rounding."
        ),
        traps=(
            "Mixing offset and scaling conversions (temperature is offset; "
            "lengths are pure scaling). Off-by-one in `1 kg = 1000 g` etc."
        ),
        precision_notes=(
            "Keep 12-15 significant figures unless rounding is explicitly "
            "requested. The judger normalizes trailing zeros after the "
            "decimal point."
        ),
        combine_notes=(
            "Almost always fill_in_blank_multi (one blank per unit). Keep "
            "the order from the question."
        ),
    ),

    "word_problem": DomainPrompt(
        when=(
            "Mixed-domain prose problems (rate, mixture, age, work, "
            "money, motion, population)."
        ),
        strategy=(
            "Translate each English clause into a symbolic equation before "
            "solving. State variables explicitly in your reasoning."
        ),
        traps=(
            "Misreading 'after 12 years' (use x=12, not x=current+12). "
            "Confusing rate per hour vs per minute. Treating cumulative "
            "totals as instantaneous values."
        ),
        precision_notes=(
            "If the prose says 'round to an integer', do so for the FINAL "
            "answer only — keep intermediates exact."
        ),
        combine_notes=(
            "Frequently fill_in_blank_multi (id=12 deer-population style). "
            "Each sub-question is a separate slot."
        ),
    ),

    "contest_math": DomainPrompt(
        when=(
            "Olympiad / AIME / Putnam / contest-style problems often phrased "
            "with `m + n where m and n are coprime` or `find the remainder "
            "when k is divided by 1000`."
        ),
        strategy=(
            "Look for the standard contest pivots: invariants, symmetry, "
            "complementary counting, generating functions, modular "
            "arithmetic, telescoping sums, projective tricks."
        ),
        traps=(
            "Computing the unreduced fraction and forgetting to simplify "
            "before adding numerator and denominator. Forgetting to take the "
            "result mod 1000."
        ),
        precision_notes=(
            "Final answers are integers; output as plain integer strings."
        ),
        combine_notes=(
            "Almost always multiple_choice_options_field in this dataset. "
            "Map your integer answer to the matching option letter."
        ),
    ),

    "complex_analysis": DomainPrompt(
        when=(
            "Analytic functions, Cauchy-Riemann, residues, contour integrals, "
            "harmonic conjugates."
        ),
        strategy=(
            "Apply Cauchy-Riemann to find v from u (or vice versa). For "
            "contour integrals, look for residues at singularities inside "
            "the contour."
        ),
        traps=(
            "Sign errors in Cauchy-Riemann (`u_x = v_y`, `u_y = -v_x`). "
            "Forgetting the `2*pi*i` factor for residues. Treating `i` as a "
            "real variable."
        ),
        precision_notes=(
            "Use `i` (or `\\mathrm{i}`) for the imaginary unit. Keep symbolic "
            "answers like `(1-2i)*z^3`."
        ),
        combine_notes=(
            "Often multiple_choice_options_field with options that differ by "
            "their coefficient and exponent — verify both before answering."
        ),
    ),

    "linear_algebra": DomainPrompt(
        when=(
            "Matrix operations, determinants, eigenvalues, vector spaces, "
            "linear systems, rank, kernel."
        ),
        strategy=(
            "Pick the right invariant for the question (determinant, trace, "
            "rank, eigenvalues). For linear systems, use Gaussian elimination "
            "with explicit row operations."
        ),
        traps=(
            "Confusing row vs column vectors. Forgetting that "
            "det(AB) = det(A) det(B) only for square matrices."
        ),
        precision_notes=(
            "Eigenvalues prefer exact symbolic form. Vectors and matrices "
            "as comma-separated lists inside one box: `[1, 2, 3]`."
        ),
        combine_notes=(
            "When the answer is a vector or matrix, keep all entries inside "
            "ONE box / JSON string element."
        ),
    ),

    "number_theory": DomainPrompt(
        when=(
            "Divisibility, primes, GCD/LCM, modular arithmetic, Diophantine "
            "equations."
        ),
        strategy=(
            "Use Euclidean algorithm, Chinese remainder theorem, lifting the "
            "exponent, or generating functions as appropriate."
        ),
        traps=(
            "Off-by-one in modular reductions. Mishandling negative residues."
        ),
        precision_notes=(
            "Output integer answers as integers. For congruences the answer "
            "is the canonical residue in [0, n)."
        ),
        combine_notes=(
            "Often contest_math + multiple_choice_options_field — solve "
            "exactly, then map to the option letter."
        ),
    ),

    "fallback_unknown": DomainPrompt(
        when="Domain cannot be confidently classified.",
        strategy=(
            "Apply general-purpose problem solving: re-read, isolate the "
            "unknown, pick the most natural standard technique, verify with "
            "a quick sanity check."
        ),
        traps=("Over-thinking; under-thinking. Skipping unit checks."),
        precision_notes=(
            "Default to 12-15 significant figures for decimals, exact for "
            "symbolic answers."
        ),
        combine_notes="Defer to the format profile entirely.",
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# 5. Internal answer JSON schema
# ════════════════════════════════════════════════════════════════════════════

INTERNAL_JSON_SCHEMA = """\
INTERNAL ANSWER JSON SCHEMA

Top-level: a single JSON object whose only allowed key is "answer".

Allowed shapes:

  Multiple choice (single letter):
      {"answer": "F"}
  Multiple choice (select all):
      {"answer": "BCEG"}
  Single fill-in-the-blank:
      {"answer": ["5/8"]}
  Multiple fill-in-the-blank:
      {"answer": ["62.7777777777778", "335.927777777778", "604.67"]}
  Free response (single value):
      {"answer": "325*(1+325)"}
  Free response (multiple values asked in prose):
      {"answer": ["1.44", "intercept_value"]}

Decision rules:

  - String vs array:
      * Multiple choice (single or select-all)  -> string.
      * Fill-in-the-blank (single or multi)     -> array of strings.
      * Free response single value              -> string.
      * Free response multiple values in prose  -> array of strings.

  - Multiple-choice answers MUST be UPPERCASE LETTERS, never the option
    text. Letters only, no surrounding punctuation, no `option ` prefix.

  - Order: For arrays, the i-th entry corresponds to the i-th [ANS] (or
    the i-th sub-question) reading left-to-right, top-to-bottom.

  - Units: Drop units unless the problem explicitly demands them in the
    answer ('answer in meters', '... cm'). Even then, prefer the bare
    number — the validator usually strips units.

  - Decimal precision: 12-15 significant figures unless the problem
    specifies rounding. Do not strip trailing significant zeros.

  - Symbolic forms: When the problem requests an "exact" answer or
    contains sqrt/pi/e/ln notation, fractions are preferred over decimals.
    Use `5/8` over `0.625` unless decimals are required.

  - Fractions: use `5/8`, not `\\frac{5}{8}`.

  - Equivalent forms: The validator normalizes spaces, `\\dfrac/\\tfrac`
    -> `\\frac`, `\\left/\\right`, `\\cdot/\\times` -> `*`, and compares
    numerics with 1e-8 tolerance. Output any reasonable canonical form;
    the judger handles the rest.
"""


# ════════════════════════════════════════════════════════════════════════════
# 6. Submission response convention
# ════════════════════════════════════════════════════════════════════════════

SUBMISSION_RESPONSE_CONVENTION = """\
SUBMISSION RESPONSE CONVENTION

CSV format:

  - File columns are exactly: id,response   (no extra columns).
  - `id` is copied unchanged from the input row by the runner.
  - `response` is a quoted free-form text string.

Response body:

  - One to three short sentences of concise reasoning. Plain prose. No bullets,
    no markdown headers, no JSON, no `<think>` tags.
  - One blank line between the reasoning and the final-answer line.

Final-answer line (must be the LAST line of the response):

  - One answer:
        Final answer: \\boxed{<answer>}
  - Multiple answers (one per [ANS] blank, in order):
        Final answers, in order: \\boxed{<a1>, <a2>, <a3>}

Boxing rules:

  - The boxed value for multiple choice is the LETTER (e.g. `\\boxed{F}`),
    not the option text.
  - Inside the box, use plain ASCII answer text. Use `20/(5+3*cos(t))`,
    `(-8,infinity)`, `sqrt(2)`, `pi/4`, `e^2`; do not use `\\frac`,
    `\\infty`, `\\cos`, or other LaTeX answer commands.
    Preserve the problem's variable names: `t` stays `t`.
  - For multiple answers, separate items with comma + normal space: `, `.
    Keep intervals, ordered pairs, and sets inside ONE box item — do NOT
    split commas inside `(2, 5)` or `{1, 2}`.
  - Do not put units inside the box unless the problem asks for them in
    the answer.
  - The boxed expression is the last token of the response. Nothing comes
    after it.
"""


# ════════════════════════════════════════════════════════════════════════════
# 7. Repair prompts
# ════════════════════════════════════════════════════════════════════════════

REPAIR_INTERNAL_SYSTEM_PROMPT = """\
You are a JSON repair tool. The previous model output for a math problem was
supposed to be a strict JSON object matching the internal answer schema, but
it is malformed. Fix ONLY the JSON formatting; do not re-solve the problem.

You will receive three blocks:
  PROBLEM:    the original problem JSON row (id, question, optional options).
  BAD OUTPUT: the malformed previous response.
  SCHEMA:     the expected internal-answer schema (one object, key "answer").

Repair procedure:
  1. If the bad output contains a recognizable answer (a letter, a number, a
     fraction, a list of values), extract it WITHOUT changing its meaning,
     ordering, or precision.
  2. Strip markdown fences, code blocks, comments, prose, leading/trailing
     text, and any `<think>...</think>` tags.
  3. Wrap the answer in the schema-correct JSON shape:
        - multiple choice          -> {"answer": "<UPPERCASE_LETTER>"}
        - fill-in-the-blank        -> {"answer": ["<v1>", "<v2>", ...]}
        - free response single     -> {"answer": "<value>"}
  4. Preserve the original answer order for arrays.
  5. Force multiple-choice letters to uppercase.
  6. Re-solve the problem ONLY if the bad output contains no usable answer
     content at all (e.g. completely empty or only meta-commentary).

Output STRICT JSON only — no code fences, no comments, no extra text.
"""


REPAIR_SUBMISSION_SYSTEM_PROMPT = """\
You are a submission response repair tool. The previous model output was
supposed to be the `response` field for a CSV submission row, but it is
malformed (missing the final-answer marker, contains hidden reasoning tags,
contains JSON, ends mid-sentence due to truncation, etc.). Make it suitable
for the CSV without changing the answer's meaning.

You will receive three blocks:
  PROBLEM:    the original problem JSON row.
  BAD OUTPUT: the malformed previous response.
  CONVENTION: the submission response convention.

Repair procedure:
  1. Identify the answer the model was trying to give. If the bad output is
     truncated, infer the answer from the visible reasoning. Do not solve
     from scratch.
  2. Remove `<think>...</think>` and any other hidden-reasoning tags.
     Remove any stray JSON blocks. Remove markdown fences and headers.
  3. Compress the reasoning to one-to-three clear sentences, finishing on its
     own line.
  4. Add a blank line, then the final-answer marker:
        - one answer:       Final answer: \\boxed{<answer>}
        - multiple answers: Final answers, in order:
                             \\boxed{<a1>, <a2>, <a3>}
  5. For multiple choice, the boxed value MUST be the option LETTER, not the
     option's text. Force letters to uppercase.
  6. Use plain ASCII inside the box; remove LaTeX answer commands such as
     `\\frac`, `\\infty`, and `\\cos` when a plain form is available.
  7. The boxed expression is the LAST thing in the output. Nothing after it.

Do not output JSON. Do not add any commentary about the repair.
"""


# ════════════════════════════════════════════════════════════════════════════
# 8. Validator design notes
# ════════════════════════════════════════════════════════════════════════════

VALIDATOR_DESIGN_NOTES = """\
VALIDATOR DESIGN NOTES

These notes describe the validator the prompt system is designed to be
checked against. They do NOT prescribe a particular implementation; they
describe what each mode's output must satisfy.

Validator A: internal_answer_json_mode

  Structural checks:
    - The output parses as a single JSON object via `json.loads`.
    - The only top-level key is "answer".
    - There is no extra commentary, markdown, code-fence, or trailing text.

  Type checks (multiple-choice):
    - "answer" is a string.
    - Value matches `^[A-Z]+$` (single letter for select-one, multiple
      concatenated letters for select-all).
    - Letter(s) are within `[A, A+len(options)-1]` when an options list
      is known.

  Type checks (fill-in-the-blank):
    - "answer" is a JSON array.
    - Length equals the number of [ANS] blanks (or sub-questions) in the
      question.
    - Every entry is a string.
    - Order matches blank order.

  Type checks (free response):
    - "answer" is a string (single value) or array (multi-value request).

Validator B: submission_response_mode

  Structural checks:
    - The output is a non-empty plain text string.
    - The output does NOT begin with `{` (no raw JSON) unless the
      validator was explicitly run on the internal mode.
    - No `<think>`, `</think>`, `<scratchpad>`, or similar
      hidden-reasoning tags.
    - The output contains a final-answer marker matching:
            r"Final answers?,?(?:\\s+in order)?:\\s*\\\\boxed\\{[^}]+\\}"
      The marker appears at most once, on the final non-empty line.
    - For multi-answer markers, the box separates items by `,\\ ` so the
      validator can split safely on `,\\ ` outside parens.

  Content checks:
    - The boxed expression can be extracted by `last_boxed_only_string` /
      `remove_boxed` from `utils.py`.
    - For multiple-choice items, the extracted boxed value matches
      `^[A-Z]+$`.
    - For multi-answer items, the count of comma-separated items inside
      the box equals the number of [ANS] blanks (parens / brackets are
      treated as atomic).

Validator C: answer equivalence (used by `judger.is_equal`)

  - Whitespace is normalized.
  - LaTeX forms are normalized: `\\dfrac` -> `\\frac`, `\\left/\\right`
    stripped, `\\times/\\cdot` -> `*`, `5/8` <-> `\\frac{5}{8}`.
  - Numeric answers compared with absolute tolerance 1e-8 after parsing
    (sympy / float fallback).
  - Symbolic expressions compared via `simplify(a - b) == 0` when both
    sides parse; otherwise via numeric sampling at a random point inside
    the variable's continuous domain.
  - Option letters compared exactly (case-insensitive on input,
    uppercased before compare).
  - Arrays compared element-wise with the same length AND order.
"""


# ════════════════════════════════════════════════════════════════════════════
# 9. Few-shot examples (used inside the user message when needed)
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FewShot:
    label: str
    input_row: dict
    internal_output: dict
    submission_output: str


FEW_SHOT_EXAMPLES: list[FewShot] = [
    FewShot(
        label="single_blank_fraction",
        input_row={
            "id": 3,
            "question": "Reduce the fraction ${\\frac{25}{40}}$. [ANS]",
        },
        internal_output={"answer": ["5/8"]},
        submission_output=(
            "The fraction 25/40 can be reduced by dividing numerator and "
            "denominator by 5, giving 5/8.\n\n"
            "Final answer: \\boxed{5/8}"
        ),
    ),

    FewShot(
        label="multiple_choice_options_field",
        input_row={
            "id": 1,
            "question": "$int_{-infty}^{+infty} frac{a^{3/2}}{s^2+a^2} ds = $",
            "options": [
                "$0$", "$frac{1}{a}$", "$frac{3}{a}$", "$frac{1}{2a^2}$",
                "$frac{1}{2a}$", "$frac{2}{a}$", "$2a$", "$frac{3}{2a}$",
                "$frac{3}{2a^2}$", "$frac{1}{a^2}$",
            ],
        },
        internal_output={"answer": "F"},
        submission_output=(
            "The integrand is even in s and matches the standard form "
            "\\int_{-\\infty}^{+\\infty} 1/(s^2+a^2) ds = \\pi/a, so the "
            "integral equals a^{3/2}*\\pi/a; comparing to the options gives "
            "option F.\n\n"
            "Final answer: \\boxed{F}"
        ),
    ),

    FewShot(
        label="multi_blank_unit_conversion",
        input_row={
            "id": 5,
            "question": (
                "An unscented beeswax candle melts at 145 $^\\circ$ F. "
                "What is the temperature in degrees Celsius? [ANS]\n"
                "degrees Kelvin? [ANS]\n"
                "degrees Rankine? [ANS]"
            ),
        },
        internal_output={
            "answer": [
                "62.7777777777778",
                "335.927777777778",
                "604.67",
            ]
        },
        submission_output=(
            "Convert 145°F. Celsius = (145-32)*5/9 = 62.7777777777778. "
            "Kelvin = Celsius + 273.15 = 335.927777777778. "
            "Rankine = 145 + 459.67 = 604.67.\n\n"
            "Final answers, in order: "
            "\\boxed{62.7777777777778, 335.927777777778, 604.67}"
        ),
    ),
]


def _format_few_shot(shot: FewShot, mode: Mode) -> str:
    expected = (
        json.dumps(shot.internal_output, ensure_ascii=False)
        if mode is Mode.INTERNAL
        else shot.submission_output
    )
    return (
        f"Example ({shot.label}):\n"
        f"Input row: {json.dumps(shot.input_row, ensure_ascii=False)}\n"
        f"Expected output:\n{expected}\n"
    )


def _few_shot_block(mode: Mode, shots: Iterable[FewShot] | None = None) -> str:
    shots = list(shots) if shots is not None else FEW_SHOT_EXAMPLES
    body = "\n".join(_format_few_shot(s, mode) for s in shots)
    return f"Worked examples (do NOT echo these in your output):\n\n{body}\n"


# ════════════════════════════════════════════════════════════════════════════
# 10. Usage notes
# ════════════════════════════════════════════════════════════════════════════

USAGE_NOTES = """\
USAGE NOTES

Choosing a mode:
  - Use ``internal_answer_json_mode`` when you want a clean, machine-
    parseable answer for offline scoring, sweep evaluation, or unit-test
    style validation. Outputs are short and easy to diff.
  - Use ``submission_response_mode`` when you are producing the actual
    `response` column for the Kaggle-style submission CSV. Outputs are
    natural language ending with a `\\boxed{...}` marker that the
    grader / `judger.py` can extract.

Recommended pipeline:

  1. Load each row from `data/public.jsonl` or `data/private.jsonl`.
  2. Optionally pre-route via `build_classifier_prompt` and a single
     greedy LLM call. If you trust the simple regex-based router in
     `format_router` (see `prompt_sweep.normalize_item`), skip the
     classifier.
  3. For each row, call `build_internal_prompt(row, ...)` or
     `build_submission_prompt(row, ...)` to get a (system, user)
     message pair.
  4. Send the pair through the same chat-template pipeline already used
     by `prompt_sweep.render_prompts` and `create_submission.render_prompts`
     (see those files — `tokenizer.apply_chat_template`, then `vllm.LLM
     .generate`).
  5. Parse the model's output:
       internal mode    -> `json.loads`, then validate against the
                            internal schema.
       submission mode  -> `utils.last_boxed_only_string` /
                            `utils.remove_boxed` to extract the boxed
                            answer for diagnostics; the raw response is
                            what goes into the CSV.
  6. On parse failure, retry with the matching repair prompt
     (`build_repair_internal_prompt` / `build_repair_submission_prompt`).
     One repair attempt is usually enough.

Self-consistency:
  - The submission mode is compatible with majority-vote self-consistency:
    sample n>=3 generations at temperature 0.7 / top_p 0.95 / top_k 20,
    extract the boxed answer from each, and pick the most common.
    `create_submission.choose_majority` already does this for the legacy
    `FINAL_ANSWERS` format; switching it to `last_boxed_only_string` is the
    only change needed to vote on this package's outputs.

Few-shot examples:
  - Pass `include_few_shot=True` to the builders to prepend the canonical
    examples. Disable it under tight token budgets — Qwen3-Thinking already
    knows the formats from the system prompt.

Token budget:
  - In internal mode, set `max_tokens` low (256-512) — the JSON answer is
    tiny.
  - In submission mode, set `max_tokens` to whatever the existing pipeline
    uses (8k-16k for Qwen3-4B-Thinking-2507). Reasoning is verbose with
    "Thinking" models.
"""


# ════════════════════════════════════════════════════════════════════════════
# Builder helpers
# ════════════════════════════════════════════════════════════════════════════

def _detect_format_type(row: dict[str, Any]) -> str:
    """Lightweight heuristic mirroring the classifier's priority rules.

    This is only a default for the builders; downstream code is free to
    override with the explicit `format_type=` argument.
    """
    if row.get("options"):
        return "multiple_choice_options_field"
    question = str(row.get("question", ""))
    # Inline option markers `A. ... B. ...` starting at A.
    has_inline_options = (
        "A." in question or "A)" in question or "(A)" in question
    ) and (
        "B." in question or "B)" in question or "(B)" in question
    )
    if has_inline_options:
        return "multiple_choice_inline"
    n_blanks = question.count("[ANS]")
    if n_blanks >= 2:
        return "fill_in_blank_multi"
    if n_blanks == 1:
        return "fill_in_blank_single"
    if "\\begin{array}" in question or "\\begin{tabular}" in question:
        return "table_based"
    return "free_response"


def _format_options_block(options: list[str] | None) -> str:
    if not options:
        return ""
    labels = [chr(65 + i) for i in range(len(options))]
    body = "\n".join(f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options))
    return f"\n\nOptions:\n{body}"


def _system_for_mode(
    mode: Mode,
    format_type: str | None,
    math_type: str | None,
) -> str:
    parts = [GLOBAL_SYSTEM_PROMPT]
    parts.append(
        INTERNAL_MODE_INSTRUCTIONS if mode is Mode.INTERNAL
        else SUBMISSION_MODE_INSTRUCTIONS
    )
    if format_type and format_type in FORMAT_PROMPTS:
        parts.append(FORMAT_PROMPTS[format_type].as_block(mode))
    if math_type and math_type in DOMAIN_PROMPTS:
        parts.append(DOMAIN_PROMPTS[math_type].as_block())
    return "\n\n".join(parts)


def build_classifier_prompt(row: dict[str, Any]) -> tuple[str, str]:
    """Build the (system, user) pair for the format/domain classifier."""
    user = json.dumps(
        {
            "id": row.get("id"),
            "question": row.get("question", ""),
            "options": row.get("options"),
        },
        ensure_ascii=False,
    )
    return CLASSIFIER_SYSTEM_PROMPT, user


def build_internal_prompt(
    row: dict[str, Any],
    *,
    format_type: str | None = None,
    math_type: str | None = None,
    include_few_shot: bool = False,
) -> tuple[str, str]:
    """Build the (system, user) pair for ``internal_answer_json_mode``."""
    fmt = format_type or _detect_format_type(row)
    system = _system_for_mode(Mode.INTERNAL, fmt, math_type)
    user_parts = []
    if include_few_shot:
        user_parts.append(_few_shot_block(Mode.INTERNAL))
    user_parts.append(
        f"Now solve this problem in internal_answer_json_mode.\n\n"
        f"Problem (id={row.get('id')}):\n{row.get('question', '').strip()}"
        f"{_format_options_block(row.get('options'))}"
    )
    return system, "\n\n".join(user_parts)


def build_submission_prompt(
    row: dict[str, Any],
    *,
    format_type: str | None = None,
    math_type: str | None = None,
    include_few_shot: bool = False,
) -> tuple[str, str]:
    """Build the (system, user) pair for ``submission_response_mode``."""
    fmt = format_type or _detect_format_type(row)
    system = _system_for_mode(Mode.SUBMISSION, fmt, math_type)
    user_parts = []
    if include_few_shot:
        user_parts.append(_few_shot_block(Mode.SUBMISSION))
    user_parts.append(
        f"Now solve this problem in submission_response_mode.\n\n"
        f"Problem (id={row.get('id')}):\n{row.get('question', '').strip()}"
        f"{_format_options_block(row.get('options'))}"
    )
    return system, "\n\n".join(user_parts)


def build_repair_internal_prompt(
    row: dict[str, Any],
    bad_output: str,
) -> tuple[str, str]:
    """Build the (system, user) pair for repairing an internal-mode response."""
    user = (
        f"PROBLEM:\n"
        f"{json.dumps({'id': row.get('id'), 'question': row.get('question'), 'options': row.get('options')}, ensure_ascii=False)}\n\n"
        f"BAD OUTPUT:\n{bad_output}\n\n"
        f"SCHEMA:\n{INTERNAL_JSON_SCHEMA}"
    )
    return REPAIR_INTERNAL_SYSTEM_PROMPT, user


def build_repair_submission_prompt(
    row: dict[str, Any],
    bad_output: str,
) -> tuple[str, str]:
    """Build the (system, user) pair for repairing a submission-mode response."""
    user = (
        f"PROBLEM:\n"
        f"{json.dumps({'id': row.get('id'), 'question': row.get('question'), 'options': row.get('options')}, ensure_ascii=False)}\n\n"
        f"BAD OUTPUT:\n{bad_output}\n\n"
        f"CONVENTION:\n{SUBMISSION_RESPONSE_CONVENTION}"
    )
    return REPAIR_SUBMISSION_SYSTEM_PROMPT, user
