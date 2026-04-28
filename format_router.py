"""Format-first routing and final-answer cleanup for math generation.

The router deliberately classifies by answer format before looking at topic.
Subject prompts are only used after a problem has been identified as ordinary
free response.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


FORMAT_MCQ = "mcq"
FORMAT_FREE_RESPONSE = "free_response"
FORMAT_MULTI_BLANK = "multi_blank"
FORMAT_MULTI_SELECT = "multi_select"
FORMAT_TRUE_FALSE = "true_false"
FORMAT_ALGORITHM_SEQUENCE = "algorithm_sequence"

PROMPT_GENERAL_MCQ = "general_mcq_eliminate"
PROMPT_ALGORITHM_SEQUENCE = "algorithm_sequence_mcq"
PROMPT_STATISTICS = "statistics_prompt"
PROMPT_CALCULUS = "calculus_prompt"
PROMPT_MULTI_BLANK = "multi_blank_free_response"
PROMPT_BASELINE = "baseline"

PROMPT_CONFIGS = {
    PROMPT_GENERAL_MCQ: "greedy_n1",
    PROMPT_ALGORITHM_SEQUENCE: "greedy_n1",
    PROMPT_MULTI_BLANK: "sc_n3",
    PROMPT_STATISTICS: "sc_n3",
    PROMPT_CALCULUS: "sc_n3",
    PROMPT_BASELINE: "sc_n3",
}


@dataclass(frozen=True)
class Route:
    format_type: str
    prompt_family: str
    config_name: str
    expected_answers: int
    subject: str = "general"
    has_options: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data


_ANS_RE = re.compile(r"\[ANS\]", re.IGNORECASE)
_OPTION_MARKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z])[\.\)]\s+")
_SELECT_ALL_RE = re.compile(
    r"select\s+all|check\s+all|all\s+that\s+apply|"
    r"there\s+may\s+be\s+more\s+than\s+one|more\s+than\s+one\s+correct|"
    r"multiple\s+correct|\[CHECK\s+ALL\s+THAT\s+APPLY\]",
    re.IGNORECASE,
)
_TRUE_FALSE_RE = re.compile(
    r"true\s+or\s+false|t\s+or\s+f|t/f|enter\s+t\s+or\s+f|"
    r"select\s+true\s+or\s+false|choose\s+true\s+or\s+false|"
    r"statements?\s+(?:is|are)\s+true\s+or\s+false",
    re.IGNORECASE,
)
_ALGORITHM_RE = re.compile(
    r"we\s+now\s+define\s+an\s+algorithm|x_list|y_list|output\s+sequence",
    re.IGNORECASE,
)
_STATISTICS_TERMS = (
    "confidence interval",
    "hypothesis",
    "p-value",
    "p value",
    "sample",
    "population",
    "standard deviation",
    "variance",
    "regression",
    "z-score",
    "z score",
    "t-test",
    "statistic",
    "histogram",
    "quartile",
    "mean",
    "median",
    "probability",
)
_CALCULUS_TERMS = (
    "derivative",
    "differentiate",
    "integral",
    "integrate",
    "limit",
    "tangent",
    "slope",
    "critical point",
    "maximize",
    "minimize",
    "local maximum",
    "local minimum",
    "inflection",
    "differential equation",
)


def answer_values(item: dict[str, Any]) -> list[str]:
    answer = item.get("answer")
    if answer is None:
        return []
    values = answer if isinstance(answer, list) else [answer]
    return [str(value).strip() for value in values]


def answer_blank_count(question: str) -> int:
    return len(_ANS_RE.findall(str(question)))


def expected_answer_count(item: dict[str, Any]) -> int:
    values = answer_values(item)
    if values:
        return len(values)
    return max(1, answer_blank_count(str(item.get("question", ""))))


def extract_embedded_options(question: str) -> tuple[str, list[str]] | None:
    """Parse one inline A./B./C. option block.

    This is intentionally used only for single-blank rows. Multi-part rows often
    contain several independent option blocks, and flattening those into one
    options list corrupts the problem.
    """
    markers = list(_OPTION_MARKER_RE.finditer(question))
    for start_idx, marker in enumerate(markers):
        if marker.group(1) != "A":
            continue

        seq = []
        expected = ord("A")
        for candidate in markers[start_idx:]:
            if ord(candidate.group(1)) != expected:
                break
            seq.append(candidate)
            expected += 1

        if len(seq) < 2:
            continue

        options = []
        for idx, option_marker in enumerate(seq):
            end = seq[idx + 1].start() if idx + 1 < len(seq) else len(question)
            option = question[option_marker.end():end].strip()
            if not option:
                break
            options.append(option)
        else:
            prompt = question[:seq[0].start()].replace("[ANS]", "").strip()
            return prompt, options

    return None


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-normalized copy with single inline option blocks exposed."""
    normalized = dict(item)
    if normalized.get("options"):
        return normalized

    question = str(normalized.get("question", ""))
    if answer_blank_count(question) > 1:
        return normalized

    parsed = extract_embedded_options(question)
    if not parsed:
        return normalized

    parsed_question, options = parsed
    gold_values = answer_values(normalized)
    if len(gold_values) == 1 and re.fullmatch(r"[A-Z]+", gold_values[0]):
        max_idx = max(ord(letter) - ord("A") for letter in gold_values[0].upper())
        if max_idx >= len(options):
            return normalized

    normalized["question"] = parsed_question
    normalized["options"] = options
    normalized["_embedded_options"] = True
    return normalized


def _answer_is_multi_letter(item: dict[str, Any]) -> bool:
    values = answer_values(item)
    return len(values) == 1 and bool(re.fullmatch(r"[A-Z]{2,}", values[0].upper()))


def is_algorithm_sequence(item: dict[str, Any]) -> bool:
    return bool(_ALGORITHM_RE.search(str(item.get("question", ""))))


def is_multi_select(item: dict[str, Any]) -> bool:
    question = str(item.get("question", ""))
    return bool(_SELECT_ALL_RE.search(question)) or _answer_is_multi_letter(item)


def is_true_false(item: dict[str, Any]) -> bool:
    question = str(item.get("question", ""))
    values = [value.upper() for value in answer_values(item)]
    if len(values) > 1 and all(value in {"T", "F", "TRUE", "FALSE"} for value in values):
        return True
    if answer_blank_count(question) > 1 and _TRUE_FALSE_RE.search(question):
        return True
    return False


def is_multi_blank(item: dict[str, Any]) -> bool:
    return answer_blank_count(str(item.get("question", ""))) > 1 or len(answer_values(item)) > 1


def detect_subject(item: dict[str, Any]) -> str:
    question = str(item.get("question", "")).lower()
    if any(term in question for term in _STATISTICS_TERMS):
        return "statistics"
    if any(term in question for term in _CALCULUS_TERMS):
        return "calculus"
    return "general"


def classify_format(item: dict[str, Any]) -> str:
    has_options = bool(item.get("options"))
    if has_options and is_algorithm_sequence(item):
        return FORMAT_ALGORITHM_SEQUENCE
    if has_options and is_multi_select(item):
        return FORMAT_MULTI_SELECT
    if has_options:
        return FORMAT_MCQ
    if is_true_false(item):
        return FORMAT_TRUE_FALSE
    if is_multi_blank(item):
        return FORMAT_MULTI_BLANK
    if is_multi_select(item):
        return FORMAT_MULTI_SELECT
    return FORMAT_FREE_RESPONSE


def prompt_family_for(format_type: str, subject: str) -> str:
    if format_type == FORMAT_ALGORITHM_SEQUENCE:
        return PROMPT_ALGORITHM_SEQUENCE
    if format_type in {FORMAT_MCQ, FORMAT_MULTI_SELECT}:
        return PROMPT_GENERAL_MCQ
    if format_type in {FORMAT_MULTI_BLANK, FORMAT_TRUE_FALSE}:
        return PROMPT_MULTI_BLANK
    if subject == "statistics":
        return PROMPT_STATISTICS
    if subject == "calculus":
        return PROMPT_CALCULUS
    return PROMPT_BASELINE


def route_item(item: dict[str, Any]) -> Route:
    format_type = classify_format(item)
    subject = detect_subject(item) if format_type == FORMAT_FREE_RESPONSE else "general"
    prompt_family = prompt_family_for(format_type, subject)
    notes: list[str] = []
    if item.get("_embedded_options"):
        notes.append("embedded_options")
    if is_multi_select(item):
        notes.append("multi_select")
    if is_true_false(item):
        notes.append("true_false")

    return Route(
        format_type=format_type,
        prompt_family=prompt_family,
        config_name=PROMPT_CONFIGS[prompt_family],
        expected_answers=expected_answer_count(item),
        subject=subject,
        has_options=bool(item.get("options")),
        notes=tuple(notes),
    )


def visible_response_text(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>") :] if think_end >= 0 else text


def extract_boxed_values(text: str) -> list[str]:
    search_text = visible_response_text(text)
    values: list[str] = []
    start = 0
    while True:
        idx = search_text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        pos = brace_start
        while pos < len(search_text) and depth > 0:
            if search_text[pos] == "{":
                depth += 1
            elif search_text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            values.append(search_text[brace_start : pos - 1].strip())
        start = max(pos, idx + 1)
    return values


def normalize_choice_letters(candidate: str, option_count: int | None = None) -> str:
    valid = {chr(65 + idx) for idx in range(option_count)} if option_count else set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    upper = candidate.upper()
    upper = re.sub(r"\\BOXED|BOXED|FINAL_ANSWERS|FINAL ANSWERS|FINAL ANSWER", " ", upper)
    upper = re.sub(
        r"\b(?:AND|OR|THE|IS|ARE|ANSWER|ANSWERS|OPTION|OPTIONS|CHOICE|CHOICES|"
        r"CORRECT|SELECT|ALL|THAT|APPLY)\b",
        " ",
        upper,
    )
    letters = re.findall(r"[A-Z]", upper)

    out: list[str] = []
    for letter in letters:
        if letter in valid and letter not in out:
            out.append(letter)
    return "".join(out)


def _normalize_option_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^\s*[A-Z][\.\):]\s*", "", text)
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.strip("$ \n\t")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _match_option_value(candidate: str, item: dict[str, Any], judger: Any | None = None) -> str:
    options = item.get("options") or []
    if not options:
        return ""

    normalized_candidate = _normalize_option_text(candidate)
    for idx, option in enumerate(options):
        if normalized_candidate == _normalize_option_text(str(option)):
            return chr(65 + idx)
        if judger is not None:
            try:
                if judger.is_equal(judger.norm_ans_str(candidate), judger.norm_ans_str(str(option))):
                    return chr(65 + idx)
            except Exception:
                pass

    return ""


def extract_choice_prediction(
    response: str,
    item: dict[str, Any],
    multi: bool = False,
    judger: Any | None = None,
) -> str:
    option_count = len(item.get("options") or []) or None
    boxes = extract_boxed_values(response)
    candidates = list(reversed(boxes))
    visible = visible_response_text(response)
    candidates.extend(reversed([line.strip() for line in visible.splitlines() if line.strip()]))

    if multi:
        for candidate in candidates:
            letters = normalize_choice_letters(candidate, option_count)
            if len(letters) > 1:
                return letters
        return ""

    valid = {chr(65 + idx) for idx in range(option_count or 26)}
    for candidate in candidates:
        stripped = candidate.strip().upper()
        match = re.fullmatch(r"([A-Z])", stripped)
        if match and match.group(1) in valid:
            return match.group(1)
        leading = re.match(r"(?:OPTION|CHOICE|ANSWER)?\s*[:\-]?\s*\(?([A-Z])\)?\b", stripped)
        if leading and leading.group(1) in valid:
            return leading.group(1)
        matched = _match_option_value(candidate, item, judger)
        if matched:
            return matched

    phrase_matches = re.findall(
        r"\b(?:OPTION|CHOICE|ANSWER)\s*(?:IS|:)?\s*\(?([A-Z])\)?\b",
        visible.upper(),
    )
    phrase_matches = [letter for letter in phrase_matches if letter in valid]
    return phrase_matches[-1] if phrase_matches else ""


def _split_answer_string(answer: str, route: Route, judger: Any | None) -> list[str]:
    answer = str(answer).strip()
    if not answer:
        return []
    if route.expected_answers <= 1:
        return [answer]

    if route.format_type == FORMAT_TRUE_FALSE:
        compact = re.sub(r"[^A-Za-z]", "", answer).upper()
        if len(compact) == route.expected_answers and set(compact) <= {"T", "F"}:
            return list(compact)

    parts: list[str]
    if judger is not None:
        try:
            parts = [part.strip() for part in judger.split_by_comma(answer)]
        except Exception:
            parts = [part.strip() for part in answer.split(",")]
    else:
        parts = [part.strip() for part in answer.split(",")]

    if len(parts) == route.expected_answers:
        return parts
    return [answer]


def final_answer_block(values: list[str]) -> str:
    clean_values = [str(value).strip() for value in values if str(value).strip()]
    if not clean_values:
        return ""
    return "FINAL_ANSWERS:\n" + "\n".join(f"\\boxed{{{value}}}" for value in clean_values)


def clean_final_response(
    response: str,
    item: dict[str, Any],
    route: Route,
    judger: Any | None = None,
) -> str:
    """Return a compact final-answer block suitable for submission."""
    if route.format_type in {FORMAT_MCQ, FORMAT_ALGORITHM_SEQUENCE, FORMAT_MULTI_SELECT}:
        letters = extract_choice_prediction(
            response,
            item,
            multi=route.format_type == FORMAT_MULTI_SELECT,
            judger=judger,
        )
        if letters:
            return final_answer_block([letters])

    if judger is not None:
        try:
            boxes = judger.extract_all_boxed(visible_response_text(response))
        except Exception:
            boxes = extract_boxed_values(response)
    else:
        boxes = extract_boxed_values(response)
    if boxes:
        if len(boxes) == 1:
            return final_answer_block(_split_answer_string(boxes[0], route, judger))
        return final_answer_block(boxes)

    if judger is not None:
        try:
            extracted = judger.extract_ans(response)
        except Exception:
            extracted = ""
        if extracted:
            return final_answer_block(_split_answer_string(extracted, route, judger))

    return ""
