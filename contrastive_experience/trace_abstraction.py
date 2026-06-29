import json
import re
from typing import Any, Dict, Optional


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def classify_schema_type(value: Any) -> str:
    if value is None:
        return "empty"
    if isinstance(value, dict):
        return "json_dict"
    if isinstance(value, list):
        return "json_list"

    text = stringify(value).strip()
    if not text:
        return "empty"
    if re.search(r"```[\s\S]*?```", text):
        return "code_block"
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return "invalid_json"
        return "json_dict" if isinstance(parsed, dict) else "json_list" if isinstance(parsed, list) else "unknown"
    return "free_text"


def classify_output_style(value: Any) -> str:
    text = stringify(value).strip()
    if not text:
        return "empty"
    if classify_schema_type(value) in {"json_dict", "json_list"}:
        return "json_answer"
    if re.search(r"```[\s\S]*?```", text) or re.search(r"\bdef\s+\w+\s*\(", text):
        return "code"
    if "\\boxed" in text or re.search(r"\bboxed\s*{", text):
        return "boxed_answer"
    if re.search(r"\b(and|or)\b|[,;/]", text) and len(text.split()) <= 12:
        return "multi_answer"
    if re.search(r"[.!?]\s*$", text) or len(re.split(r"\s+", text)) > 12:
        return "full_sentence"
    word_count = len(re.findall(r"\S+", text))
    if word_count <= 3:
        return "concise_entity"
    if word_count <= 8:
        return "short_phrase"
    return "unknown"


def infer_answer_type(value: Any) -> str:
    text = stringify(value).strip()
    if not text:
        return "unknown"
    if classify_output_style(value) == "code":
        return "code"
    lowered = text.lower()
    if lowered in {"yes", "no", "true", "false"}:
        return "yes_no"
    if re.fullmatch(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", text):
        return "number"
    if re.fullmatch(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", text) or re.search(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", lowered
    ):
        return "date"
    if re.fullmatch(r"[A-Z][A-Za-z0-9&.' -]{1,80}", text):
        return "entity"
    if len(text.split()) <= 8:
        return "phrase"
    return "unknown"


def infer_parse_status(output: Any, error_type: Optional[str] = None) -> str:
    if error_type:
        lowered = error_type.lower()
        if "timeout" in lowered:
            return "timeout"
        if "json" in lowered:
            return "invalid_json"
        return "failed"
    schema_type = classify_schema_type(output)
    if schema_type == "empty":
        return "empty"
    if schema_type == "invalid_json":
        return "invalid_json"
    return "ok"


def output_length(value: Any) -> int:
    return len(stringify(value).split())


def abstract_value(value: Any, error_type: Optional[str] = None) -> Dict[str, Any]:
    return {
        "schema_type": classify_schema_type(value),
        "output_style": classify_output_style(value),
        "answer_type": infer_answer_type(value),
        "output_length": output_length(value),
        "parse_status": infer_parse_status(value, error_type),
    }
