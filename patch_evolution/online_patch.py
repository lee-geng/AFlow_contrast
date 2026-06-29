import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from patch_evolution.patch_spec import make_patch_spec


ANSWER_FIELDS = ["answer", "response", "solution", "output"]
NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]


def parse_preview(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def select_answer_target(record: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    operators = record.get("operator_traces") or []
    for operator in reversed(operators):
        value = parse_preview(operator.get("output_preview"))
        if isinstance(value, dict):
            fields = [field for field in ANSWER_FIELDS if field in value and isinstance(value[field], str)]
            if fields:
                return operator, fields
    if operators:
        return operators[-1], list(ANSWER_FIELDS)
    return None, list(ANSWER_FIELDS)


def infer_answer_style_transforms(record: Dict[str, Any], operator: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    observed_text = _observed_text(record, operator)
    transforms: List[str] = []
    rules: List[str] = []

    if "```" in observed_text:
        transforms.append("strip_code_fence")
        rules.append("answer_text contains ```")
    if "\\boxed" in observed_text or "boxed{" in observed_text.lower():
        transforms.append("strip_boxed")
        rules.append("answer_text contains \\boxed")
    if re.search(r"\b(?:the\s+)?answer\s+(?:is|:)", observed_text, flags=re.IGNORECASE):
        transforms.append("strip_answer_prefix")
        rules.append("answer_text icontains answer is")
    if re.search(r"\bfinal\s+answer\s*(?:is|:)", observed_text, flags=re.IGNORECASE):
        transforms.append("strip_final_answer_prefix")
        rules.append("answer_text icontains final answer")

    expected = str(record.get("expected") or "")
    if _looks_numeric(expected):
        number_word = _first_number_word(observed_text)
        if number_word:
            transforms.append("number_words_to_digits")
            rules.append(f"answer_text icontains {number_word}")

    if transforms and "trim_terminal_punctuation" not in transforms:
        transforms.append("trim_terminal_punctuation")

    return _dedupe(transforms), _dedupe(rules)


def build_answer_style_patch(record: Dict[str, Any], diagnosis: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    operator, fields = select_answer_target(record)
    if not operator:
        return None

    transforms, rules = infer_answer_style_transforms(record, operator)
    if not transforms or not rules:
        return None

    target_operator = operator.get("operator_type") or operator.get("operator_id") or "workflow_output"
    dataset = str(record.get("dataset") or "task")
    sample_id = str(record.get("sample_id") or "sample")
    patch_key = json.dumps(
        {
            "dataset": dataset,
            "target_operator": target_operator,
            "fields": fields,
            "transforms": transforms,
            "rules": rules,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(patch_key.encode("utf-8")).hexdigest()[:10]
    patch_id = _safe_id(f"online_{dataset}_{target_operator}_{digest}")

    if len(rules) == 1:
        trigger = {
            "observable_signals": ["output"],
            "decision_rule": rules[0],
            "threshold": 0.5,
            "detector_type": "rule",
        }
    else:
        trigger = {
            "observable_signals": ["output"],
            "decision_rule": "",
            "rules": rules,
            "threshold": 0.5,
            "detector_type": "composite",
        }

    field_requirements = [f"field:{field}" for field in fields[:1]] if fields else []
    return make_patch_spec(
        patch_id=patch_id,
        source_trace_id=sample_id,
        target_operator=target_operator,
        diagnosed_bottleneck=(diagnosis or {}).get("suspected_bottleneck", "answer_style_or_extraction"),
        failure_symptom=(diagnosis or {}).get("failure_symptom", "answer style differs from scorer expectation"),
        trigger=trigger,
        fixer={
            "type": "contract_repair",
            "name": "answer_style_transform",
            "definition": "apply observable answer-style transforms to answer fields",
            "fields": fields,
            "transforms": transforms,
            "input_mapping": {},
            "output_mapping": {},
        },
        scope={
            "task_type": dataset,
            "operator": target_operator,
            "conditions": ["observable answer style trigger fires"],
        },
        contract={
            "input_requirements": [],
            "output_requirements": ["non_empty", *field_requirements],
            "downstream_compatibility": ["preserve original output container and non-answer fields"],
        },
        status="candidate",
    )


def build_numeric_normalization_patch(record: Dict[str, Any], diagnosis: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    operator, fields = select_answer_target(record)
    if not operator:
        return None
    observed = _observed_text(record, operator).strip()
    transforms: List[str] = []
    rules: List[str] = []

    if re.search(r"\b-?\d+(?:\.\d+)?%", observed):
        transforms.append("strip_percent")
        rules.append("answer_text contains %")
    if re.search(r"\b-?\d+\.\d*0\b", observed):
        transforms.append("trim_numeric_trailing_zeros")
        rules.append("answer_text contains .")
    if re.search(r"\b0\.\d+\b", observed):
        transforms.append("strip_leading_zero_decimal")
        rules.append("answer_text contains 0.")
    if re.search(r"\b-?\d+(?:\.\d+)?\s+million\b", observed, flags=re.IGNORECASE):
        transforms.append("million_to_number")
        rules.append("answer_text icontains million")

    transforms = _dedupe([*transforms, "trim_terminal_punctuation"])
    rules = _dedupe(rules)
    if not rules or len(transforms) == 1:
        return None
    return _build_contract_patch(
        record=record,
        diagnosis=diagnosis,
        operator=operator,
        fields=fields,
        transforms=transforms,
        rules=rules,
        patch_kind="numeric_normalization",
        symptom="numeric answer formatting differs from scorer expectation",
    )


def build_missing_answer_llm_patch(record: Dict[str, Any], diagnosis: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    operator = select_missing_answer_operator(record)
    if not operator:
        return None
    target_operator = operator.get("operator_type") or operator.get("operator_id") or "workflow_output"
    if target_operator != "AnswerGenerate":
        return None
    return _build_llm_patch(
        record=record,
        diagnosis=diagnosis,
        operator=operator,
        fields=["answer"],
        patch_kind="missing_answer_field_llm",
        trigger={
            "observable_signals": ["output_fields"],
            "decision_rule": "field_missing answer",
            "threshold": 0.5,
            "detector_type": "rule",
        },
        definition=(
            "The operator output is missing the answer field. Read the task prompt and the thought field, "
            "then return the shortest final answer that should populate the answer field."
        ),
        contract_requirements=["non_empty", "field:answer"],
    )


def build_concise_answer_llm_patch(record: Dict[str, Any], diagnosis: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    operator, fields = select_answer_target(record)
    if not operator or not fields:
        return None
    target_operator = operator.get("operator_type") or operator.get("operator_id") or "workflow_output"
    if target_operator != "AnswerGenerate":
        return None
    parsed = parse_preview(operator.get("output_preview"))
    if not isinstance(parsed, dict) or not any(field in parsed for field in fields):
        return None
    answer_text = _observed_answer_field(operator, fields)
    if len(answer_text.split()) < 6:
        return None
    return _build_llm_patch(
        record=record,
        diagnosis=diagnosis,
        operator=operator,
        fields=fields[:1],
        patch_kind="concise_answer_llm",
        trigger={
            "observable_signals": ["answer_word_count"],
            "decision_rule": "answer_word_count >= 6",
            "threshold": 0.5,
            "detector_type": "rule",
        },
        definition=(
            "Rewrite the answer field into the shortest DROP-style final answer. Keep only the entity, "
            "number, date, duration, or option requested by the question. Do not include explanations."
        ),
        contract_requirements=["non_empty", f"field:{fields[0]}"],
    )


def build_online_patch_candidates(
    record: Dict[str, Any],
    diagnosis: Dict[str, Any] = None,
    enable_llm_repair: bool = True,
) -> List[Dict[str, Any]]:
    builders = [build_answer_style_patch, build_numeric_normalization_patch]
    if enable_llm_repair:
        builders = [build_missing_answer_llm_patch, *builders, build_concise_answer_llm_patch]
    candidates = []
    seen = set()
    for builder in builders:
        patch = builder(record, diagnosis=diagnosis)
        if patch and patch["patch_id"] not in seen:
            candidates.append(patch)
            seen.add(patch["patch_id"])
    return candidates


def select_missing_answer_operator(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    operators = record.get("operator_traces") or []
    prediction = str(record.get("prediction") or "")
    for operator in reversed(operators):
        if operator.get("operator_type") != "AnswerGenerate":
            continue
        value = parse_preview(operator.get("output_preview"))
        if isinstance(value, dict) and "answer" not in value:
            return operator
    if prediction.strip() in {"'answer'", '"answer"', "KeyError('answer')", "KeyError: 'answer'"}:
        for operator in reversed(operators):
            if operator.get("operator_type") == "AnswerGenerate":
                return operator
    return None


def _build_contract_patch(
    record: Dict[str, Any],
    diagnosis: Dict[str, Any],
    operator: Dict[str, Any],
    fields: List[str],
    transforms: List[str],
    rules: List[str],
    patch_kind: str,
    symptom: str,
) -> Dict[str, Any]:
    target_operator = operator.get("operator_type") or operator.get("operator_id") or "workflow_output"
    dataset = str(record.get("dataset") or "task")
    sample_id = str(record.get("sample_id") or "sample")
    patch_id = _patch_id(dataset, target_operator, patch_kind, {"fields": fields, "transforms": transforms, "rules": rules})
    trigger = _rules_trigger(rules)
    field_requirements = [f"field:{field}" for field in fields[:1]] if fields else []
    return make_patch_spec(
        patch_id=patch_id,
        source_trace_id=sample_id,
        target_operator=target_operator,
        diagnosed_bottleneck=(diagnosis or {}).get("suspected_bottleneck", patch_kind),
        failure_symptom=symptom,
        trigger=trigger,
        fixer={
            "type": "contract_repair",
            "name": patch_kind,
            "definition": f"apply {patch_kind} transforms to answer fields",
            "fields": fields,
            "transforms": transforms,
            "input_mapping": {},
            "output_mapping": {},
        },
        scope={"task_type": dataset, "operator": target_operator, "conditions": [symptom]},
        contract={
            "input_requirements": [],
            "output_requirements": ["non_empty", *field_requirements],
            "downstream_compatibility": ["preserve original output container and non-answer fields"],
        },
        status="candidate",
    )


def _build_llm_patch(
    record: Dict[str, Any],
    diagnosis: Dict[str, Any],
    operator: Dict[str, Any],
    fields: List[str],
    patch_kind: str,
    trigger: Dict[str, Any],
    definition: str,
    contract_requirements: List[str],
) -> Dict[str, Any]:
    target_operator = operator.get("operator_type") or operator.get("operator_id") or "workflow_output"
    dataset = str(record.get("dataset") or "task")
    sample_id = str(record.get("sample_id") or "sample")
    patch_id = _patch_id(dataset, target_operator, patch_kind, {"fields": fields, "trigger": trigger})
    return make_patch_spec(
        patch_id=patch_id,
        source_trace_id=sample_id,
        target_operator=target_operator,
        diagnosed_bottleneck=(diagnosis or {}).get("suspected_bottleneck", patch_kind),
        failure_symptom=(diagnosis or {}).get("failure_symptom", patch_kind),
        trigger=trigger,
        fixer={
            "type": "llm_repair",
            "name": patch_kind,
            "definition": definition,
            "fields": fields,
            "max_input_chars": 2600,
            "max_output_chars": 1200,
            "input_mapping": {},
            "output_mapping": {},
        },
        scope={"task_type": dataset, "operator": target_operator, "conditions": [patch_kind]},
        contract={
            "input_requirements": [],
            "output_requirements": contract_requirements,
            "downstream_compatibility": ["preserve output container and repair only final answer fields"],
        },
        status="candidate",
    )


def _rules_trigger(rules: List[str]) -> Dict[str, Any]:
    if len(rules) == 1:
        return {
            "observable_signals": ["answer_text"],
            "decision_rule": rules[0],
            "threshold": 0.5,
            "detector_type": "rule",
        }
    return {
        "observable_signals": ["answer_text"],
        "decision_rule": "",
        "rules": rules,
        "match": "any",
        "threshold": 0.5,
        "detector_type": "composite",
    }


def _observed_text(record: Dict[str, Any], operator: Dict[str, Any]) -> str:
    values = [record.get("prediction"), operator.get("output_preview")]
    parsed = parse_preview(operator.get("output_preview"))
    if isinstance(parsed, dict):
        values.extend(parsed.get(field) for field in ANSWER_FIELDS)
    return "\n".join(str(value) for value in values if value is not None)


def _observed_answer_field(operator: Dict[str, Any], fields: List[str]) -> str:
    parsed = parse_preview(operator.get("output_preview"))
    if not isinstance(parsed, dict):
        return str(operator.get("output_preview") or "")
    for field in fields:
        value = parsed.get(field)
        if isinstance(value, str):
            return value
    return ""


def _first_number_word(text: str) -> Optional[str]:
    lowered = text.lower()
    for word in NUMBER_WORDS:
        if re.search(rf"\b{word}\b", lowered):
            return word
    return None


def _looks_numeric(text: str) -> bool:
    stripped = text.strip()
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped))


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _patch_id(dataset: str, target_operator: str, patch_kind: str, payload: Dict[str, Any]) -> str:
    patch_key = json.dumps(
        {"dataset": dataset, "target_operator": target_operator, "patch_kind": patch_kind, **payload},
        sort_keys=True,
    )
    digest = hashlib.sha1(patch_key.encode("utf-8")).hexdigest()[:10]
    return _safe_id(f"online_{dataset}_{target_operator}_{patch_kind}_{digest}")
