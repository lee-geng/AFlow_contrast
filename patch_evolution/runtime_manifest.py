from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TriggerSpec:
    type: str
    field: Optional[str] = None
    value: Any = None
    value_type: Optional[str] = None


@dataclass
class OperatorManifest:
    name: str
    input_fields: List[str]
    output_fields: List[str]
    observable_fields: List[str]
    patchable_fields: List[str]
    allowed_triggers: List[TriggerSpec]
    allowed_fixers: List[str]
    forbidden_writes: List[str]

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "allowed_triggers": [asdict(trigger) for trigger in self.allowed_triggers],
        }


TRIGGER_REGISTRY = {
    "field_missing": "Field is absent from output_fields.",
    "field_present": "Field is present in output_fields.",
    "bool_equals": "Boolean runtime field equals a provided boolean value.",
    "enum_equals": "Runtime enum/string field equals one of its executable enum values.",
    "text_empty": "Text runtime field is empty after stripping whitespace.",
    "int_gte": "Integer runtime field is greater than or equal to a threshold.",
    "contains": "Text runtime field contains a case-sensitive substring.",
    "icontains": "Text runtime field contains a case-insensitive substring.",
}

FIXER_REGISTRY = {
    "contract_repair": {
        "description": "Repair output contract shape and required patchable fields only.",
        "requires_writes": True,
    },
    "answer_normalize": {
        "description": "Normalize an existing answer/raw response into a concise final answer.",
        "requires_writes": True,
    },
    "llm_repair": {
        "description": "Restricted local extraction/repair using only operator input and original output.",
        "requires_writes": True,
    },
}

FORBIDDEN_SIGNALS = [
    "verifier_score",
    "confidence",
    "uncertainty",
    "probability",
    "reward",
    "gold_answer",
    "expected_label",
    "evaluator_feedback",
]

ANSWER_REPAIR_FIXERS = {"answer_normalize", "llm_repair"}


ANSWER_TRIGGER_SPECS = [
    TriggerSpec("field_missing", field="answer"),
    TriggerSpec("field_present", field="answer"),
    TriggerSpec("bool_equals", field="output_empty", value=True, value_type="bool"),
    TriggerSpec("bool_equals", field="schema_valid", value=False, value_type="bool"),
    TriggerSpec("enum_equals", field="parse_status", value="failed", value_type="str"),
    TriggerSpec("text_empty", field="answer_text"),
    TriggerSpec("int_gte", field="answer_word_count", value_type="int"),
    TriggerSpec("contains", field="answer_text", value_type="str"),
    TriggerSpec("icontains", field="answer_text", value_type="str"),
]

DROP_OPERATOR_MANIFESTS = {
    "AnswerGenerate": OperatorManifest(
        name="AnswerGenerate",
        input_fields=["question", "context"],
        output_fields=["answer", "raw_response"],
        observable_fields=[
            "answer",
            "answer_text",
            "raw_response",
            "output_text",
            "output_empty",
            "schema_valid",
            "parse_status",
            "answer_word_count",
        ],
        patchable_fields=["answer"],
        allowed_triggers=ANSWER_TRIGGER_SPECS,
        allowed_fixers=["contract_repair", "answer_normalize", "llm_repair"],
        forbidden_writes=[
            "question",
            "context",
            "raw_response",
            "gold_answer",
            "expected_label",
            "verifier_score",
            "confidence",
        ],
    ),
    "FinalFormat": OperatorManifest(
        name="FinalFormat",
        input_fields=["answer"],
        output_fields=["answer"],
        observable_fields=[
            "answer",
            "answer_text",
            "output_text",
            "output_empty",
            "schema_valid",
            "parse_status",
            "answer_word_count",
        ],
        patchable_fields=["answer"],
        allowed_triggers=ANSWER_TRIGGER_SPECS,
        allowed_fixers=["contract_repair", "answer_normalize", "llm_repair"],
        forbidden_writes=[
            "question",
            "context",
            "raw_response",
            "gold_answer",
            "expected_label",
            "verifier_score",
            "confidence",
        ],
    ),
    "ScEnsemble": OperatorManifest(
        name="ScEnsemble",
        input_fields=["candidate_solutions"],
        output_fields=["solution_letter", "thought"],
        observable_fields=[
            "solution_letter",
            "thought",
            "output_empty",
            "schema_valid",
            "parse_status",
        ],
        patchable_fields=["solution_letter"],
        allowed_triggers=[
            TriggerSpec("field_missing", field="solution_letter"),
            TriggerSpec("field_present", field="solution_letter"),
            TriggerSpec("bool_equals", field="output_empty", value=True, value_type="bool"),
            TriggerSpec("bool_equals", field="schema_valid", value=False, value_type="bool"),
            TriggerSpec("enum_equals", field="parse_status", value="failed", value_type="str"),
        ],
        allowed_fixers=["contract_repair"],
        forbidden_writes=[
            "answer",
            "gold_answer",
            "expected_label",
            "question",
            "context",
            "thought",
            "verifier_score",
            "confidence",
        ],
    ),
}

DROP_PATCH_PROFILE = {
    "allowed_target_operators": ["AnswerGenerate", "FinalFormat"],
    "allowed_trigger_types": list(TRIGGER_REGISTRY.keys()),
    "allowed_fixers": ["contract_repair", "answer_normalize", "llm_repair"],
    "forbidden_signals": FORBIDDEN_SIGNALS,
}


def get_operator_manifests(dataset: str) -> Dict[str, OperatorManifest]:
    if str(dataset).upper() == "DROP":
        return DROP_OPERATOR_MANIFESTS
    return DROP_OPERATOR_MANIFESTS


def get_operator_manifest(dataset: str, operator_name: str) -> Optional[OperatorManifest]:
    return get_operator_manifests(dataset).get(operator_name)


def get_patch_profile(dataset: str) -> Dict[str, Any]:
    if str(dataset).upper() == "DROP":
        return DROP_PATCH_PROFILE
    return DROP_PATCH_PROFILE


def prompt_manifest_payload(dataset: str) -> Dict[str, Any]:
    profile = get_patch_profile(dataset)
    manifests = get_operator_manifests(dataset)
    allowed_targets = profile.get("allowed_target_operators") or list(manifests)
    return {
        "task_family": dataset,
        "trigger_registry": TRIGGER_REGISTRY,
        "fixer_registry": FIXER_REGISTRY,
        "task_patch_profile": profile,
        "operator_manifests": [
            manifests[name].to_prompt_dict()
            for name in allowed_targets
            if name in manifests
        ],
    }
