import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PATCH_STATUSES = {
    "candidate",
    "static_rejected",
    "static_valid",
    "validated",
    "shadow_rejected",
    "active_guarded",
    "disabled",
    "accepted",
    "archived",
    "consolidated",
    "pruned",
    "merged",
    "promoted",
}
DETECTOR_TYPES = {"rule", "verifier", "llm_judge", "composite"}
FIXER_TYPES = {
    "existing_operator",
    "prompt_patch",
    "mini_workflow",
    "tool_call",
    "contract_repair",
    "answer_normalize",
    "llm_repair",
}
ONLINE_FORBIDDEN_SIGNALS = {"gold_answer", "gold_answer_correctness", "expected", "label", "ground_truth"}


def default_telemetry() -> Dict[str, Any]:
    return {
        "checked_count": 0,
        "activation_count": 0,
        "accepted_count": 0,
        "rollback_count": 0,
        "repair_success_count": 0,
        "regression_count": 0,
        "average_gain": 0.0,
        "average_cost_overhead": 0.0,
        "average_latency_overhead": 0.0,
    }


def make_patch_spec(
    patch_id: str,
    source_trace_id: str,
    target_operator: str,
    diagnosed_bottleneck: str,
    failure_symptom: str,
    trigger: Dict[str, Any],
    fixer: Dict[str, Any],
    scope: Dict[str, Any],
    contract: Dict[str, Any],
    rollback: Dict[str, Any] = None,
    status: str = "candidate",
) -> Dict[str, Any]:
    return {
        "patch_id": patch_id,
        "source_trace_id": source_trace_id,
        "target_operator": target_operator,
        "diagnosed_bottleneck": diagnosed_bottleneck,
        "failure_symptom": failure_symptom,
        "trigger": trigger,
        "fixer": fixer,
        "scope": scope,
        "contract": contract,
        "rollback": rollback or {"strategy": "return_original", "fallback": None},
        "telemetry": default_telemetry(),
        "status": status,
    }


def validate_patch_schema(patch: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    required = [
        "patch_id",
        "source_trace_id",
        "target_operator",
        "diagnosed_bottleneck",
        "failure_symptom",
        "trigger",
        "fixer",
        "scope",
        "contract",
        "rollback",
        "telemetry",
        "status",
    ]
    for key in required:
        if key not in patch:
            errors.append(f"Missing field: {key}")
    if errors:
        return False, errors

    if patch["status"] not in PATCH_STATUSES:
        errors.append(f"Invalid status: {patch['status']}")
    detector_type = patch["trigger"].get("detector_type") or patch["trigger"].get("detector")
    if detector_type not in DETECTOR_TYPES:
        errors.append(f"Invalid detector_type: {detector_type}")
    if patch["fixer"].get("type") not in FIXER_TYPES:
        errors.append(f"Invalid fixer type: {patch['fixer'].get('type')}")
    signals = patch["trigger"].get("observable_signals", [])
    decision_rule = str(patch["trigger"].get("decision_rule", ""))
    for forbidden in ONLINE_FORBIDDEN_SIGNALS:
        if forbidden in signals or forbidden in decision_rule:
            errors.append(f"Trigger uses unavailable online signal: {forbidden}")
    return not errors, errors


def validate_trigger_signals_available(patch: Dict[str, Any], trace_ir) -> Tuple[bool, List[str]]:
    available = set()
    for node in getattr(trace_ir, "nodes", []):
        available.update(node.operator.metadata.keys())
        available.update({"operator_output", "operator_input", "error_message", "verifier_score", "confidence"})
    available.update({"schema_valid", "output_empty", "tool_error", "parse_status", "error_message"})
    missing = [signal for signal in patch.get("trigger", {}).get("observable_signals", []) if signal not in available]
    return not missing, [f"Unavailable trigger signal: {signal}" for signal in missing]


def validate_fixer_interface(patch: Dict[str, Any], workflow: Any = None) -> Tuple[bool, List[str]]:
    fixer = patch.get("fixer", {})
    errors = []
    if fixer.get("type") == "existing_operator" and not fixer.get("name"):
        errors.append("existing_operator fixer requires a name")
    if fixer.get("type") in {"prompt_patch", "mini_workflow", "contract_repair"} and "definition" not in fixer:
        errors.append(f"{fixer.get('type')} fixer requires a definition")
    if workflow is not None and fixer.get("type") == "existing_operator":
        name = fixer.get("name")
        if isinstance(workflow, dict):
            operators = workflow.get("operators", {})
            if operators and name not in operators:
                errors.append(f"Fixer operator not available in workflow: {name}")
        elif name and not hasattr(workflow, name):
            errors.append(f"Fixer operator not available in workflow: {name}")
    return not errors, errors


def validate_contract_definition(patch: Dict[str, Any]) -> Tuple[bool, List[str]]:
    contract = patch.get("contract", {})
    errors = []
    for key in ("input_requirements", "output_requirements", "downstream_compatibility"):
        if key not in contract:
            errors.append(f"Contract missing {key}")
        elif not isinstance(contract[key], list):
            errors.append(f"Contract field must be a list: {key}")
    return not errors, errors


def save_patch(patch: Dict[str, Any], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(patch, fout, ensure_ascii=False, indent=2)


def load_patch(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fin:
        return json.load(fin)


def clone_patch(patch: Dict[str, Any]) -> Dict[str, Any]:
    cloned = deepcopy(patch)
    cloned.setdefault("telemetry", default_telemetry())
    return cloned


def referenced_prompt_names(graph_text: str) -> List[str]:
    return sorted(set(re.findall(r"prompt_custom\.([A-Za-z_][A-Za-z0-9_]*)", graph_text)))
