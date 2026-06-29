import operator
import re
from typing import Any, Dict, Tuple


OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


class TriggerDetector:
    def __init__(self, llm_judge=None, enable_llm_judge: bool = False):
        self.llm_judge = llm_judge
        self.enable_llm_judge = enable_llm_judge

    def evaluate(self, trace_state: Dict[str, Any], patch_spec: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        trigger = patch_spec.get("trigger", {})
        detector_type = trigger.get("detector") or trigger.get("detector_type", "rule")
        if detector_type == "composite":
            return self._evaluate_composite(trace_state, trigger)
        if detector_type == "llm_judge" and self.enable_llm_judge and self.llm_judge:
            return float(self.llm_judge(trace_state, patch_spec)), {"path": "llm_judge"}
        if detector_type == "verifier":
            return self._evaluate_verifier(trace_state, trigger)
        return self._evaluate_rule(trace_state, trigger)

    def should_trigger(self, trace_state: Dict[str, Any], patch_spec: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        score, info = self.evaluate(trace_state, patch_spec)
        threshold = patch_spec.get("trigger", {}).get("threshold")
        if threshold is None:
            threshold = 0.5
        return score > threshold, score, info

    def _evaluate_rule(self, trace_state: Dict[str, Any], trigger: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        if trigger.get("type"):
            score = 1.0 if self._primitive_matches(trace_state, trigger) else 0.0
            return score, {"path": "primitive", "trigger": trigger}
        rule = str(trigger.get("decision_rule", "")).strip()
        if not rule:
            return 0.0, {"path": "rule", "reason": "empty_rule"}
        score = 1.0 if self._rule_matches(trace_state, rule) else 0.0
        return score, {"path": "rule", "rule": rule}

    def _evaluate_verifier(self, trace_state: Dict[str, Any], trigger: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        threshold = trigger.get("threshold")
        verifier_score = trace_state.get("verifier_score")
        if verifier_score is None:
            return self._evaluate_rule(trace_state, trigger)
        if threshold is None:
            threshold = 0.5
        return (1.0 if verifier_score < threshold else 0.0), {"path": "verifier", "verifier_score": verifier_score}

    def _evaluate_composite(self, trace_state: Dict[str, Any], trigger: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        scores = []
        for rule in trigger.get("rules", []):
            if isinstance(rule, dict):
                score, _ = self._evaluate_rule(trace_state, rule)
            else:
                score, _ = self._evaluate_rule(trace_state, {"decision_rule": rule})
            scores.append(score)
        if not scores:
            return self._evaluate_rule(trace_state, trigger)
        match_mode = str(trigger.get("op") or trigger.get("match") or "any").lower()
        if match_mode in {"all", "all_of", "and"}:
            score = 1.0 if all(value > 0 for value in scores) else 0.0
        else:
            score = max(scores)
        return score, {"path": "composite", "scores": scores, "match": match_mode}

    def _primitive_matches(self, trace_state: Dict[str, Any], trigger: Dict[str, Any]) -> bool:
        trigger_type = trigger.get("type")
        field = trigger.get("field")
        if trigger_type == "field_missing":
            return field not in trace_state.get("output_fields", [])
        if trigger_type == "field_present":
            return field in trace_state.get("output_fields", [])
        if trigger_type == "bool_equals":
            return self._value(trace_state, field) is bool(trigger.get("value"))
        if trigger_type == "enum_equals":
            return str(self._value(trace_state, field)) == str(trigger.get("value"))
        if trigger_type == "text_empty":
            return str(self._value(trace_state, field) or "").strip() == ""
        if trigger_type == "int_gte":
            try:
                left = int(self._value(trace_state, field) or 0)
                right = int(trigger.get("value"))
                return left >= right
            except (TypeError, ValueError):
                return False
        if trigger_type == "contains":
            return str(trigger.get("value") or "") in str(self._value(trace_state, field) or "")
        if trigger_type == "icontains":
            return str(trigger.get("value") or "").lower() in str(self._value(trace_state, field) or "").lower()
        return False

    def _rule_matches(self, trace_state: Dict[str, Any], rule: str) -> bool:
        lowered = rule.lower()
        if lowered in {"schema_valid == false", "not schema_valid"}:
            return trace_state.get("schema_valid") is False
        if lowered in {"tool_error == true", "tool_error"}:
            return bool(trace_state.get("tool_error"))
        if lowered in {"output_empty == true", "output_empty"}:
            if "output_empty" in trace_state:
                return bool(trace_state.get("output_empty"))
            output = trace_state.get("output")
            return output is None or output == "" or output == {} or output == []
        field_missing_match = re.match(r"field_missing\s+([A-Za-z_][A-Za-z0-9_]*)", rule, re.IGNORECASE)
        if field_missing_match:
            field = field_missing_match.group(1)
            return field not in trace_state.get("output_fields", [])
        field_present_match = re.match(r"field_present\s+([A-Za-z_][A-Za-z0-9_]*)", rule, re.IGNORECASE)
        if field_present_match:
            field = field_present_match.group(1)
            return field in trace_state.get("output_fields", [])
        icontains_match = re.match(r"([\w.]+)\s+icontains\s+(.+)", rule, re.IGNORECASE)
        if icontains_match:
            key, needle = icontains_match.groups()
            return needle.strip("'\"").lower() in str(self._value(trace_state, key)).lower()
        contains_match = re.match(r"([\w.]+)\s+contains\s+(.+)", rule)
        if contains_match:
            key, needle = contains_match.groups()
            return needle.strip("'\"") in str(self._value(trace_state, key))
        compare_match = re.match(r"(\w+)\s*(<=|>=|==|!=|<|>)\s*([0-9.]+|true|false|null)", lowered)
        if compare_match:
            key, op, raw_value = compare_match.groups()
            left = self._value(trace_state, key)
            right: Any
            if raw_value == "true":
                right = True
            elif raw_value == "false":
                right = False
            elif raw_value == "null":
                right = None
            else:
                right = float(raw_value)
                left = float(left) if left is not None else None
            return left is not None and OPS[op](left, right)
        return False

    def _value(self, trace_state: Dict[str, Any], key: str) -> Any:
        if key in trace_state:
            return trace_state.get(key)
        current: Any = trace_state
        for part in key.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current
