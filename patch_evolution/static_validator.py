import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from patch_evolution.runtime_manifest import (
    ANSWER_REPAIR_FIXERS,
    FIXER_REGISTRY,
    FORBIDDEN_SIGNALS,
    TRIGGER_REGISTRY,
    OperatorManifest,
    get_operator_manifest,
    get_patch_profile,
)


@dataclass
class StaticValidationResult:
    status: str
    ok: bool
    errors: List[str] = field(default_factory=list)
    normalized_patch: Optional[Dict[str, Any]] = None


class StaticPatchValidator:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.profile = get_patch_profile(dataset)

    def normalize_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(patch)
        target_operator = str(normalized.get("target_operator") or "")
        manifest = get_operator_manifest(self.dataset, target_operator)
        fixer = self._normalize_fixer(normalized.get("fixer") or {}, manifest)
        trigger = normalize_trigger(normalized.get("trigger") or {})
        normalized["fixer"] = fixer
        normalized["trigger"] = trigger
        normalized["patch_key"] = patch_memory_key(self.dataset, normalized)
        normalized.setdefault("source_trace_ids", [])
        source_trace_id = normalized.get("source_trace_id")
        if source_trace_id and source_trace_id not in normalized["source_trace_ids"]:
            normalized["source_trace_ids"].append(source_trace_id)
        return normalized

    def validate(self, patch: Dict[str, Any]) -> StaticValidationResult:
        normalized = self.normalize_patch(patch)
        target_operator = str(normalized.get("target_operator") or "")
        manifest = get_operator_manifest(self.dataset, target_operator)
        if manifest is None:
            return self._reject("rejected_unknown_operator", normalized, f"Unknown operator: {target_operator}")

        fixer = normalized.get("fixer") or {}
        fixer_type = str(fixer.get("type") or "")
        if fixer_type not in FIXER_REGISTRY:
            return self._reject("rejected_unknown_fixer", normalized, f"Unknown fixer: {fixer_type}")
        if fixer_type not in self.profile.get("allowed_fixers", []):
            return self._reject("rejected_fixer_not_allowed", normalized, f"Fixer not allowed by task profile: {fixer_type}")
        if fixer_type not in manifest.allowed_fixers:
            return self._reject("rejected_fixer_not_allowed", normalized, f"Fixer not allowed for {manifest.name}: {fixer_type}")

        writes = _string_list(fixer.get("writes"))
        if not writes:
            return self._reject("rejected_target_field_mismatch", normalized, "Fixer writes must be non-empty")
        if not set(writes).issubset(set(manifest.patchable_fields)):
            return self._reject(
                "rejected_target_field_mismatch",
                normalized,
                f"Fixer writes {writes} outside patchable fields {manifest.patchable_fields}",
            )
        forbidden_write = sorted(set(writes).intersection(manifest.forbidden_writes))
        if forbidden_write:
            return self._reject("rejected_forbidden_write", normalized, f"Forbidden writes: {forbidden_write}")
        if manifest.name == "ScEnsemble" and "answer" in writes:
            return self._reject("rejected_target_field_mismatch", normalized, "ScEnsemble cannot write answer")
        if manifest.name == "ScEnsemble" and fixer_type in ANSWER_REPAIR_FIXERS:
            return self._reject("rejected_fixer_not_allowed", normalized, "ScEnsemble cannot use answer repair fixers")

        trigger_status, trigger_errors = self._validate_trigger(normalized.get("trigger") or {}, manifest)
        if trigger_status != "static_valid":
            return self._reject(trigger_status, normalized, "; ".join(trigger_errors))

        normalized["status"] = "candidate"
        return StaticValidationResult(status="static_valid", ok=True, normalized_patch=normalized)

    def _validate_trigger(self, trigger: Dict[str, Any], manifest: OperatorManifest) -> Tuple[str, List[str]]:
        detector = str(trigger.get("detector") or trigger.get("detector_type") or "rule")
        if detector == "composite":
            rules = trigger.get("rules") or []
            if len(rules) > 2:
                return "rejected_trigger_not_allowed", ["Composite triggers are limited to at most two rules"]
            if not rules:
                return "rejected_unknown_trigger", ["Composite trigger has no rules"]
            for child in rules:
                status, errors = self._validate_trigger(normalize_trigger(child), manifest)
                if status != "static_valid":
                    return status, errors
            return "static_valid", []

        trigger_type = str(trigger.get("type") or "")
        field = trigger.get("field")
        if field in self.profile.get("forbidden_signals", FORBIDDEN_SIGNALS):
            return "rejected_invalid_trigger_signal", [f"Forbidden trigger signal: {field}"]
        if trigger_type not in TRIGGER_REGISTRY:
            if field in self.profile.get("forbidden_signals", FORBIDDEN_SIGNALS):
                return "rejected_invalid_trigger_signal", [f"Forbidden trigger signal: {field}"]
            return "rejected_unknown_trigger", [f"Unknown trigger type: {trigger_type}"]
        if trigger_type not in self.profile.get("allowed_trigger_types", []):
            return "rejected_trigger_not_allowed", [f"Trigger type not allowed by task profile: {trigger_type}"]
        manifest_types = {spec.type for spec in manifest.allowed_triggers}
        if trigger_type not in manifest_types:
            return "rejected_trigger_not_allowed", [f"Trigger type not allowed for {manifest.name}: {trigger_type}"]
        if field and field not in manifest.observable_fields:
            return "rejected_invalid_trigger_signal", [f"Trigger field not observable for {manifest.name}: {field}"]
        if field and field in self.profile.get("forbidden_signals", FORBIDDEN_SIGNALS):
            return "rejected_invalid_trigger_signal", [f"Forbidden trigger signal: {field}"]
        if not self._trigger_field_allowed(trigger, manifest):
            return "rejected_trigger_not_allowed", [
                f"Trigger {trigger_type} on {field} is not allowed for {manifest.name}"
            ]
        return "static_valid", []

    def _trigger_field_allowed(self, trigger: Dict[str, Any], manifest: OperatorManifest) -> bool:
        trigger_type = trigger.get("type")
        field = trigger.get("field")
        for spec in manifest.allowed_triggers:
            if spec.type != trigger_type:
                continue
            if spec.field is None or spec.field == field:
                return True
        return False

    def _normalize_fixer(self, fixer: Dict[str, Any], manifest: Optional[OperatorManifest]) -> Dict[str, Any]:
        normalized = dict(fixer)
        fields = _string_list(normalized.get("fields"))
        writes = _string_list(normalized.get("writes"))
        if not writes:
            writes = fields
        if not writes and manifest is not None:
            writes = manifest.patchable_fields[:1]
        normalized["writes"] = writes
        if fields and "fields" not in normalized:
            normalized["fields"] = fields
        elif writes and "fields" not in normalized:
            normalized["fields"] = writes
        return normalized

    def _reject(self, status: str, patch: Dict[str, Any], error: str) -> StaticValidationResult:
        patch = dict(patch)
        patch["status"] = "static_rejected"
        return StaticValidationResult(status=status, ok=False, errors=[error], normalized_patch=patch)


def normalize_trigger(trigger: Any) -> Dict[str, Any]:
    if isinstance(trigger, dict):
        detector = str(trigger.get("detector") or trigger.get("detector_type") or "rule")
        if detector == "composite":
            raw_rules = trigger.get("rules") or []
            rules = [normalize_trigger(rule) for rule in raw_rules[:2]]
            return {"detector": "composite", "op": str(trigger.get("op") or trigger.get("match") or "AND").upper(), "rules": rules}
        if trigger.get("type"):
            normalized = dict(trigger)
            normalized["detector"] = detector
            normalized.pop("detector_type", None)
            return normalized
        decision_rule = str(trigger.get("decision_rule") or "").strip()
        if decision_rule:
            return _parse_rule_string(decision_rule)
        rules = trigger.get("rules") or []
        if len(rules) == 1:
            return normalize_trigger(rules[0])
        if rules:
            return {"detector": "composite", "op": str(trigger.get("match") or "AND").upper(), "rules": [normalize_trigger(rule) for rule in rules[:2]]}
        return {"detector": "rule", "type": "unsupported_raw", "decision_rule": ""}
    if isinstance(trigger, str):
        return _parse_rule_string(trigger)
    return {"detector": "rule", "type": "unsupported_raw", "decision_rule": str(trigger)}


def _parse_rule_string(rule: str) -> Dict[str, Any]:
    text = rule.strip()
    lowered = text.lower()

    field_missing_call = re.match(r"field_missing\s*\(?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\)?", text, re.IGNORECASE)
    if field_missing_call:
        return {"detector": "rule", "type": "field_missing", "field": field_missing_call.group(1)}
    answer_missing = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s+missing$", text, re.IGNORECASE)
    if answer_missing:
        return {"detector": "rule", "type": "field_missing", "field": answer_missing.group(1)}
    field_present_call = re.match(r"field_present\s*\(?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\)?", text, re.IGNORECASE)
    if field_present_call:
        return {"detector": "rule", "type": "field_present", "field": field_present_call.group(1)}

    if lowered in {"output_empty", "output_empty == true"}:
        return {"detector": "rule", "type": "bool_equals", "field": "output_empty", "value": True}
    if lowered in {"schema_valid == false", "not schema_valid"}:
        return {"detector": "rule", "type": "bool_equals", "field": "schema_valid", "value": False}

    bool_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(true|false)", lowered)
    if bool_match:
        field, raw_value = bool_match.groups()
        return {"detector": "rule", "type": "bool_equals", "field": field, "value": raw_value == "true"}

    enum_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*==\s*['\"]?([A-Za-z_][A-Za-z0-9_.-]*)['\"]?", text)
    if enum_match:
        field, value = enum_match.groups()
        return {"detector": "rule", "type": "enum_equals", "field": field, "value": value}

    int_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*>=\s*(\d+)", text)
    if int_match:
        field, value = int_match.groups()
        return {"detector": "rule", "type": "int_gte", "field": field, "value": int(value)}

    icontains = re.match(r"([\w.]+)\s+icontains\s+(.+)", text, re.IGNORECASE)
    if icontains:
        field, value = icontains.groups()
        return {"detector": "rule", "type": "icontains", "field": field, "value": value.strip("'\"")}
    contains = re.match(r"([\w.]+)\s+contains\s+(.+)", text)
    if contains:
        field, value = contains.groups()
        return {"detector": "rule", "type": "contains", "field": field, "value": value.strip("'\"")}

    numeric_compare = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|!=|<|>)\s*([0-9.]+)", text)
    if numeric_compare:
        field = numeric_compare.group(1)
        return {"detector": "rule", "type": "unsupported_raw", "field": field, "decision_rule": text}

    return {"detector": "rule", "type": "unsupported_raw", "decision_rule": text}


def patch_memory_key(dataset: str, patch: Dict[str, Any]) -> str:
    trigger = patch.get("trigger") or {}
    fixer = patch.get("fixer") or {}
    trigger_signature = _trigger_signature(trigger)
    fixer_signature = _fixer_signature(fixer)
    return "::".join(
        [
            str(dataset),
            str(patch.get("target_operator") or ""),
            trigger_signature,
            fixer_signature,
        ]
    )


def _trigger_signature(trigger: Dict[str, Any]) -> str:
    detector = str(trigger.get("detector") or trigger.get("detector_type") or "rule")
    if detector == "composite":
        rules = [normalize_trigger(rule) for rule in trigger.get("rules") or []]
        payload = {
            "detector": "composite",
            "op": str(trigger.get("op") or trigger.get("match") or "AND").upper(),
            "rules": [_canonical_trigger(rule) for rule in rules],
        }
        return f"composite:{payload['op'].lower()}:{_short_hash(payload)}"

    canonical = _canonical_trigger(trigger)
    trigger_type = str(canonical.get("type") or detector or "rule")
    field = str(canonical.get("field") or "")
    value = canonical.get("value", None)
    if value is not None:
        return f"{_safe_fragment(trigger_type)}:{_safe_fragment(field)}:{_safe_fragment(value)}"
    if canonical.get("decision_rule"):
        return f"{_safe_fragment(trigger_type)}:{_safe_fragment(field)}:{_short_hash(canonical.get('decision_rule'))}"
    return f"{_safe_fragment(trigger_type)}:{_safe_fragment(field)}"


def _canonical_trigger(trigger: Dict[str, Any]) -> Dict[str, Any]:
    canonical = {
        "detector": str(trigger.get("detector") or trigger.get("detector_type") or "rule"),
        "type": str(trigger.get("type") or ""),
        "field": str(trigger.get("field") or ""),
    }
    if "value" in trigger:
        canonical["value"] = trigger.get("value")
    if trigger.get("decision_rule"):
        canonical["decision_rule"] = str(trigger.get("decision_rule"))
    return canonical


def _fixer_signature(fixer: Dict[str, Any]) -> str:
    writes = sorted(_string_list(fixer.get("writes")) or _string_list(fixer.get("fields")))
    payload = {
        "type": str(fixer.get("type") or ""),
        "name": str(fixer.get("name") or ""),
        "writes": writes,
        "fields": sorted(_string_list(fixer.get("fields"))),
        "transforms": sorted(_string_list(fixer.get("transforms"))),
        "repair_goal": str(fixer.get("repair_goal") or ""),
        "definition": str(fixer.get("definition") or ""),
        "output_mapping": fixer.get("output_mapping") if isinstance(fixer.get("output_mapping"), dict) else {},
    }
    if "fallback_output" in fixer:
        payload["fallback_output"] = fixer.get("fallback_output")
    return f"{_safe_fragment(payload['type'])}:{_safe_fragment(','.join(writes))}:{_short_hash(payload)}"


def _short_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _safe_fragment(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:48] or "none"


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []
