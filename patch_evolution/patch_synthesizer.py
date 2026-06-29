import hashlib
import json
import re
from typing import Any, Dict, List, Tuple

from patch_evolution.patch_spec import default_telemetry, make_patch_spec, validate_patch_schema
from patch_evolution.runtime_manifest import prompt_manifest_payload
from patch_evolution.static_validator import normalize_trigger


SUPPORTED_RUNTIME_SIGNALS = {
    "input",
    "output",
    "output_fields",
    "output_text",
    "answer_text",
    "answer_word_count",
    "output_word_count",
    "output_empty",
    "schema_valid",
    "parse_status",
    "error_message",
    "tool_error",
    "verifier_score",
    "confidence",
}
SUPPORTED_CONTRACT_TRANSFORMS = [
    "strip_code_fence",
    "strip_boxed",
    "strip_answer_prefix",
    "strip_final_answer_prefix",
    "number_words_to_digits",
    "strip_percent",
    "trim_numeric_trailing_zeros",
    "strip_leading_zero_decimal",
    "million_to_number",
    "trim_terminal_punctuation",
]
DEFAULT_FIELDS = ["answer", "response", "solution", "output"]


class PatchSynthesizer:
    def __init__(self, llm=None, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

    async def synthesize(
        self,
        trace_ir,
        diagnosis: Dict[str, Any],
        available_operators: List[str] = None,
        available_runtime_signals: List[str] = None,
        task_description: str = "",
    ) -> List[Dict[str, Any]]:
        if self.llm is None:
            return [self._fallback_patch(trace_ir, diagnosis)]

        prompt = self._prompt(trace_ir, diagnosis, available_operators or [], available_runtime_signals or [], task_description)
        last_error = None
        for _ in range(self.max_retries):
            response = await self.llm(prompt)
            try:
                patches = self._parse_response(response)
                valid = []
                for patch in patches:
                    ok, errors = validate_patch_schema(patch)
                    if not ok:
                        raise ValueError("; ".join(errors))
                    valid.append(patch)
                return valid
            except Exception as exc:
                last_error = exc
                prompt += f"\nPrevious JSON was invalid: {exc}. Return strict valid JSON only."
        raise ValueError(f"Patch synthesis failed: {last_error}")

    async def synthesize_failure_family(
        self,
        record: Dict[str, Any],
        diagnosis: Dict[str, Any],
        max_candidates: int = 1,
    ) -> List[Dict[str, Any]]:
        if self.llm is None:
            return []

        prompt = self._failure_family_prompt(record, diagnosis, max_candidates=max_candidates)
        last_error = None
        for _ in range(self.max_retries):
            response = await self.llm(prompt)
            try:
                patches = self._parse_response(response)
                normalized = []
                for patch in patches[: max(1, max_candidates)]:
                    normalized_patch = self._normalize_failure_family_patch(patch, record, diagnosis)
                    ok, errors = validate_patch_schema(normalized_patch)
                    if not ok:
                        raise ValueError("; ".join(errors))
                    normalized.append(normalized_patch)
                return normalized
            except Exception as exc:
                last_error = exc
                prompt += f"\nPrevious JSON was invalid: {exc}. Return strict valid JSON only."
        raise ValueError(f"Failure-family patch synthesis failed: {last_error}")

    def _fallback_patch(self, trace_ir, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        trace_id = trace_ir.metadata.trace_id
        target = diagnosis.get("target_operator", "workflow_output")
        symptom = diagnosis.get("failure_symptom", "unknown failure")
        signals = diagnosis.get("suggested_observable_signals", ["output_empty"])
        return make_patch_spec(
            patch_id=f"patch_{trace_id}_{target}".replace(" ", "_"),
            source_trace_id=trace_id,
            target_operator=target,
            diagnosed_bottleneck=diagnosis.get("suspected_bottleneck", ""),
            failure_symptom=symptom,
            trigger={
                "observable_signals": signals,
                "decision_rule": "output_empty == true" if "output_empty" in signals else f"{signals[0]} == true",
                "threshold": 0.5,
                "detector_type": "rule",
            },
            fixer={
                "type": diagnosis.get("suggested_fixer_type", "contract_repair"),
                "name": "local_repair",
                "definition": "minimal local guarded repair hypothesis",
                "input_mapping": {},
                "output_mapping": {},
            },
            scope={"task_type": trace_ir.metadata.task_name, "operator": target, "conditions": [symptom]},
            contract={
                "input_requirements": [],
                "output_requirements": ["non_empty"],
                "downstream_compatibility": ["preserve original output type when possible"],
            },
        )

    def _prompt(self, trace_ir, diagnosis, operators, signals, task_description):
        payload = {
            "trace": trace_ir.to_dict(),
            "diagnosis": diagnosis,
            "available_operators": operators,
            "available_runtime_signals": signals,
            "task_description": task_description,
        }
        return (
            "Propose one local guarded PatchSpec JSON object for Patch-as-Hypothesis. "
            "Do not rewrite workflow code. Use only observable runtime signals, never gold labels. "
            "Return either a PatchSpec object or a list of PatchSpec objects.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    def _failure_family_prompt(self, record: Dict[str, Any], diagnosis: Dict[str, Any], max_candidates: int) -> str:
        payload = self._compact_failure_payload(record, diagnosis)
        dataset = str(record.get("dataset") or "task")
        manifest_payload = prompt_manifest_payload(dataset)
        return (
            "You are generating reusable online workflow-repair PatchSpec objects inside a fixed executable DSL.\n"
            f"Current task_family: {dataset}.\n"
            "Do not solve the current sample. Do not output a corrected sample answer, gold answer, expected label, "
            "or any sample-specific final answer.\n"
            "The LLM may only instantiate PatchSpec objects from the provided operator manifest, trigger registry, "
            "fixer registry, task patch profile, observable runtime fields, and patchable output fields.\n"
            "Do not invent runtime capabilities, trigger strings, target operators, writable fields, or fixer types.\n"
            "Do not use verifier_score, confidence, uncertainty, probability, reward, gold_answer, expected_label, "
            "or evaluator_feedback unless they are explicitly listed in the provided operator manifest. "
            "In this workflow they are forbidden.\n"
            "Do not target ScEnsemble for final answer repair. ScEnsemble outputs solution_letter and thought, not answer.\n"
            "Choose target_operator only from the provided operator manifest. Choose trigger only from the target "
            "operator's allowed_triggers. Choose fixer only from the target operator's allowed_fixers. "
            "fixer.writes must be a subset of target_operator.patchable_fields.\n"
            "For missing answer fields or parse/schema failures where the existing output contains thought/raw_response, "
            "prefer fixer.type=llm_repair with writes=[\"answer\"]. Use contract_repair only when you provide an "
            "executable deterministic transform, fallback_output, or output_mapping. Use answer_normalize when the "
            "answer field already exists but has removable style/format noise.\n"
            "Composite triggers may contain at most two child rules. Prefer one precise patch over multiple broad patches. "
            f"Return at most {min(max(1, max_candidates), 2)} patches. If no valid reusable patch exists, return {{\"patches\": []}}.\n"
            "Return strict JSON only with this schema and no markdown: "
            "{\"patches\":[{\"target_operator\":\"AnswerGenerate\",\"diagnosed_bottleneck\":\"...\","
            "\"failure_symptom\":\"...\",\"trigger\":{\"detector\":\"rule\",\"type\":\"field_missing\","
            "\"field\":\"answer\"},\"fixer\":{\"type\":\"llm_repair\",\"writes\":[\"answer\"],"
            "\"repair_goal\":\"...\"},\"scope\":{\"task_family\":\"DROP\",\"applies_to\":\"...\","
            "\"does_not_apply_to\":\"...\"},\"contract\":{\"required_runtime_fields\":[\"...\"],"
            "\"target_output_fields\":[\"...\"],\"patchable_fields\":[\"...\"],"
            "\"static_validity_reason\":\"...\"}}]}.\n"
            f"Runtime manifest and registries:\n{json.dumps(manifest_payload, ensure_ascii=False)}\n"
            f"Failure payload:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    def _compact_failure_payload(self, record: Dict[str, Any], diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        operator_traces = []
        for operator in (record.get("operator_traces") or [])[-3:]:
            operator_traces.append(
                {
                    "call_index": operator.get("call_index"),
                    "operator_id": operator.get("operator_id"),
                    "operator_type": operator.get("operator_type"),
                    "schema_type": operator.get("schema_type"),
                    "output_style": operator.get("output_style"),
                    "answer_type": operator.get("answer_type"),
                    "parse_status": operator.get("parse_status"),
                    "error_type": operator.get("error_type"),
                    "output_fields": self._preview_output_fields(operator.get("output_preview")),
                    "input_preview": self._preview(operator.get("input_preview"), 180),
                    "output_preview": self._preview(operator.get("output_preview"), 180),
                }
            )
        return {
            "dataset": record.get("dataset"),
            "sample_id": record.get("sample_id"),
            "workflow_id": record.get("workflow_id"),
            "sample_preview": self._preview(record.get("sample"), 220),
            "prediction_preview": self._preview(record.get("prediction"), 180),
            "sample_score": record.get("sample_score"),
            "diagnosis": {
                "target_operator": diagnosis.get("target_operator"),
                "suspected_bottleneck": diagnosis.get("suspected_bottleneck"),
                "failure_symptom": diagnosis.get("failure_symptom"),
                "evidence": list(diagnosis.get("evidence") or [])[:2],
                "suggested_observable_signals": list(diagnosis.get("suggested_observable_signals") or [])[:4],
            },
            "operator_traces": operator_traces,
        }

    def _normalize_failure_family_patch(
        self,
        patch: Dict[str, Any],
        record: Dict[str, Any],
        diagnosis: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(patch, dict):
            raise ValueError("patch must be a JSON object")

        dataset = str(record.get("dataset") or "task")
        sample_id = str(record.get("sample_id") or "sample")
        target_operator = str(patch.get("target_operator") or self._default_target_operator(record, diagnosis))
        diagnosed_bottleneck = str(
            patch.get("diagnosed_bottleneck") or diagnosis.get("suspected_bottleneck") or "failure_family"
        )
        failure_symptom = str(patch.get("failure_symptom") or diagnosis.get("failure_symptom") or "observable failure")
        fields = self._default_fields(record)
        mapping_output = bool(fields)

        fixer = self._normalize_fixer(patch.get("fixer"), diagnosis, fields)
        trigger = self._normalize_trigger(patch.get("trigger"), diagnosis, fixer, record)
        fixer, coerced_from = self._coerce_weak_contract_repair(
            fixer=fixer,
            trigger=trigger,
            target_operator=target_operator,
            diagnosis=diagnosis,
        )
        scope = self._normalize_scope(patch.get("scope"), dataset, target_operator, failure_symptom)
        contract = self._normalize_contract(patch.get("contract"), fixer, mapping_output)

        normalized = {
            "patch_id": self._patch_id(dataset, target_operator, fixer, trigger, diagnosed_bottleneck),
            "source_trace_id": sample_id,
            "target_operator": target_operator,
            "diagnosed_bottleneck": diagnosed_bottleneck,
            "failure_symptom": failure_symptom,
            "trigger": trigger,
            "fixer": fixer,
            "scope": scope,
            "contract": contract,
            "rollback": patch.get("rollback")
            if isinstance(patch.get("rollback"), dict)
            else {"strategy": "return_original", "fallback": None},
            "telemetry": patch.get("telemetry")
            if isinstance(patch.get("telemetry"), dict)
            else default_telemetry(),
            "status": "candidate",
            "generation": {
                "mode": "failure_family_llm",
                "prompt_version": 3,
                "sample_bound": False,
            },
        }
        if coerced_from:
            normalized["generation"]["coerced_fixer_from"] = coerced_from
        generalization_note = patch.get("generalization_note")
        if generalization_note:
            normalized["generalization_note"] = str(generalization_note)
        return normalized

    def _normalize_trigger(
        self,
        trigger: Any,
        diagnosis: Dict[str, Any],
        fixer: Dict[str, Any],
        record: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        trigger = trigger if isinstance(trigger, dict) else {}
        normalized = normalize_trigger(trigger)
        if self._should_rewrite_parse_failure_to_missing_answer(normalized, fixer, record):
            normalized = {"detector": "rule", "type": "field_missing", "field": "answer"}
        if normalized.get("type") == "unsupported_raw" and not normalized.get("decision_rule"):
            normalized = {"detector": "rule", "type": "field_missing", "field": (fixer.get("writes") or ["answer"])[0]}
        return normalized

    def _should_rewrite_parse_failure_to_missing_answer(
        self,
        trigger: Dict[str, Any],
        fixer: Dict[str, Any],
        record: Dict[str, Any] = None,
    ) -> bool:
        writes = set(fixer.get("writes") or fixer.get("fields") or [])
        if "answer" not in writes:
            return False
        if trigger.get("type") != "enum_equals" or trigger.get("field") != "parse_status":
            return False
        if str(trigger.get("value") or "").lower() not in {"failed", "invalid_json", "empty"}:
            return False
        return self._record_has_missing_answer(record)

    def _record_has_missing_answer(self, record: Dict[str, Any] = None) -> bool:
        if not isinstance(record, dict):
            return False
        prediction = str(record.get("prediction") or "").strip()
        if prediction in {"'answer'", '"answer"', "KeyError('answer')", "KeyError: 'answer'"}:
            return True
        for operator in reversed(record.get("operator_traces") or []):
            if operator.get("operator_type") != "AnswerGenerate":
                continue
            parsed = self._parse_preview(operator.get("output_preview"))
            if isinstance(parsed, dict) and "answer" not in parsed:
                return True
        return False

    def _coerce_weak_contract_repair(
        self,
        fixer: Dict[str, Any],
        trigger: Dict[str, Any],
        target_operator: str,
        diagnosis: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        if target_operator != "AnswerGenerate" or fixer.get("type") != "contract_repair":
            return fixer, ""
        writes = fixer.get("writes") or fixer.get("fields") or []
        if "answer" not in writes:
            return fixer, ""
        has_concrete_repair = bool(fixer.get("transforms") or fixer.get("output_mapping") or "fallback_output" in fixer)
        if has_concrete_repair:
            return fixer, ""
        if not self._answer_repair_trigger(trigger):
            return fixer, ""
        field = "answer"
        normalized = {
            "type": "llm_repair",
            "name": self._safe_name(fixer.get("name") or "family_missing_answer_llm_repair"),
            "definition": str(
                fixer.get("definition")
                or fixer.get("repair_goal")
                or self._default_llm_definition(diagnosis, field)
            ),
            "repair_goal": str(
                fixer.get("repair_goal")
                or "Extract the missing final answer from existing thought/raw_response without introducing new facts."
            ),
            "fields": [field],
            "writes": [field],
            "max_input_chars": self._safe_int(fixer.get("max_input_chars"), 1800),
            "max_output_chars": self._safe_int(fixer.get("max_output_chars"), 1400),
            "input_mapping": fixer.get("input_mapping") if isinstance(fixer.get("input_mapping"), dict) else {},
            "output_mapping": fixer.get("output_mapping") if isinstance(fixer.get("output_mapping"), dict) else {},
        }
        return normalized, "contract_repair"

    def _answer_repair_trigger(self, trigger: Dict[str, Any]) -> bool:
        if trigger.get("detector") == "composite":
            return any(self._answer_repair_trigger(rule) for rule in trigger.get("rules") or [])
        trigger_type = trigger.get("type")
        field = trigger.get("field")
        if trigger_type == "field_missing" and field == "answer":
            return True
        if trigger_type == "text_empty" and field == "answer_text":
            return True
        if trigger_type == "bool_equals" and field in {"schema_valid", "output_empty"}:
            return True
        if trigger_type == "enum_equals" and field == "parse_status":
            return True
        return False

    def _normalize_fixer(self, fixer: Any, diagnosis: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        fixer = fixer if isinstance(fixer, dict) else {}
        fixer_type = str(fixer.get("type") or diagnosis.get("suggested_fixer_type") or "llm_repair").strip()
        if fixer_type not in {"contract_repair", "answer_normalize", "llm_repair"}:
            fixer_type = "llm_repair"

        normalized_fields = self._normalize_fields(fixer.get("writes") or fixer.get("fields"), fields)
        if fixer_type == "contract_repair":
            transforms = [
                transform
                for transform in self._normalize_string_list(fixer.get("transforms"))
                if transform in SUPPORTED_CONTRACT_TRANSFORMS
            ]
            output_mapping = fixer.get("output_mapping") if isinstance(fixer.get("output_mapping"), dict) else {}
            has_concrete_repair = bool(transforms or output_mapping or "fallback_output" in fixer)
            normalized = {
                "type": "contract_repair",
                "name": self._safe_name(fixer.get("name") or "family_contract_repair"),
                "definition": str(
                    fixer.get("definition")
                    or fixer.get("repair_goal")
                    or "Recover required patchable output fields without changing unpatchable fields."
                ),
                "repair_goal": str(
                    fixer.get("repair_goal")
                    or "Recover the required patchable field from existing output/raw_response if possible."
                ),
                "fields": normalized_fields,
                "writes": normalized_fields,
                "transforms": transforms,
                "input_mapping": fixer.get("input_mapping") if isinstance(fixer.get("input_mapping"), dict) else {},
                "output_mapping": output_mapping,
            }
            if "fallback_output" in fixer:
                normalized["fallback_output"] = fixer.get("fallback_output")
            return normalized

        if fixer_type == "answer_normalize":
            return {
                "type": "answer_normalize",
                "name": self._safe_name(fixer.get("name") or "family_answer_normalize"),
                "definition": str(
                    fixer.get("definition")
                    or fixer.get("repair_goal")
                    or "Rewrite the existing answer into a concise DROP-style final answer without explanation."
                ),
                "repair_goal": str(
                    fixer.get("repair_goal")
                    or "Normalize an existing answer/raw response; do not solve the sample from scratch."
                ),
                "fields": normalized_fields,
                "writes": normalized_fields,
                "transforms": [
                    transform
                    for transform in self._normalize_string_list(fixer.get("transforms"))
                    if transform in SUPPORTED_CONTRACT_TRANSFORMS
                ],
                "input_mapping": fixer.get("input_mapping") if isinstance(fixer.get("input_mapping"), dict) else {},
                "output_mapping": fixer.get("output_mapping") if isinstance(fixer.get("output_mapping"), dict) else {},
            }

        return {
            "type": "llm_repair",
            "name": self._safe_name(fixer.get("name") or "family_llm_repair"),
            "definition": str(
                fixer.get("definition")
                or fixer.get("repair_goal")
                or self._default_llm_definition(diagnosis, normalized_fields[0] if normalized_fields else "answer")
            ),
            "repair_goal": str(
                fixer.get("repair_goal")
                or "Use only existing operator output/raw response to repair the patchable field."
            ),
            "fields": normalized_fields,
            "writes": normalized_fields,
            "max_input_chars": self._safe_int(fixer.get("max_input_chars"), 2200),
            "max_output_chars": self._safe_int(fixer.get("max_output_chars"), 900),
            "input_mapping": fixer.get("input_mapping") if isinstance(fixer.get("input_mapping"), dict) else {},
            "output_mapping": fixer.get("output_mapping") if isinstance(fixer.get("output_mapping"), dict) else {},
        }

    def _normalize_scope(
        self,
        scope: Any,
        dataset: str,
        target_operator: str,
        failure_symptom: str,
    ) -> Dict[str, Any]:
        scope = scope if isinstance(scope, dict) else {}
        conditions = self._normalize_string_list(scope.get("conditions"))
        if not conditions:
            conditions = [failure_symptom]
        return {
            "task_type": str(scope.get("task_type") or dataset),
            "operator": str(scope.get("operator") or target_operator),
            "conditions": conditions,
        }

    def _normalize_contract(self, contract: Any, fixer: Dict[str, Any], mapping_output: bool) -> Dict[str, Any]:
        contract = contract if isinstance(contract, dict) else {}
        input_requirements = self._normalize_string_list(contract.get("input_requirements"))
        output_requirements = self._normalize_string_list(contract.get("output_requirements"))
        if not output_requirements:
            output_requirements = ["non_empty"]
            fields = fixer.get("fields") or []
            if mapping_output and fields:
                output_requirements.append(f"field:{fields[0]}")
        downstream_compatibility = self._normalize_string_list(contract.get("downstream_compatibility"))
        if not downstream_compatibility:
            downstream_compatibility = ["Preserve the original output container when possible."]
        return {
            "input_requirements": input_requirements,
            "output_requirements": output_requirements,
            "downstream_compatibility": downstream_compatibility,
            "required_runtime_fields": self._normalize_string_list(contract.get("required_runtime_fields")),
            "target_output_fields": self._normalize_string_list(contract.get("target_output_fields")) or fixer.get("fields") or [],
            "patchable_fields": self._normalize_string_list(contract.get("patchable_fields")) or fixer.get("writes") or [],
            "static_validity_reason": str(contract.get("static_validity_reason") or ""),
        }

    def _default_target_operator(self, record: Dict[str, Any], diagnosis: Dict[str, Any]) -> str:
        if diagnosis.get("target_operator"):
            return str(diagnosis["target_operator"])
        operators = record.get("operator_traces") or []
        if operators:
            operator = operators[-1]
            return str(operator.get("operator_type") or operator.get("operator_id") or "workflow_output")
        return "workflow_output"

    def _default_fields(self, record: Dict[str, Any]) -> List[str]:
        operators = record.get("operator_traces") or []
        for operator in reversed(operators):
            parsed = self._parse_preview(operator.get("output_preview"))
            if isinstance(parsed, dict):
                fields = [field for field in DEFAULT_FIELDS if field in parsed]
                if fields:
                    return fields
        return []

    def _default_llm_definition(self, diagnosis: Dict[str, Any], field: str) -> str:
        symptom = str(diagnosis.get("failure_symptom") or "the observed failure")
        bottleneck = str(diagnosis.get("suspected_bottleneck") or "local answer repair")
        return (
            f"Repair outputs from the `{bottleneck}` failure family. Use only the operator input and original output. "
            f"Rewrite the `{field}` field into the shortest scorer-compatible final answer for this task type, "
            f"while fixing the observable symptom: {symptom}. Do not add explanations."
        )

    def _fallback_rule(self, observable_signals: List[str], fixer: Dict[str, Any]) -> str:
        signals = set(observable_signals or [])
        if "output_empty" in signals:
            return "output_empty == true"
        if "schema_valid" in signals or "parse_status" in signals:
            return "schema_valid == false"
        if "output_fields" in signals:
            fields = fixer.get("fields") or ["answer"]
            return f"field_missing {fields[0]}"
        if "verifier_score" in signals:
            return "verifier_score < 1"
        if "confidence" in signals:
            return "confidence < 0.5"
        if "answer_word_count" in signals or fixer.get("type") == "llm_repair":
            return "answer_word_count >= 6"
        return "output_empty == true"

    def _patch_id(
        self,
        dataset: str,
        target_operator: str,
        fixer: Dict[str, Any],
        trigger: Dict[str, Any],
        diagnosed_bottleneck: str,
    ) -> str:
        payload = {
            "dataset": dataset,
            "target_operator": target_operator,
            "diagnosed_bottleneck": diagnosed_bottleneck,
            "fixer": fixer,
            "trigger": trigger,
        }
        digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        fixer_name = self._safe_name(fixer.get("name") or fixer.get("type") or "family_patch")
        return f"online_{dataset}_{target_operator}_{fixer_name}_{digest}"

    def _normalize_fields(self, candidate_fields: Any, default_fields: List[str]) -> List[str]:
        fields = self._normalize_string_list(candidate_fields)
        valid_fields = [field for field in fields if field in DEFAULT_FIELDS]
        if valid_fields:
            return valid_fields
        if default_fields:
            return default_fields[:1]
        return ["answer"]

    def _normalize_string_list(self, values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        normalized = []
        for value in values:
            text = str(value).strip()
            if text:
                normalized.append(text)
        return normalized

    def _parse_preview(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value

    def _preview_output_fields(self, value: Any) -> List[str]:
        parsed = self._parse_preview(value)
        if isinstance(parsed, dict):
            return sorted(str(key) for key in parsed.keys())
        return []

    def _preview(self, value: Any, max_chars: int) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False)
            except TypeError:
                text = str(value)
        return text if len(text) <= max_chars else text[: max(0, max_chars - 3)] + "..."

    def _safe_name(self, value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "patch")).strip("._") or "patch"

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        text = response.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("patches"), list):
            return data["patches"]
        return data if isinstance(data, list) else [data]
