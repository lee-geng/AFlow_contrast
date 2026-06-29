import json
import re
from typing import Any, Dict, List


class ProbeBeforeEdit:
    def diagnose(self, failed_trace, workflow_definition: Any = None, success_traces: List[Any] = None, evaluator=None) -> Dict[str, Any]:
        nodes = getattr(failed_trace, "nodes", [])
        target = nodes[-1].operator if nodes else None
        target_operator = target.operator_name if target else "workflow_output"
        output = target.operator_output if target else getattr(failed_trace, "final_output", "")
        metadata = target.metadata if target else {}
        error = target.error_message if target else None

        diagnosis = self._detect_schema_failure(target_operator, output, metadata, error)
        if diagnosis:
            return diagnosis
        diagnosis = self._detect_retrieval_failure(target_operator, metadata, output)
        if diagnosis:
            return diagnosis
        diagnosis = self._detect_tool_failure(target_operator, metadata, error, output)
        if diagnosis:
            return diagnosis
        diagnosis = self._detect_math_failure(target_operator, metadata, output)
        if diagnosis:
            return diagnosis
        return self._detect_solver_unsupported(target_operator, metadata, output)

    def _base(self, target_operator: str, bottleneck: str, symptom: str, evidence: List[str], signals: List[str], fixer: str):
        return {
            "target_operator": target_operator,
            "suspected_bottleneck": bottleneck,
            "failure_symptom": symptom,
            "evidence": evidence,
            "suggested_observable_signals": signals,
            "suggested_fixer_type": fixer,
        }

    def _detect_schema_failure(self, target_operator, output, metadata, error):
        parse_status = metadata.get("parse_status")
        if error or parse_status in {"failed", "invalid_json", "empty"}:
            return self._base(
                target_operator,
                "output_schema_or_format",
                "operator output is empty, invalid, or failed parsing",
                [str(error or parse_status)],
                ["parse_status", "schema_valid", "output_empty", "error_message"],
                "contract_repair",
            )
        if isinstance(output, str) and output.strip().startswith(("{", "[")):
            try:
                json.loads(output)
            except json.JSONDecodeError as exc:
                return self._base(
                    target_operator,
                    "output_schema_or_format",
                    "JSON-like output failed to parse",
                    [str(exc)],
                    ["schema_valid", "parse_status"],
                    "contract_repair",
                )
        return None

    def _detect_retrieval_failure(self, target_operator, metadata, output):
        if metadata.get("retrieval_score", 1.0) is not None and metadata.get("retrieval_score", 1.0) < 0.2:
            return self._base(
                target_operator,
                "retrieval_or_evidence_insufficiency",
                "retrieval score is low",
                [f"retrieval_score={metadata.get('retrieval_score')}"],
                ["retrieval_score", "empty_docs", "entity_coverage"],
                "existing_operator",
            )
        if metadata.get("empty_docs") is True:
            return self._base(
                target_operator,
                "retrieval_or_evidence_insufficiency",
                "retrieval returned empty evidence",
                ["empty_docs=True"],
                ["empty_docs", "retrieval_score"],
                "existing_operator",
            )
        return None

    def _detect_tool_failure(self, target_operator, metadata, error, output):
        if metadata.get("tool_error") or (error and "tool" in str(error).lower()):
            return self._base(
                target_operator,
                "tool_use_failure",
                "tool call failed or returned invalid result",
                [str(error or metadata.get("tool_error"))],
                ["tool_error", "error_message"],
                "tool_call",
            )
        return None

    def _detect_math_failure(self, target_operator, metadata, output):
        verifier_score = metadata.get("verifier_score")
        if verifier_score is not None and verifier_score < 1.0:
            return self._base(
                target_operator,
                "math_or_symbolic_inconsistency",
                "verifier score indicates numeric or symbolic inconsistency",
                [f"verifier_score={verifier_score}"],
                ["verifier_score", "equation_check_pass"],
                "existing_operator",
            )
        if isinstance(output, str) and re.search(r"\d", output) and metadata.get("calculation_inconsistency"):
            return self._base(
                target_operator,
                "math_or_symbolic_inconsistency",
                "calculation inconsistency was observed",
                ["calculation_inconsistency=True"],
                ["calculation_inconsistency"],
                "existing_operator",
            )
        return None

    def _detect_solver_unsupported(self, target_operator, metadata, output):
        return self._base(
            target_operator,
            "unsupported_or_ungrounded_answer",
            "answer may be unsupported by available evidence or low reliability",
            [f"confidence={metadata.get('confidence', 'unknown')}"],
            ["confidence", "verifier_score", "evidence_support"],
            "prompt_patch",
        )

