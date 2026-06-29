from typing import Any, Dict, List

from patch_evolution.contracts import ContractChecker
from patch_evolution.trigger_detector import TriggerDetector


DEFAULT_THRESHOLDS = {
    "repair_gain": 0.0,
    "regression_rate": 0.02,
    "false_activation_rate": 0.20,
    "cost_overhead": 1.0,
    "contract_failure_rate": 0.10,
}


class ShadowValidator:
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.detector = TriggerDetector()
        self.contract_checker = ContractChecker()

    def validate(self, patch_spec: Dict[str, Any], failed_traces: List[Any], success_traces: List[Any]) -> Dict[str, Any]:
        failed_states = [self._trace_to_state(trace) for trace in failed_traces]
        success_states = [self._trace_to_state(trace) for trace in success_traces]
        failed_activations = sum(1 for state in failed_states if self.detector.should_trigger(state, patch_spec)[0])
        success_activations = sum(1 for state in success_states if self.detector.should_trigger(state, patch_spec)[0])

        repair_gain = self._average([state.get("simulated_gain", 0.0) for state in failed_states])
        false_activation_rate = success_activations / len(success_states) if success_states else 0.0
        regression_rate = self._average([state.get("simulated_regression", 0.0) for state in success_states])
        cost_overhead = patch_spec.get("telemetry", {}).get("average_cost_overhead", 0.0)
        contract_failure_rate = self._average([state.get("contract_failure", 0.0) for state in failed_states])

        passed = (
            repair_gain > self.thresholds["repair_gain"]
            and regression_rate <= self.thresholds["regression_rate"]
            and false_activation_rate <= self.thresholds["false_activation_rate"]
            and cost_overhead <= self.thresholds["cost_overhead"]
            and contract_failure_rate <= self.thresholds["contract_failure_rate"]
        )
        return {
            "passed": passed,
            "failed_trace_count": len(failed_states),
            "success_trace_count": len(success_states),
            "failed_activation_rate": failed_activations / len(failed_states) if failed_states else 0.0,
            "false_activation_rate": false_activation_rate,
            "repair_gain": repair_gain,
            "regression_rate": regression_rate,
            "cost_overhead": cost_overhead,
            "contract_failure_rate": contract_failure_rate,
            "recommended_status": "validated" if passed else "disabled",
        }

    def _trace_to_state(self, trace: Any) -> Dict[str, Any]:
        if isinstance(trace, dict):
            return trace
        nodes = getattr(trace, "nodes", [])
        operator = nodes[-1].operator if nodes else None
        metadata = operator.metadata if operator else {}
        return {
            "output": operator.operator_output if operator else getattr(trace, "final_output", None),
            "schema_valid": metadata.get("parse_status") not in {"failed", "invalid_json", "empty"},
            "parse_status": metadata.get("parse_status"),
            "error_message": operator.error_message if operator else None,
            "verifier_score": operator.verifier_score if operator else None,
            "confidence": operator.confidence if operator else None,
            **metadata,
        }

    def _average(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

