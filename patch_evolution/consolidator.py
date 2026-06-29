from collections import defaultdict
from typing import Any, Dict, List

CONSOLIDATABLE_STATUSES = {"validated", "active_guarded", "promoted", "merged"}


DEFAULT_WEIGHTS = {
    "alpha": 1.0,
    "beta": 0.5,
    "gamma": 0.3,
    "lambda": 1.0,
    "mu": 0.2,
}


class OfflineConsolidator:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def credit(self, patch: Dict[str, Any]) -> float:
        telemetry = patch.get("telemetry", {})
        checked = max(1, telemetry.get("checked_count", 0))
        activations = telemetry.get("activation_count", 0)
        accepted = telemetry.get("accepted_count", 0)
        regressions = telemetry.get("regression_count", 0)
        trigger_precision = accepted / max(1, activations)
        activation_frequency = activations / checked
        regression_rate = regressions / max(1, accepted)
        return (
            self.weights["alpha"] * telemetry.get("average_gain", 0.0)
            + self.weights["beta"] * trigger_precision
            + self.weights["gamma"] * activation_frequency
            - self.weights["lambda"] * regression_rate
            - self.weights["mu"] * telemetry.get("average_cost_overhead", 0.0)
        )

    def decide_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        if patch.get("status") not in CONSOLIDATABLE_STATUSES:
            return {"patch_id": patch.get("patch_id"), "decision": "prune", "credit": 0.0}
        telemetry = patch.get("telemetry", {})
        activations = telemetry.get("activation_count", 0)
        accepted = telemetry.get("accepted_count", 0)
        rollbacks = telemetry.get("rollback_count", 0)
        regressions = telemetry.get("regression_count", 0)
        credit = self.credit(patch)
        contract_failure_rate = rollbacks / max(1, activations)
        regression_rate = regressions / max(1, accepted)

        if activations < 2 and telemetry.get("average_gain", 0.0) <= 0:
            decision = "prune"
        elif regression_rate > 0.1 or contract_failure_rate > 0.2:
            decision = "prune"
        elif activations >= 20 and credit > 0.8 and regression_rate <= 0.02:
            decision = "promote"
        elif credit > 0:
            decision = "keep"
        else:
            decision = "prune"
        return {"patch_id": patch.get("patch_id"), "decision": decision, "credit": credit}

    def group_similar_for_merge(self, patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups = defaultdict(list)
        for patch in patches:
            if patch.get("status") not in CONSOLIDATABLE_STATUSES:
                continue
            key = (patch.get("target_operator"), patch.get("failure_symptom"), patch.get("fixer", {}).get("type"))
            groups[key].append(patch)
        return [
            {"decision": "merge", "patch_ids": [patch["patch_id"] for patch in group], "key": key}
            for key, group in groups.items()
            if len(group) > 1
        ]

    def consolidate(self, patches: List[Dict[str, Any]]) -> Dict[str, Any]:
        decisions = [self.decide_patch(patch) for patch in patches]
        merge_groups = self.group_similar_for_merge(patches)
        return {"decisions": decisions, "merge_groups": merge_groups}
