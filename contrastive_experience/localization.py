from typing import Any, Dict, List


def localize_bottleneck(divergence_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not divergence_scores:
        return {
            "candidate_operator": None,
            "confidence": 0.0,
            "evidence": "No operator divergence scores were available.",
            "alternative_candidates": [],
        }

    by_call = sorted(divergence_scores, key=lambda item: _call_index(item.get("operator_id", "")))
    previous_output = {}
    for item in by_call:
        call_index = _call_index(item.get("operator_id", ""))
        prior = previous_output.get(call_index - 1, 0.0)
        item["divergence_jump"] = item.get("output_div", 0.0) - prior
        previous_output[call_index] = item.get("output_div", 0.0)

    def localization_score(item):
        early_bonus = max(0.0, 1.0 - 0.05 * _call_index(item.get("operator_id", "")))
        interpretable = 0.1 if item.get("main_signal") != "weak divergence" else 0.0
        return (
            item.get("bottleneck_score", 0.0)
            + 0.4 * max(0.0, item.get("amplification", 0.0))
            + 0.3 * max(0.0, item.get("divergence_jump", 0.0))
            + interpretable
            + 0.05 * early_bonus
        )

    ranked = sorted(divergence_scores, key=localization_score, reverse=True)
    best = dict(ranked[0])
    confidence = min(1.0, max(0.0, localization_score(best)))
    return {
        "candidate_operator": best.get("operator_id"),
        "confidence": confidence,
        "evidence": (
            f"Candidate bottleneck {best.get('operator_id')} shows {best.get('main_signal')} "
            f"with amplification {best.get('amplification', 0.0):.3f} and divergence jump "
            f"{best.get('divergence_jump', 0.0):.3f}."
        ),
        "candidate": best,
        "alternative_candidates": ranked[1:4],
    }


def _call_index(operator_id: str) -> int:
    if "#" not in operator_id:
        return 0
    try:
        return int(operator_id.rsplit("#", 1)[1])
    except ValueError:
        return 0

