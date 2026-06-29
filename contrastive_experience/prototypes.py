from collections import Counter, defaultdict
from typing import Any, Dict, List


def _operator_trace(sample: Dict[str, Any], operator_id: str) -> Dict[str, Any]:
    for trace in sample.get("operator_traces", []):
        if trace.get("operator_id") == operator_id:
            return trace
    return {}


def _signature(op_trace: Dict[str, Any]) -> tuple:
    return (
        op_trace.get("answer_type", "unknown"),
        op_trace.get("output_style", "unknown"),
        op_trace.get("schema_type", "unknown"),
        op_trace.get("parse_status", "unknown"),
    )


def _sample_summary(sample: Dict[str, Any], operator_id: str) -> Dict[str, Any]:
    op_trace = _operator_trace(sample, operator_id)
    return {
        "sample_id": sample.get("sample_id"),
        "prediction": sample.get("prediction"),
        "expected": sample.get("expected"),
        "sample_score": sample.get("sample_score"),
        "operator_output": op_trace.get("output_preview"),
        "operator_input": op_trace.get("input_preview"),
        "features": {
            "answer_type": op_trace.get("answer_type"),
            "output_style": op_trace.get("output_style"),
            "schema_type": op_trace.get("schema_type"),
            "parse_status": op_trace.get("parse_status"),
        },
    }


def select_prototypes(
    success_traces: List[Dict[str, Any]],
    failure_traces: List[Dict[str, Any]],
    candidate_operator: str,
    limit: int = 3,
) -> Dict[str, Any]:
    failure_groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for sample in failure_traces:
        failure_groups[_signature(_operator_trace(sample, candidate_operator))].append(sample)

    if failure_groups:
        dominant_signature, dominant_failures = max(failure_groups.items(), key=lambda item: len(item[1]))
    else:
        dominant_signature, dominant_failures = tuple(), []

    success_ranked = []
    for sample in success_traces:
        sig = _signature(_operator_trace(sample, candidate_operator))
        match_score = sum(1 for a, b in zip(sig, dominant_signature) if a == b)
        success_ranked.append((match_score, sample))
    success_ranked.sort(key=lambda item: item[0], reverse=True)

    criteria = [
        "same candidate operator",
        "similar answer_type when available",
        "stable success parse_status/output_style",
        "dominant failure signature contrast",
    ]

    return {
        "candidate_operator": candidate_operator,
        "success_prototypes": [_sample_summary(sample, candidate_operator) for _, sample in success_ranked[:limit]],
        "failure_prototypes": [_sample_summary(sample, candidate_operator) for sample in dominant_failures[:limit]],
        "matching_criteria": criteria,
        "dominant_failure_signature": list(dominant_signature),
        "failure_signature_counts": {"|".join(map(str, key)): len(value) for key, value in failure_groups.items()},
    }

