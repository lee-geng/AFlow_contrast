import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional


def _distribution(values: Iterable[str]) -> Dict[str, float]:
    counts = Counter(v or "unknown" for v in values)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    score = 0.0
    for key, value in p.items():
        other = q.get(key, 0.0)
        if value > 0 and other > 0:
            score += value * math.log(value / other, 2)
    return score


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    midpoint = {key: (p.get(key, 0.0) + q.get(key, 0.0)) / 2 for key in keys}
    return (_kl_divergence(p, midpoint) + _kl_divergence(q, midpoint)) / 2


def _operator_buckets(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in records:
        for op_trace in sample.get("operator_traces", []):
            key = op_trace.get("operator_id") or f"{op_trace.get('operator_type', 'operator')}#{op_trace.get('call_index', 0)}"
            buckets[key].append(op_trace)
    return buckets


def _top_value(traces: List[Dict[str, Any]], key: str) -> str:
    counts = Counter(trace.get(key) or "unknown" for trace in traces)
    return counts.most_common(1)[0][0] if counts else "unknown"


def _ok_rate(traces: List[Dict[str, Any]]) -> float:
    if not traces:
        return 0.0
    return sum(1 for trace in traces if trace.get("parse_status") == "ok") / len(traces)


def compute_operator_divergence(
    success_traces: List[Dict[str, Any]],
    failure_traces: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    beta: float = 0.5,
    gamma: float = 0.3,
) -> List[Dict[str, Any]]:
    weights = weights or {
        "schema_div": 0.35,
        "output_style_div": 0.40,
        "error_div": 0.25,
        "semantic_div": 0.0,
    }
    success_ops = _operator_buckets(success_traces)
    failure_ops = _operator_buckets(failure_traces)
    ranked = []

    for operator_id in sorted(set(success_ops) | set(failure_ops)):
        s_ops = success_ops.get(operator_id, [])
        f_ops = failure_ops.get(operator_id, [])
        schema_div = js_divergence(
            _distribution(op.get("schema_type", "unknown") for op in s_ops),
            _distribution(op.get("schema_type", "unknown") for op in f_ops),
        )
        style_div = js_divergence(
            _distribution(op.get("output_style", "unknown") for op in s_ops),
            _distribution(op.get("output_style", "unknown") for op in f_ops),
        )
        error_div = js_divergence(
            _distribution(op.get("parse_status", "unknown") for op in s_ops),
            _distribution(op.get("parse_status", "unknown") for op in f_ops),
        )
        input_schema_div = js_divergence(
            _distribution(op.get("input_schema_type", "unknown") for op in s_ops),
            _distribution(op.get("input_schema_type", "unknown") for op in f_ops),
        )
        input_style_div = js_divergence(
            _distribution(op.get("input_output_style", "unknown") for op in s_ops),
            _distribution(op.get("input_output_style", "unknown") for op in f_ops),
        )
        output_div = (
            weights.get("schema_div", 0.0) * schema_div
            + weights.get("output_style_div", 0.0) * style_div
            + weights.get("error_div", 0.0) * error_div
        )
        input_div = 0.5 * input_schema_div + 0.5 * input_style_div
        amplification = output_div - input_div
        error_gap = max(0.0, _ok_rate(s_ops) - _ok_rate(f_ops))
        bottleneck_score = output_div + beta * amplification + gamma * error_gap

        signals = []
        if style_div >= schema_div and style_div > 0:
            signals.append("output_style divergence")
        if schema_div > 0:
            signals.append("schema divergence")
        if error_div > 0 or error_gap > 0:
            signals.append("parse/error divergence")

        ranked.append(
            {
                "operator_id": operator_id,
                "operator_type": _top_value(s_ops + f_ops, "operator_type"),
                "bottleneck_score": bottleneck_score,
                "input_div": input_div,
                "output_div": output_div,
                "amplification": amplification,
                "schema_div": schema_div,
                "output_style_div": style_div,
                "error_div": error_div,
                "error_gap": error_gap,
                "main_signal": "/".join(signals) if signals else "weak divergence",
                "success_pattern": {
                    "output_style_top": _top_value(s_ops, "output_style"),
                    "schema_type_top": _top_value(s_ops, "schema_type"),
                    "parse_ok_rate": _ok_rate(s_ops),
                },
                "failure_pattern": {
                    "output_style_top": _top_value(f_ops, "output_style"),
                    "schema_type_top": _top_value(f_ops, "schema_type"),
                    "parse_ok_rate": _ok_rate(f_ops),
                },
            }
        )

    return sorted(ranked, key=lambda item: item["bottleneck_score"], reverse=True)
