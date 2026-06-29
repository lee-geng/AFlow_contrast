import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


QA_DATASETS = {"HotpotQA", "DROP"}


def dataset_success_threshold(dataset: Optional[str], override: Optional[float] = None) -> float:
    if override is not None:
        return override
    return 0.3 if dataset in QA_DATASETS else 1.0


def load_trace_records(trace_file: Union[str, Path]) -> List[Dict[str, Any]]:
    records = []
    with Path(trace_file).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_final_result(record: Dict[str, Any], success_threshold: Optional[float] = None) -> str:
    if record.get("final_result") in {"success", "failure"}:
        return record["final_result"]
    threshold = dataset_success_threshold(record.get("dataset"), success_threshold)
    try:
        score = float(record.get("sample_score", 0))
    except (TypeError, ValueError):
        score = 0.0
    return "success" if score >= threshold else "failure"


def group_traces(records: Iterable[Dict[str, Any]], success_threshold: Optional[float] = None) -> Dict[str, Any]:
    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    failure_types = Counter()

    for record in records:
        result = normalize_final_result(record, success_threshold=success_threshold)
        record = dict(record)
        record["final_result"] = result
        if result == "success":
            successes.append(record)
        else:
            failures.append(record)
            if record.get("error_type"):
                failure_types[record["error_type"]] += 1
            else:
                failure_types["score_below_threshold"] += 1

    total = len(successes) + len(failures)
    return {
        "success_traces": successes,
        "failure_traces": failures,
        "stats": {
            "total_samples": total,
            "success_count": len(successes),
            "failure_count": len(failures),
            "success_rate": len(successes) / total if total else 0.0,
            "failure_type_distribution": dict(failure_types),
        },
    }
