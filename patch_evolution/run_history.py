import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_HISTORY_DIR = "experiment_history"
HISTORY_SCHEMA_VERSION = 1


def write_run_history(
    summary: Dict[str, Any],
    history_dir: str = DEFAULT_HISTORY_DIR,
    summary_path: Optional[str] = None,
    exec_model_name: Optional[str] = None,
) -> Dict[str, str]:
    """Append a compact run record and refresh latest pointers for future sessions."""
    record = build_history_record(summary, summary_path=summary_path, exec_model_name=exec_model_name)
    root = Path(history_dir) / str(record.get("dataset") or "unknown")
    root.mkdir(parents=True, exist_ok=True)

    run_id = str(record.get("run_id") or "unknown_run")
    run_path = root / f"{run_id}.json"
    latest_json_path = root / "latest.json"
    runs_jsonl_path = root / "runs.jsonl"
    latest_context_path = root / "latest_context.md"

    _write_json(run_path, record)
    _write_json(latest_json_path, record)
    with runs_jsonl_path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    latest_context_path.write_text(format_latest_context(record), encoding="utf-8")

    return {
        "history_dir": str(root),
        "run_record_path": str(run_path),
        "latest_json_path": str(latest_json_path),
        "runs_jsonl_path": str(runs_jsonl_path),
        "latest_context_path": str(latest_context_path),
    }


def build_history_record(
    summary: Dict[str, Any],
    summary_path: Optional[str] = None,
    exec_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    compilation = summary.get("workflow_compilation") or {}
    compiled_eval = summary.get("compiled_workflow_evaluation") or {}
    consolidation = summary.get("consolidation") or {}
    online_metrics = _metrics_from_summary(summary)
    compiled_metrics = _metrics_from_summary(compiled_eval) if compiled_eval else None
    latest_phase = "compiled_eval" if compiled_metrics else "online_repair"
    latest_metrics = compiled_metrics or online_metrics

    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": summary.get("dataset"),
        "run_id": summary.get("run_id"),
        "workflow_id": summary.get("workflow_id"),
        "workflow_dir": summary.get("workflow_dir"),
        "exec_model_name": exec_model_name,
        "repair_rounds_completed": summary.get("repair_rounds_completed"),
        "latest_phase": latest_phase,
        "latest_metrics": latest_metrics,
        "online_repair_metrics": online_metrics,
        "compiled_eval_metrics": compiled_metrics,
        "patch_stats": {
            "accepted_patch_count": summary.get("accepted_patch_count", 0),
            "rejected_patch_count": summary.get("rejected_patch_count", 0),
            "skipped_patch_count": summary.get("skipped_patch_count", 0),
            "static_rejected_patch_count": summary.get("static_rejected_patch_count", 0),
            "duplicate_patch_count": summary.get("duplicate_patch_count", 0),
            "shadow_retry_count": summary.get("shadow_retry_count", 0),
            "accepted_patch_ids": summary.get("accepted_patch_ids") or [],
        },
        "compiled_workflow": {
            "compiled_workflow_dir": compilation.get("compiled_workflow_dir"),
            "compiled_patch_count": compilation.get("compiled_patch_count"),
            "inserted_nodes": compilation.get("inserted_nodes") or [],
            "content_repair_batches": compilation.get("content_repair_batches") or [],
            "compiled_patch_ids": compilation.get("compiled_patch_ids") or [],
        },
        "content_node_counts": {
            "operator_counts": compiled_eval.get("operator_counts") or {},
            "status_counts": compiled_eval.get("content_node_status_counts") or {},
            "accept_counts": compiled_eval.get("content_node_accept_counts") or {},
        },
        "failure_samples": _failure_samples(summary, compiled_eval),
        "paths": {
            "summary_path": summary_path,
            "trace_root": summary.get("trace_root"),
            "patch_registry_dir": summary.get("patch_registry_dir"),
            "repair_events_path": summary.get("repair_events_path"),
            "consolidation_path": consolidation.get("path"),
            "compiled_workflow_dir": compilation.get("compiled_workflow_dir"),
            "compiled_patches_path": compilation.get("compiled_patches_path"),
            "workflow_edit_manifest_path": compilation.get("edit_manifest_path"),
            "compiled_eval_summary_path": compiled_eval.get("path"),
        },
        "round_summaries": _compact_rounds(summary.get("round_summaries") or []),
    }


def format_latest_context(record: Dict[str, Any]) -> str:
    metrics = record.get("latest_metrics") or {}
    online = record.get("online_repair_metrics") or {}
    compiled = record.get("compiled_eval_metrics") or {}
    paths = record.get("paths") or {}
    failures = record.get("failure_samples") or []
    patch_stats = record.get("patch_stats") or {}
    workflow = record.get("compiled_workflow") or {}

    lines = [
        "# Latest Workflow Repair Run",
        "",
        "Use this file as compact context when opening a new Codex conversation.",
        "",
        f"- dataset: {record.get('dataset')}",
        f"- run_id: {record.get('run_id')}",
        f"- workflow_id: {record.get('workflow_id')}",
        f"- model: {record.get('exec_model_name')}",
        f"- latest_phase: {record.get('latest_phase')}",
        f"- latest_score: {metrics.get('average_score')}",
        f"- latest_success/failure: {metrics.get('success_count')} / {metrics.get('failure_count')}",
        f"- online_score: {online.get('average_score')}",
        f"- compiled_score: {compiled.get('average_score')}",
        f"- repair_rounds_completed: {record.get('repair_rounds_completed')}",
        f"- accepted_patch_count: {patch_stats.get('accepted_patch_count')}",
        f"- inserted_nodes: {', '.join(_node_names(workflow.get('inserted_nodes') or []))}",
        "",
        "## Key Paths",
        f"- summary: {paths.get('summary_path')}",
        f"- repair_events: {paths.get('repair_events_path')}",
        f"- trace_root: {paths.get('trace_root')}",
        f"- patch_registry: {paths.get('patch_registry_dir')}",
        f"- compiled_workflow: {paths.get('compiled_workflow_dir')}",
        f"- compiled_eval_summary: {paths.get('compiled_eval_summary_path')}",
        "",
        "## Remaining Failures",
    ]
    if failures:
        for failure in failures[:30]:
            lines.append(
                f"- sample {failure.get('sample_id')}: pred={failure.get('prediction')} "
                f"expected={failure.get('expected')} score={failure.get('score')}"
            )
    else:
        lines.append("- none recorded")
    lines.extend(
        [
            "",
            "## Next Useful Action",
            "- If compiled_eval_summary is empty but compiled_workflow exists, rerun with --eval_compiled_workflow.",
        ]
    )
    return "\n".join(lines) + "\n"


def _metrics_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_count": summary.get("sample_count"),
        "average_score": summary.get("average_score"),
        "baseline_average_score": summary.get("baseline_average_score"),
        "success_threshold": summary.get("success_threshold"),
        "success_count": summary.get("success_count"),
        "failure_count": summary.get("failure_count"),
        "exact_count": summary.get("exact_count"),
        "partial_count": summary.get("partial_count"),
        "zero_count": summary.get("zero_count"),
    }


def _failure_samples(summary: Dict[str, Any], compiled_eval: Dict[str, Any]) -> List[Dict[str, Any]]:
    if compiled_eval.get("failure_samples"):
        return compiled_eval["failure_samples"]
    if summary.get("failure_samples"):
        return summary["failure_samples"]
    return []


def _compact_rounds(round_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for item in round_summaries:
        compact.append(
            {
                "repair_round": item.get("repair_round"),
                "average_score": item.get("average_score"),
                "baseline_average_score": item.get("baseline_average_score"),
                "success_count": item.get("success_count"),
                "failure_count": item.get("failure_count"),
                "accepted_patch_count": item.get("accepted_patch_count"),
                "rejected_patch_count": item.get("rejected_patch_count"),
                "skipped_patch_count": item.get("skipped_patch_count"),
            }
        )
    return compact


def _node_names(nodes: List[Dict[str, Any]]) -> List[str]:
    return [str(node.get("node_name")) for node in nodes if node.get("node_name")]


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
