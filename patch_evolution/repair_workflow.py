import argparse
import asyncio
import ast
import importlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Keep legacy relative paths such as config/, data/, and workspace/ stable even
# when this script is launched by absolute path from another directory.
os.chdir(PROJECT_ROOT)

from contrastive_experience.grouping import dataset_success_threshold
from contrastive_experience.trace_context import (
    TraceConfig,
    finish_sample_trace,
    make_sample_id,
    reset_trace_config,
    set_trace_config,
    start_sample_trace,
)
from contrastive_experience.trace_logger import default_run_id
from contrastive_experience.verification import summarize_verification
from patch_evolution.consolidator import OfflineConsolidator
from patch_evolution.guarded_runtime import PatchRuntimeConfig, reset_patch_runtime, set_patch_runtime
from patch_evolution.online_patch import build_online_patch_candidates
from patch_evolution.patch_registry import ACTIVE_STATUSES, PatchRegistry
from patch_evolution.patch_spec import referenced_prompt_names, validate_patch_schema
from patch_evolution.patch_synthesizer import PatchSynthesizer
from patch_evolution.probe import ProbeBeforeEdit
from patch_evolution.run_history import DEFAULT_HISTORY_DIR, write_run_history
from patch_evolution.static_validator import StaticPatchValidator
from patch_evolution.trace_ir import contrastive_record_to_trace_ir
from patch_evolution.workflow_compiler import WorkflowPatchCompiler
from scripts.async_llm import LLMsConfig
from scripts.evaluator import DatasetType, Evaluator
from tqdm import tqdm


DATASETS = ["DROP", "HotpotQA", "MATH", "GSM8K", "MBPP", "HumanEval", "LiveCodeBench"]


@dataclass
class EvaluationOutcome:
    index: int
    sample_id: str
    problem: Dict[str, Any]
    result: Optional[tuple]
    record: Dict[str, Any]
    baseline_score: float
    score: float
    cost: float


class WorkflowRepairRunner:
    def __init__(
        self,
        dataset: DatasetType,
        workflow_dir: str,
        exec_model_name: str,
        trace_dir: str,
        patch_registry_dir: str,
        output_dir: str,
        run_id: Optional[str] = None,
        trace_full_io: bool = False,
        trace_preview_chars: int = 512,
        success_threshold: Optional[float] = None,
        matched_successes: int = 3,
        regression_tolerance: float = 0.05,
        online_patch: bool = True,
        enable_llm_repair: bool = True,
        enable_llm_patch_synthesis: bool = False,
        llm_patch_max_candidates: int = 1,
        consolidate: bool = True,
        compile_workflow: bool = False,
        compiled_workflow_dir: Optional[str] = None,
        compile_min_credit: float = 0.0,
        compile_max_patches: int = 8,
        compile_content_nodes: bool = False,
        content_repair_families: Optional[List[str]] = None,
        content_repair_batches: Optional[List[str]] = None,
        eval_compiled_workflow: bool = False,
        compiled_eval_max_samples: Optional[int] = None,
        history_dir: str = DEFAULT_HISTORY_DIR,
        save_history: bool = True,
        is_test: bool = False,
        stop_when_no_new_patches: bool = False,
        show_progress: bool = True,
    ):
        self.dataset = dataset
        self.workflow_dir = Path(workflow_dir)
        self.exec_model_name = exec_model_name
        self.trace_dir = trace_dir
        self.patch_registry_dir = patch_registry_dir
        self.output_dir = Path(output_dir)
        self.run_id = run_id or default_run_id()
        self.trace_full_io = trace_full_io
        self.trace_preview_chars = trace_preview_chars
        self.success_threshold = dataset_success_threshold(dataset, success_threshold)
        self.matched_successes = matched_successes
        self.regression_tolerance = regression_tolerance
        self.online_patch = online_patch
        self.enable_llm_repair = enable_llm_repair
        self.enable_llm_patch_synthesis = enable_llm_patch_synthesis
        self.llm_patch_max_candidates = max(1, llm_patch_max_candidates)
        self.consolidate = consolidate
        self.compile_workflow = compile_workflow
        self.compiled_workflow_dir = compiled_workflow_dir
        self.compile_min_credit = compile_min_credit
        self.compile_max_patches = compile_max_patches
        self.content_repair_batches = content_repair_batches
        self.compile_content_nodes = compile_content_nodes or bool(content_repair_batches)
        self.content_repair_families = content_repair_families
        self.eval_compiled_workflow = eval_compiled_workflow
        self.compiled_eval_max_samples = compiled_eval_max_samples
        self.history_dir = history_dir
        self.save_history = save_history
        self.is_test = is_test
        self.stop_when_no_new_patches = stop_when_no_new_patches
        self.current_repair_round = 1
        self.total_repair_rounds = 1
        self.show_progress = show_progress

        self.repair_root = self.output_dir / dataset / self.run_id
        self.log_path = self.repair_root / "logs"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.events_path = self.repair_root / "repair_events.jsonl"
        self.summary_path = self.repair_root / "summary.json"

        self.registry = PatchRegistry(patch_registry_dir)
        self.static_validator = StaticPatchValidator(dataset)
        self.workflow_class = load_workflow_class(self.workflow_dir)
        self.round_id = parse_round_id(self.workflow_dir)
        self.workflow_id = self.workflow_dir.name

        models_config = LLMsConfig.default()
        self.exec_llm_config = models_config.get(exec_model_name)
        evaluator = Evaluator(eval_path=str(self.log_path))
        data_path = evaluator._get_data_path(dataset, is_test)
        benchmark_class = evaluator.dataset_configs[dataset]
        self.benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=str(self.log_path))
        self.graph = self.workflow_class(name=dataset, llm_config=self.exec_llm_config, dataset=dataset)
        self.patch_synthesizer = (
            PatchSynthesizer(llm=getattr(self.graph, "llm", None))
            if self.enable_llm_patch_synthesis
            else None
        )

    async def run(
        self,
        indices: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
        start_index: int = 0,
        repair_rounds: int = 1,
    ):
        data = await self.benchmark.load_data(indices)
        if indices is None and start_index:
            data = data[start_index:]
        if max_samples is not None:
            data = data[:max_samples]

        round_summaries: List[Dict[str, Any]] = []
        all_repair_events: List[Dict[str, Any]] = []
        final_outcomes: List[EvaluationOutcome] = []
        self.total_repair_rounds = max(1, repair_rounds)

        for repair_round in range(1, self.total_repair_rounds + 1):
            self.current_repair_round = repair_round
            outcomes, repair_events = await self._run_single_repair_round(data, indices, start_index)
            final_outcomes = outcomes
            all_repair_events.extend(repair_events)
            round_summary = self._build_summary(outcomes, repair_events)
            round_summary["repair_round"] = repair_round
            self._write_json(self.repair_root / f"round_{repair_round}_summary.json", round_summary)
            round_summaries.append(round_summary)

            accepted_count = round_summary["accepted_patch_count"]
            if self.stop_when_no_new_patches and accepted_count == 0:
                break

        summary = self._build_multi_round_summary(round_summaries, all_repair_events, final_outcomes)
        if self.consolidate:
            summary["consolidation"] = self._write_consolidation()
        if self.compile_workflow:
            summary["workflow_compilation"] = self._compile_workflow_edit()
            if self.eval_compiled_workflow:
                summary["compiled_workflow_evaluation"] = await self._evaluate_compiled_workflow(
                    summary["workflow_compilation"],
                    data,
                    indices,
                    start_index,
                )
        self._write_json(self.summary_path, summary)
        if self.save_history:
            summary["run_history"] = write_run_history(
                summary,
                history_dir=self.history_dir,
                summary_path=str(self.summary_path),
                exec_model_name=self.exec_model_name,
            )
            self._write_json(self.summary_path, summary)
        return summary

    async def _run_single_repair_round(self, data: List[Dict[str, Any]], indices, start_index: int):
        success_history: List[EvaluationOutcome] = []
        outcomes: List[EvaluationOutcome] = []
        repair_events: List[Dict[str, Any]] = []

        accepted_count = 0
        rejected_count = 0
        skipped_count = 0
        failures_seen = 0
        iterator = enumerate(data)
        progress = None
        if self.show_progress:
            progress = tqdm(
                iterator,
                total=len(data),
                desc=f"Repair round {self.current_repair_round}/{self.total_repair_rounds}",
                unit="sample",
                dynamic_ncols=True,
            )
        else:
            progress = iterator

        for offset, problem in progress:
            original_index = indices[offset] if indices is not None else start_index + offset
            outcome = await self._evaluate_one(original_index, problem, phase="main", patch_enabled=True)
            outcomes.append(outcome)
            if self._is_success(outcome.score):
                success_history.append(outcome)
                self._update_progress(progress, failures_seen, accepted_count, rejected_count, skipped_count)
                continue
            failures_seen += 1
            if self.online_patch:
                events = await self._attempt_online_repair(outcome, problem, success_history)
                accepted_event = None
                for event in events:
                    repair_events.append(event)
                    self._append_jsonl(self.events_path, event)
                    if event.get("status") == "accepted":
                        accepted_event = event
                        accepted_count += 1
                    elif event.get("status") == "rejected":
                        rejected_count += 1
                    elif event.get("status") == "skipped":
                        skipped_count += 1
                if accepted_event:
                    outcome.score = _as_float(accepted_event.get("repaired_score"))
                    outcome.cost = _as_float(accepted_event.get("repaired_cost"))
                    if self._is_success(outcome.score):
                        success_history.append(outcome)
            self._update_progress(progress, failures_seen, accepted_count, rejected_count, skipped_count)
        return outcomes, repair_events

    def _update_progress(self, progress, failures_seen: int, accepted_count: int, rejected_count: int, skipped_count: int) -> None:
        if self.show_progress and hasattr(progress, "set_postfix"):
            progress.set_postfix(
                fail=failures_seen,
                accepted=accepted_count,
                rejected=rejected_count,
                skipped=skipped_count,
                refresh=False,
            )

    async def _evaluate_one(
        self,
        index: int,
        problem: Dict[str, Any],
        phase: str,
        patch_enabled: bool,
        graph: Optional[Any] = None,
        workflow_id: Optional[str] = None,
    ) -> EvaluationOutcome:
        active_graph = graph or self.graph
        active_workflow_id = workflow_id or self.workflow_id
        trace_config = TraceConfig.create(
            enabled=True,
            dataset=self.dataset,
            trace_dir=self.trace_dir,
            run_id=self.run_id,
            round_id=None,
            workflow_id=f"{active_workflow_id}_repair_round_{self.current_repair_round}_{phase}",
            trace_full_io=self.trace_full_io,
            trace_preview_chars=self.trace_preview_chars,
            success_threshold=self.success_threshold,
        )
        trace_token = set_trace_config(trace_config)
        patch_token = set_patch_runtime(
            PatchRuntimeConfig(
                enabled=patch_enabled,
                registry_dir=self.patch_registry_dir,
                llm=getattr(active_graph, "llm", None),
            )
        )
        sample_id = make_sample_id(index, problem)
        sample_token = start_sample_trace(sample_id, problem)
        result = None
        try:
            result = await self.benchmark.evaluate_problem(problem, active_graph)
            trace_result = self.benchmark._trace_result_from_tuple(result, trace_config=trace_config)
        except Exception as exc:
            trace_result = {
                "final_result": "failure",
                "prediction": str(exc),
                "expected": None,
                "sample_score": 0.0,
                "cost": 0.0,
            }
        finally:
            record = finish_sample_trace(sample_token, trace_result)
            reset_trace_config(trace_token)
            reset_patch_runtime(patch_token)

        record = record or {}
        score = _as_float(record.get("sample_score"))
        cost = _as_float(record.get("cost"))
        return EvaluationOutcome(
            index=index,
            sample_id=sample_id,
            problem=problem,
            result=result,
            record=record,
            baseline_score=score,
            score=score,
            cost=cost,
        )

    async def _evaluate_compiled_workflow(
        self,
        compilation: Dict[str, Any],
        data: List[Dict[str, Any]],
        indices: Optional[List[int]],
        start_index: int,
    ) -> Dict[str, Any]:
        compiled_dir = Path(compilation.get("compiled_workflow_dir", ""))
        if not compiled_dir.exists():
            return {
                "status": "skipped",
                "reason": "compiled_workflow_dir_missing",
                "compiled_workflow_dir": str(compiled_dir),
            }

        eval_data = data
        if self.compiled_eval_max_samples is not None:
            eval_data = data[: self.compiled_eval_max_samples]
        compiled_workflow_class = load_workflow_class(compiled_dir)
        compiled_graph = compiled_workflow_class(name=self.dataset, llm_config=self.exec_llm_config, dataset=self.dataset)
        compiled_workflow_id = compiled_dir.name

        outcomes: List[EvaluationOutcome] = []
        progress = enumerate(eval_data)
        if self.show_progress:
            progress = tqdm(
                progress,
                total=len(eval_data),
                desc="Evaluating compiled workflow",
                unit="sample",
                dynamic_ncols=True,
            )

        previous_round = self.current_repair_round
        self.current_repair_round = 0
        try:
            for offset, problem in progress:
                original_index = indices[offset] if indices is not None else start_index + offset
                outcome = await self._evaluate_one(
                    original_index,
                    problem,
                    phase="compiled_eval",
                    patch_enabled=False,
                    graph=compiled_graph,
                    workflow_id=compiled_workflow_id,
                )
                outcomes.append(outcome)
        finally:
            self.current_repair_round = previous_round

        summary = self._build_evaluation_summary(
            outcomes,
            workflow_dir=str(compiled_dir),
            workflow_id=compiled_workflow_id,
            phase="compiled_eval",
        )
        output_path = self.repair_root / "compiled_workflow_eval_summary.json"
        self._write_json(output_path, summary)
        summary["path"] = str(output_path)
        return summary

    async def _attempt_online_repair(
        self,
        failed_outcome: EvaluationOutcome,
        failed_problem: Dict[str, Any],
        success_history: List[EvaluationOutcome],
    ) -> List[Dict[str, Any]]:
        trace_ir = contrastive_record_to_trace_ir(failed_outcome.record)
        diagnosis = ProbeBeforeEdit().diagnose(trace_ir)
        candidates = build_online_patch_candidates(
            failed_outcome.record,
            diagnosis=diagnosis,
            enable_llm_repair=self.enable_llm_repair,
        )
        if self.patch_synthesizer is not None:
            try:
                synthesized = await self.patch_synthesizer.synthesize_failure_family(
                    failed_outcome.record,
                    diagnosis,
                    max_candidates=self.llm_patch_max_candidates,
                )
                candidates = self._dedupe_patch_candidates([*candidates, *synthesized])
            except Exception:
                candidates = self._dedupe_patch_candidates(candidates)
        if not candidates:
            return [
                self._repair_event(
                    failed_outcome,
                    diagnosis,
                    patch=None,
                    status="skipped",
                    reason="no_observable_patch_candidate",
                )
            ]

        events = []
        for patch in candidates:
            ok, errors = validate_patch_schema(patch)
            if not ok:
                events.append(
                    self._repair_event(
                        failed_outcome,
                        diagnosis,
                        patch=patch,
                        status="skipped",
                        reason="invalid_patch_schema",
                        extra={"schema_errors": errors},
                    )
                )
                continue
            static_result = self.static_validator.validate(patch)
            patch = static_result.normalized_patch or patch
            if not static_result.ok:
                self.registry.save_patch(patch)
                events.append(
                    self._repair_event(
                        failed_outcome,
                        diagnosis,
                        patch=patch,
                        status="static_rejected",
                        reason=static_result.status,
                        extra={"static_validation": {"status": static_result.status, "errors": static_result.errors}},
                    )
                )
                continue

            existing = self.registry.find_by_patch_key(patch.get("patch_key")) if patch.get("patch_key") else None
            if existing is None:
                existing = self.registry.load_patch(patch["patch_id"]) if self.registry.has_patch(patch["patch_id"]) else None
            retrying_shadow_patch = False
            if existing:
                self._update_patch_memory(existing, failed_outcome)
                existing = self.registry.load_patch(existing["patch_id"])
            if existing and existing.get("status") in ACTIVE_STATUSES:
                events.append(
                    self._repair_event(
                        failed_outcome,
                        diagnosis,
                        patch=existing,
                        status="skipped",
                        reason="duplicate_patch_already_active",
                        extra={"duplicate_patch_key": existing.get("patch_key")},
                    )
                )
                continue
            if existing and existing.get("status") in {"disabled", "static_rejected"}:
                events.append(
                    self._repair_event(
                        failed_outcome,
                        diagnosis,
                        patch=existing,
                        status="skipped",
                        reason=f"duplicate_patch_previously_{existing.get('status')}",
                        extra={"duplicate_patch_key": existing.get("patch_key")},
                    )
                )
                continue
            if existing and existing.get("status") == "shadow_rejected":
                if self._should_retry_shadow_rejected(existing, failed_outcome):
                    patch = existing
                    retrying_shadow_patch = True
                else:
                    events.append(
                        self._repair_event(
                            failed_outcome,
                            diagnosis,
                            patch=existing,
                            status="skipped",
                            reason="duplicate_patch_previously_shadow_rejected",
                            extra={"duplicate_patch_key": existing.get("patch_key")},
                        )
                    )
                    continue

            if not retrying_shadow_patch:
                self._initialize_patch_memory(patch, failed_outcome)
            patch["status"] = "active_guarded"
            self.registry.save_patch(patch)

            repaired = await self._evaluate_one(failed_outcome.index, failed_problem, phase="target_replay", patch_enabled=True)
            matched_successes = success_history[-self.matched_successes :] if self.matched_successes > 0 else []
            regressions = 0
            replayed_successes = []
            for success in matched_successes:
                replay = await self._evaluate_one(success.index, success.problem, phase="success_replay", patch_enabled=True)
                replayed_successes.append(
                    {
                        "sample_id": success.sample_id,
                        "baseline_score": success.score,
                        "replay_score": replay.score,
                    }
                )
                if replay.score < self.success_threshold or replay.score + self.regression_tolerance < success.score:
                    regressions += 1

            target_repair_rate = 1.0 if self._is_success(repaired.score) else max(0.0, repaired.score - failed_outcome.score)
            success_regression_rate = regressions / len(matched_successes) if matched_successes else 0.0
            cost_increase = _relative_cost_increase(failed_outcome.cost, repaired.cost)
            verification = summarize_verification(
                target_repair_rate=target_repair_rate,
                success_regression_rate=success_regression_rate,
                non_target_failure_increase=0.0,
                cost_increase=cost_increase,
            )

            patch = self.registry.load_patch(patch["patch_id"])
            patch["status"] = "active_guarded" if verification["accepted"] else "shadow_rejected"
            self._record_patch_replay_result(patch, failed_outcome, verification["accepted"])
            telemetry = patch.setdefault("telemetry", {})
            telemetry["average_gain"] = max(0.0, repaired.score - failed_outcome.score)
            telemetry["repair_success_count"] = telemetry.get("repair_success_count", 0) + (1 if self._is_success(repaired.score) else 0)
            telemetry["repair_failure_count"] = telemetry.get("repair_failure_count", 0) + (0 if self._is_success(repaired.score) else 1)
            telemetry["regression_count"] = telemetry.get("regression_count", 0) + regressions
            telemetry["average_cost_overhead"] = cost_increase
            self.registry.save_patch(patch)

            event = self._repair_event(
                failed_outcome,
                diagnosis,
                patch=patch,
                status="accepted" if verification["accepted"] else "rejected",
                reason="verification_passed" if verification["accepted"] else "verification_failed",
                extra={
                    "baseline_score": failed_outcome.score,
                    "repaired_score": repaired.score,
                    "repaired_cost": repaired.cost,
                    "verification": verification,
                    "static_validation": {"status": static_result.status, "errors": static_result.errors},
                    "matched_successes": replayed_successes,
                    "retrying_shadow_patch": retrying_shadow_patch,
                },
            )
            events.append(event)
            if verification["accepted"]:
                return events
        return events

    def _repair_event(
        self,
        failed_outcome: EvaluationOutcome,
        diagnosis: Dict[str, Any],
        patch: Optional[Dict[str, Any]],
        status: str,
        reason: str,
        extra: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        event = {
            "dataset": self.dataset,
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "repair_round": self.current_repair_round,
            "sample_id": failed_outcome.sample_id,
            "status": status,
            "reason": reason,
            "diagnosis": diagnosis,
            "patch_id": patch.get("patch_id") if patch else None,
            "target_operator": patch.get("target_operator") if patch else diagnosis.get("target_operator"),
        }
        if patch:
            event["patch"] = patch
        if extra:
            event.update(extra)
        return event

    def _build_evaluation_summary(
        self,
        outcomes: List[EvaluationOutcome],
        workflow_dir: str,
        workflow_id: str,
        phase: str,
    ) -> Dict[str, Any]:
        scores = [outcome.score for outcome in outcomes]
        records = [outcome.record for outcome in outcomes]
        failures = [outcome for outcome in outcomes if not self._is_success(outcome.score)]
        operator_counts = Counter()
        content_node_status_counts = Counter()
        content_node_accept_counts = Counter()
        for record in records:
            for op_trace in record.get("operator_traces") or []:
                operator_type = op_trace.get("operator_type")
                operator_counts[operator_type] += 1
                if op_trace.get("formatter_mode") != "content_repair":
                    continue
                status = self._content_node_status(op_trace)
                content_node_status_counts[f"{operator_type}:{status}"] += 1
                if status == "accepted":
                    content_node_accept_counts[operator_type] += 1
        return {
            "dataset": self.dataset,
            "workflow_dir": workflow_dir,
            "workflow_id": workflow_id,
            "run_id": self.run_id,
            "phase": phase,
            "sample_count": len(outcomes),
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "success_threshold": self.success_threshold,
            "success_count": sum(1 for score in scores if self._is_success(score)),
            "failure_count": len(failures),
            "exact_count": sum(1 for score in scores if score >= 1.0),
            "partial_count": sum(1 for score in scores if 0.0 < score < 1.0),
            "zero_count": sum(1 for score in scores if score == 0.0),
            "operator_counts": dict(operator_counts),
            "content_node_status_counts": dict(content_node_status_counts),
            "content_node_accept_counts": dict(content_node_accept_counts),
            "trace_root": str(Path(self.trace_dir) / self.dataset / self.run_id),
            "failure_samples": [
                {
                    "sample_id": outcome.sample_id,
                    "score": outcome.score,
                    "prediction": outcome.record.get("prediction"),
                    "expected": outcome.record.get("expected"),
                }
                for outcome in failures
            ],
        }

    def _content_node_status(self, op_trace: Dict[str, Any]) -> str:
        raw = op_trace.get("output_preview") or ""
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return "unknown"
        if isinstance(parsed, dict):
            return str(parsed.get("status") or "unknown")
        return "unknown"

    def _build_summary(self, outcomes: List[EvaluationOutcome], repair_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores = [outcome.score for outcome in outcomes]
        baseline_scores = [outcome.baseline_score for outcome in outcomes]
        accepted = [event for event in repair_events if event["status"] == "accepted"]
        rejected = [event for event in repair_events if event["status"] == "rejected"]
        skipped = [event for event in repair_events if event["status"] == "skipped"]
        static_rejected = [event for event in repair_events if event["status"] == "static_rejected"]
        return {
            "dataset": self.dataset,
            "workflow_dir": str(self.workflow_dir),
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "sample_count": len(outcomes),
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "baseline_average_score": sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0,
            "success_threshold": self.success_threshold,
            "success_count": sum(1 for score in scores if self._is_success(score)),
            "failure_count": sum(1 for score in scores if not self._is_success(score)),
            "repair_attempt_count": len(repair_events),
            "accepted_patch_count": len(accepted),
            "rejected_patch_count": len(rejected),
            "skipped_patch_count": len(skipped),
            "candidate_patch_count": len([event for event in repair_events if event.get("patch_id")]),
            "normalized_patch_count": len([event for event in repair_events if event.get("patch", {}).get("patch_key")]),
            "static_valid_patch_count": len(
                [event for event in repair_events if event.get("static_validation", {}).get("status") == "static_valid"]
            ),
            "static_rejected_patch_count": len(static_rejected),
            "static_rejection_reasons": _count_by(static_rejected, "reason"),
            "source_trace_replay_success_count": len(
                [event for event in repair_events if self._is_success(_as_float(event.get("repaired_score")))]
            ),
            "shadow_pool_pass_count": len(accepted),
            "active_guarded_patch_count": len(accepted),
            "duplicate_patch_count": len(
                [event for event in repair_events if str(event.get("reason", "")).startswith("duplicate_patch")]
            ),
            "shadow_retry_count": len([event for event in repair_events if event.get("retrying_shadow_patch")]),
            "forbidden_signal_patch_count": len(
                [event for event in static_rejected if event.get("reason") == "rejected_invalid_trigger_signal"]
            ),
            "scensemble_answer_repair_attempt_count": len(
                [
                    event
                    for event in repair_events
                    if event.get("target_operator") == "ScEnsemble"
                    and "answer" in (event.get("patch", {}).get("fixer", {}).get("writes") or [])
                ]
            ),
            "scensemble_answer_repair_count": len(
                [
                    event
                    for event in repair_events
                    if event.get("status") in {"accepted", "rejected"}
                    and event.get("target_operator") == "ScEnsemble"
                    and "answer" in (event.get("patch", {}).get("fixer", {}).get("writes") or [])
                ]
            ),
            "accepted_patch_ids": [event["patch_id"] for event in accepted if event.get("patch_id")],
            "trace_root": str(Path(self.trace_dir) / self.dataset / self.run_id),
            "patch_registry_dir": self.patch_registry_dir,
            "repair_events_path": str(self.events_path),
        }

    def _build_multi_round_summary(
        self,
        round_summaries: List[Dict[str, Any]],
        repair_events: List[Dict[str, Any]],
        final_outcomes: List[EvaluationOutcome],
    ) -> Dict[str, Any]:
        last = round_summaries[-1] if round_summaries else {}
        first = round_summaries[0] if round_summaries else {}
        accepted_events = [event for event in repair_events if event.get("status") == "accepted"]
        rejected_events = [event for event in repair_events if event.get("status") == "rejected"]
        skipped_events = [event for event in repair_events if event.get("status") == "skipped"]
        static_rejected_events = [event for event in repair_events if event.get("status") == "static_rejected"]
        return {
            "dataset": self.dataset,
            "workflow_dir": str(self.workflow_dir),
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "repair_rounds_completed": len(round_summaries),
            "sample_count": len(final_outcomes),
            "baseline_average_score": first.get("baseline_average_score", 0.0),
            "average_score": last.get("average_score", 0.0),
            "success_threshold": self.success_threshold,
            "success_count": last.get("success_count", 0),
            "failure_count": last.get("failure_count", 0),
            "repair_attempt_count": len(repair_events),
            "accepted_patch_count": len(accepted_events),
            "rejected_patch_count": len(rejected_events),
            "skipped_patch_count": len(skipped_events),
            "candidate_patch_count": len([event for event in repair_events if event.get("patch_id")]),
            "normalized_patch_count": len([event for event in repair_events if event.get("patch", {}).get("patch_key")]),
            "static_valid_patch_count": len(
                [event for event in repair_events if event.get("static_validation", {}).get("status") == "static_valid"]
            ),
            "static_rejected_patch_count": len(static_rejected_events),
            "static_rejection_reasons": _count_by(static_rejected_events, "reason"),
            "source_trace_replay_success_count": len(
                [event for event in repair_events if self._is_success(_as_float(event.get("repaired_score")))]
            ),
            "shadow_pool_pass_count": len(accepted_events),
            "active_guarded_patch_count": len(accepted_events),
            "duplicate_patch_count": len(
                [event for event in repair_events if str(event.get("reason", "")).startswith("duplicate_patch")]
            ),
            "shadow_retry_count": len([event for event in repair_events if event.get("retrying_shadow_patch")]),
            "forbidden_signal_patch_count": len(
                [event for event in static_rejected_events if event.get("reason") == "rejected_invalid_trigger_signal"]
            ),
            "scensemble_answer_repair_attempt_count": len(
                [
                    event
                    for event in repair_events
                    if event.get("target_operator") == "ScEnsemble"
                    and "answer" in (event.get("patch", {}).get("fixer", {}).get("writes") or [])
                ]
            ),
            "scensemble_answer_repair_count": len(
                [
                    event
                    for event in repair_events
                    if event.get("status") in {"accepted", "rejected"}
                    and event.get("target_operator") == "ScEnsemble"
                    and "answer" in (event.get("patch", {}).get("fixer", {}).get("writes") or [])
                ]
            ),
            "accepted_patch_ids": sorted({event["patch_id"] for event in accepted_events if event.get("patch_id")}),
            "trace_root": str(Path(self.trace_dir) / self.dataset / self.run_id),
            "patch_registry_dir": self.patch_registry_dir,
            "repair_events_path": str(self.events_path),
            "failure_samples": [
                {
                    "sample_id": outcome.sample_id,
                    "score": outcome.score,
                    "prediction": outcome.record.get("prediction"),
                    "expected": outcome.record.get("expected"),
                }
                for outcome in final_outcomes
                if not self._is_success(outcome.score)
            ],
            "round_summaries": round_summaries,
        }

    def _write_consolidation(self) -> Dict[str, Any]:
        patches = self.registry.load_all()
        result = OfflineConsolidator().consolidate(patches)
        output_path = self.registry.consolidated_dir / f"{self.dataset}_{self.run_id}_consolidation.json"
        self._write_json(output_path, result)
        return {"path": str(output_path), **result}

    def _compile_workflow_edit(self) -> Dict[str, Any]:
        output_dir = self.compiled_workflow_dir or str(self.repair_root / "compiled_workflows")
        result = WorkflowPatchCompiler(
            min_credit=self.compile_min_credit,
            max_patches=self.compile_max_patches,
            compile_content_nodes=self.compile_content_nodes,
            content_repair_families=self.content_repair_families,
            content_repair_batches=self.content_repair_batches,
        ).compile_from_registry(
            workflow_dir=str(self.workflow_dir),
            patch_registry_dir=self.patch_registry_dir,
            output_dir=output_dir,
            dataset=self.dataset,
            run_id=self.run_id,
        )
        return result.to_dict()

    def _is_success(self, score: float) -> bool:
        return score >= self.success_threshold

    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _initialize_patch_memory(self, patch: Dict[str, Any], failed_outcome: EvaluationOutcome) -> None:
        patch.setdefault("source_trace_ids", [])
        if failed_outcome.sample_id not in patch["source_trace_ids"]:
            patch["source_trace_ids"].append(failed_outcome.sample_id)
        stats = patch.setdefault("memory_stats", {})
        stats["seen_count"] = stats.get("seen_count", 0) + 1
        stats["last_seen_round"] = self.current_repair_round
        stats["last_seen_sample_id"] = failed_outcome.sample_id

    def _update_patch_memory(self, patch: Dict[str, Any], failed_outcome: EvaluationOutcome) -> None:
        patch.setdefault("source_trace_ids", [])
        if failed_outcome.sample_id not in patch["source_trace_ids"]:
            patch["source_trace_ids"].append(failed_outcome.sample_id)
        stats = patch.setdefault("memory_stats", {})
        stats["seen_count"] = stats.get("seen_count", 0) + 1
        stats["duplicate_count"] = stats.get("duplicate_count", 0) + 1
        stats["last_seen_round"] = self.current_repair_round
        stats["last_seen_sample_id"] = failed_outcome.sample_id
        self.registry.save_patch(patch)

    def _should_retry_shadow_rejected(self, patch: Dict[str, Any], failed_outcome: EvaluationOutcome) -> bool:
        failed_ids = set(str(sample_id) for sample_id in patch.get("shadow_failed_sample_ids") or [])
        if failed_outcome.sample_id in failed_ids:
            return False
        fixer_type = patch.get("fixer", {}).get("type")
        generation_mode = patch.get("generation", {}).get("mode")
        if fixer_type != "llm_repair" and generation_mode != "failure_family_llm":
            return False
        telemetry = patch.get("telemetry", {})
        failure_count = int(telemetry.get("repair_failure_count") or 0)
        success_count = int(telemetry.get("repair_success_count") or 0)
        retry_limit = 3 if fixer_type == "llm_repair" else 2
        return success_count > 0 or failure_count < retry_limit

    def _record_patch_replay_result(
        self,
        patch: Dict[str, Any],
        failed_outcome: EvaluationOutcome,
        accepted: bool,
    ) -> None:
        key = "shadow_success_sample_ids" if accepted else "shadow_failed_sample_ids"
        patch.setdefault(key, [])
        if failed_outcome.sample_id not in patch[key]:
            patch[key].append(failed_outcome.sample_id)

    def _dedupe_patch_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped = []
        seen = set()
        for patch in candidates:
            patch_key = patch.get("patch_key") or patch.get("patch_id")
            if not patch_key or patch_key in seen:
                continue
            seen.add(patch_key)
            deduped.append(patch)
        return deduped

    def _write_json(self, path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fout:
            json.dump(record, fout, ensure_ascii=False, indent=2)


def load_workflow_class(workflow_dir: Path):
    workflow_dir = workflow_dir.resolve()
    validate_workflow_dir(workflow_dir)
    try:
        rel = workflow_dir.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"workflow_dir must be inside the project root for package imports: {workflow_dir}") from exc

    module_name = ".".join(rel.parts) + ".graph"
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    if not hasattr(module, "Workflow"):
        raise AttributeError(f"{workflow_dir / 'graph.py'} does not define class Workflow")
    return module.Workflow


def validate_workflow_dir(workflow_dir: Path) -> None:
    graph_path = workflow_dir / "graph.py"
    prompt_path = workflow_dir / "prompt.py"
    if not graph_path.exists():
        raise FileNotFoundError(f"Workflow graph not found: {graph_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Workflow prompt not found: {prompt_path}")

    graph_text = graph_path.read_text(encoding="utf-8")
    prompt_text = prompt_path.read_text(encoding="utf-8")
    referenced = set(referenced_prompt_names(graph_text))
    defined = defined_prompt_symbols(prompt_text)
    missing = sorted(referenced - defined)
    if missing:
        raise ValueError(
            "Workflow static validation failed: graph.py references prompt_custom symbols "
            f"that prompt.py does not define: {', '.join(missing)}"
        )


def defined_prompt_symbols(prompt_text: str) -> set:
    symbols = set()
    try:
        tree = ast.parse(prompt_text)
    except SyntaxError:
        return symbols
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
    return symbols


def parse_round_id(workflow_dir: Path) -> Optional[int]:
    match = re.search(r"round_(\d+)", workflow_dir.name)
    return int(match.group(1)) if match else None


def parse_indices(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv(raw: Optional[Any]) -> Optional[List[str]]:
    if not raw:
        return None
    if isinstance(raw, (list, tuple)):
        values = []
        for item in raw:
            values.extend(str(item).split(","))
    else:
        values = str(raw).split(",")
    return [part.strip() for part in values if part.strip()]


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _relative_cost_increase(before: float, after: float) -> float:
    increase = max(0.0, after - before)
    return increase / before if before > 0 else increase


def _count_by(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Repair one existing workflow with online Patch-as-Hypothesis")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--workflow_dir", required=True, help="Existing round_N workflow directory to repair")
    parser.add_argument("--exec_model_name", default=None, help="Execution model name from config/config2.yaml")
    parser.add_argument("--model_name", default=None, help="Alias for --exec_model_name")
    parser.add_argument("--trace_dir", default="traces_repair")
    parser.add_argument("--patch_registry_dir", default="results/workflow_repair/patch_registry")
    parser.add_argument("--output_dir", default="results/workflow_repair")
    parser.add_argument("--history_dir", default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--no_history", action="store_true", help="Disable structured run history and latest context files")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--trace_full_io", action="store_true")
    parser.add_argument("--trace_preview_chars", type=int, default=512)
    parser.add_argument("--success_threshold", type=float, default=None)
    parser.add_argument("--indices", default=None, help="Comma-separated dataset indices to run")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--repair_rounds", type=int, default=1, help="Number of iterative repair passes over the selected samples")
    parser.add_argument("--matched_successes", type=int, default=3)
    parser.add_argument("--regression_tolerance", type=float, default=0.05)
    parser.add_argument("--stop_when_no_new_patches", action="store_true")
    parser.add_argument("--no_llm_repair", action="store_true", help="Disable LLM-based local repair candidates")
    parser.add_argument(
        "--llm_generate_patches",
        action="store_true",
        help="Use the workflow LLM to synthesize reusable failure-family PatchSpec candidates",
    )
    parser.add_argument("--llm_patch_max_candidates", type=int, default=1)
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--no_online_patch", action="store_true")
    parser.add_argument("--no_consolidate", action="store_true")
    parser.add_argument(
        "--compile_workflow",
        action="store_true",
        help="Compile validated/active runtime patches into an explicit edited workflow directory",
    )
    parser.add_argument("--compiled_workflow_dir", default=None)
    parser.add_argument("--compile_min_credit", type=float, default=0.0)
    parser.add_argument("--compile_max_patches", type=int, default=8)
    parser.add_argument(
        "--eval_compiled_workflow",
        action="store_true",
        help="After compiling, evaluate the compiled workflow on the selected samples without online runtime patches",
    )
    parser.add_argument(
        "--compiled_eval_max_samples",
        type=int,
        default=None,
        help="Optional cap for compiled workflow evaluation samples; defaults to the same selected sample set",
    )
    parser.add_argument(
        "--compile_content_nodes",
        action="store_true",
        help="Insert guarded content-level repair nodes into the compiled workflow",
    )
    parser.add_argument(
        "--content_repair_families",
        nargs="+",
        default="numeric_content,entity_boundary,multi_span,duration",
        help="Content repair families to enable. Accepts comma-separated or space-separated values.",
    )
    parser.add_argument(
        "--content_repair_batches",
        nargs="+",
        default=None,
        help=(
            "Optional explicit content repair node batches to insert. "
            "Choices/aliases include: all, numeric, multispan, entity, duration."
        ),
    )
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


async def main_async():
    args = parse_args()
    exec_model_name = args.exec_model_name or args.model_name
    if not exec_model_name:
        raise ValueError("Please pass --exec_model_name or --model_name")
    content_repair_batches = parse_csv(args.content_repair_batches)
    compile_content_nodes = args.compile_content_nodes or bool(content_repair_batches)

    runner = WorkflowRepairRunner(
        dataset=args.dataset,
        workflow_dir=args.workflow_dir,
        exec_model_name=exec_model_name,
        trace_dir=args.trace_dir,
        patch_registry_dir=args.patch_registry_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        trace_full_io=args.trace_full_io,
        trace_preview_chars=args.trace_preview_chars,
        success_threshold=args.success_threshold,
        matched_successes=args.matched_successes,
        regression_tolerance=args.regression_tolerance,
        online_patch=not args.no_online_patch,
        enable_llm_repair=not args.no_llm_repair,
        enable_llm_patch_synthesis=args.llm_generate_patches,
        llm_patch_max_candidates=args.llm_patch_max_candidates,
        consolidate=not args.no_consolidate,
        compile_workflow=args.compile_workflow or compile_content_nodes,
        compiled_workflow_dir=args.compiled_workflow_dir,
        compile_min_credit=args.compile_min_credit,
        compile_max_patches=args.compile_max_patches,
        compile_content_nodes=compile_content_nodes,
        content_repair_families=parse_csv(args.content_repair_families),
        content_repair_batches=content_repair_batches,
        eval_compiled_workflow=args.eval_compiled_workflow,
        compiled_eval_max_samples=args.compiled_eval_max_samples,
        history_dir=args.history_dir,
        save_history=not args.no_history,
        is_test=args.test,
        stop_when_no_new_patches=args.stop_when_no_new_patches,
        show_progress=not args.no_progress,
    )
    summary = await runner.run(
        indices=parse_indices(args.indices),
        max_samples=args.max_samples,
        start_index=args.start_index,
        repair_rounds=args.repair_rounds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
