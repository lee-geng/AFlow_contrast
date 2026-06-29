import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from patch_evolution.repair_workflow import (  # noqa: E402
    DATASETS,
    WorkflowRepairRunner,
    parse_csv,
    parse_indices,
)
from patch_evolution.run_history import DEFAULT_HISTORY_DIR  # noqa: E402


SCHEMA_VERSION = 1


class ValidationWorkflowLoop:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset = args.dataset
        self.exec_model_name = args.exec_model_name or args.model_name
        if not self.exec_model_name:
            raise ValueError("Please pass --exec_model_name or --model_name")

        self.initial_workflow_dir = Path(args.workflow_dir)
        self.current_workflow_dir = self.initial_workflow_dir
        self.output_dir = Path(args.output_dir)
        self.run_id = args.run_id or f"validation_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.loop_root = self.output_dir / self.dataset / self.run_id
        self.loop_summary_path = self.loop_root / "loop_summary.json"
        self.content_repair_batches = parse_csv(args.content_repair_batches)
        self.content_repair_families = parse_csv(args.content_repair_families)
        self.indices = parse_indices(args.indices)
        self.loop_root.mkdir(parents=True, exist_ok=True)

    async def run(self) -> Dict[str, Any]:
        iterations: List[Dict[str, Any]] = []
        best_score: Optional[float] = None
        best_failure_count: Optional[int] = None
        best_workflow_dir = str(self.current_workflow_dir)
        no_improvement_rounds = 0
        stop_reason = "max_iterations_reached"

        for iteration in range(1, max(1, self.args.validation_iterations) + 1):
            iteration_run_id = f"{self.run_id}_iter{iteration:02d}"
            runner = WorkflowRepairRunner(
                dataset=self.dataset,
                workflow_dir=str(self.current_workflow_dir),
                exec_model_name=self.exec_model_name,
                trace_dir=self.args.trace_dir,
                patch_registry_dir=self.args.patch_registry_dir,
                output_dir=self.args.output_dir,
                run_id=iteration_run_id,
                trace_full_io=self.args.trace_full_io,
                trace_preview_chars=self.args.trace_preview_chars,
                success_threshold=self.args.success_threshold,
                matched_successes=self.args.matched_successes,
                regression_tolerance=self.args.regression_tolerance,
                online_patch=not self.args.no_online_patch,
                enable_llm_repair=not self.args.no_llm_repair,
                enable_llm_patch_synthesis=self.args.llm_generate_patches,
                llm_patch_max_candidates=self.args.llm_patch_max_candidates,
                consolidate=not self.args.no_consolidate,
                compile_workflow=True,
                compiled_workflow_dir=self.args.compiled_workflow_dir,
                compile_min_credit=self.args.compile_min_credit,
                compile_max_patches=self.args.compile_max_patches,
                compile_content_nodes=self.args.compile_content_nodes or bool(self.content_repair_batches),
                content_repair_families=self.content_repair_families,
                content_repair_batches=self.content_repair_batches,
                eval_compiled_workflow=not self.args.no_eval_compiled_workflow,
                compiled_eval_max_samples=self.args.compiled_eval_max_samples,
                history_dir=self.args.history_dir,
                save_history=not self.args.no_history,
                is_test=self.args.test,
                stop_when_no_new_patches=self.args.stop_when_no_new_patches,
                show_progress=not self.args.no_progress,
            )
            summary = await runner.run(
                indices=self.indices,
                max_samples=self.args.max_samples,
                start_index=self.args.start_index,
                repair_rounds=self.args.repair_rounds_per_iteration,
            )

            record = self._iteration_record(iteration, iteration_run_id, summary)
            candidate_score = record.get("candidate_score")
            candidate_failure_count = record.get("candidate_failure_count")
            if candidate_score is not None and (
                best_score is None or candidate_score > best_score + self.args.min_improvement
            ):
                best_score = candidate_score
                best_failure_count = candidate_failure_count
                best_workflow_dir = record.get("candidate_workflow_dir") or best_workflow_dir
                no_improvement_rounds = 0
                record["improved_best"] = True
            else:
                no_improvement_rounds += 1
                record["improved_best"] = False

            if record["adopted"]:
                self.current_workflow_dir = Path(record["candidate_workflow_dir"])

            iterations.append(record)
            loop_summary = self._loop_summary(
                iterations=iterations,
                status="running",
                stop_reason=None,
                best_score=best_score,
                best_failure_count=best_failure_count,
                best_workflow_dir=best_workflow_dir,
            )
            self._write_json(self.loop_summary_path, loop_summary)

            if candidate_failure_count == 0:
                stop_reason = "validation_failures_repaired"
                break
            if self.args.stop_when_no_adoption and not record["adopted"]:
                stop_reason = "candidate_workflow_not_adopted"
                break
            if self.args.stop_on_no_improvement and no_improvement_rounds >= self.args.patience:
                stop_reason = "no_validation_improvement"
                break

        final_summary = self._loop_summary(
            iterations=iterations,
            status="complete",
            stop_reason=stop_reason,
            best_score=best_score,
            best_failure_count=best_failure_count,
            best_workflow_dir=best_workflow_dir,
        )
        self._write_json(self.loop_summary_path, final_summary)
        return final_summary

    def _iteration_record(self, iteration: int, iteration_run_id: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        compilation = summary.get("workflow_compilation") or {}
        compiled_eval = summary.get("compiled_workflow_evaluation") or {}
        online_score = _as_float(summary.get("average_score"))
        baseline_score = _as_float(summary.get("baseline_average_score"))
        compiled_score = _as_float(compiled_eval.get("average_score"))
        candidate_score = compiled_score if compiled_score is not None else online_score
        candidate_failure_count = _as_int(
            compiled_eval.get("failure_count")
            if compiled_eval
            else summary.get("failure_count")
        )
        candidate_workflow_dir = compilation.get("compiled_workflow_dir")
        compiled_path_exists = bool(candidate_workflow_dir and Path(candidate_workflow_dir).exists())
        eval_available = bool(compiled_eval and compiled_eval.get("status") != "skipped")
        adopted = False
        adoption_reason = "not_evaluated"
        if not compiled_path_exists:
            adoption_reason = "compiled_workflow_missing"
        elif self.args.no_eval_compiled_workflow and self.args.adopt_without_compiled_eval:
            adopted = True
            adoption_reason = "adopt_without_compiled_eval"
        elif eval_available and candidate_score is not None and baseline_score is not None:
            if candidate_score + self.args.adoption_tolerance >= baseline_score:
                adopted = True
                adoption_reason = "non_regression"
            else:
                adoption_reason = "compiled_validation_regressed"
        elif eval_available and candidate_score is not None:
            adopted = True
            adoption_reason = "compiled_eval_available"

        return {
            "iteration": iteration,
            "run_id": iteration_run_id,
            "source_workflow_dir": str(self.current_workflow_dir),
            "summary_path": str(self.output_dir / self.dataset / iteration_run_id / "summary.json"),
            "trace_root": summary.get("trace_root"),
            "baseline_score": baseline_score,
            "online_score": online_score,
            "compiled_score": compiled_score,
            "candidate_score": candidate_score,
            "candidate_failure_count": candidate_failure_count,
            "success_count": compiled_eval.get("success_count") if compiled_eval else summary.get("success_count"),
            "sample_count": compiled_eval.get("sample_count") if compiled_eval else summary.get("sample_count"),
            "accepted_patch_count": summary.get("accepted_patch_count", 0),
            "compiled_patch_count": compilation.get("compiled_patch_count", 0),
            "inserted_nodes": compilation.get("inserted_nodes") or [],
            "candidate_workflow_dir": candidate_workflow_dir,
            "adopted": adopted,
            "adoption_reason": adoption_reason,
            "failure_samples": (compiled_eval.get("failure_samples") if compiled_eval else summary.get("failure_samples")) or [],
        }

    def _loop_summary(
        self,
        iterations: List[Dict[str, Any]],
        status: str,
        stop_reason: Optional[str],
        best_score: Optional[float],
        best_failure_count: Optional[int],
        best_workflow_dir: str,
    ) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "stop_reason": stop_reason,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "exec_model_name": self.exec_model_name,
            "initial_workflow_dir": str(self.initial_workflow_dir),
            "final_workflow_dir": str(self.current_workflow_dir),
            "best_workflow_dir": best_workflow_dir,
            "best_score": best_score,
            "best_failure_count": best_failure_count,
            "iterations_completed": len(iterations),
            "validation_iterations_requested": self.args.validation_iterations,
            "max_samples": self.args.max_samples,
            "indices": self.indices,
            "loop_summary_path": str(self.loop_summary_path),
            "iterations": iterations,
        }

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iteratively repair, compile, and validate a workflow.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--workflow_dir", required=True, help="Initial workflow directory for the validation loop")
    parser.add_argument("--exec_model_name", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--trace_dir", default="traces_validation_loop")
    parser.add_argument("--patch_registry_dir", default="results/workflow_validation_loop/patch_registry")
    parser.add_argument("--output_dir", default="results/workflow_validation_loop")
    parser.add_argument("--compiled_workflow_dir", default="compiled_workflows_validation_loop")
    parser.add_argument("--history_dir", default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--validation_iterations", type=int, default=3)
    parser.add_argument("--repair_rounds_per_iteration", type=int, default=1)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--trace_full_io", action="store_true")
    parser.add_argument("--trace_preview_chars", type=int, default=512)
    parser.add_argument("--success_threshold", type=float, default=None)
    parser.add_argument("--matched_successes", type=int, default=3)
    parser.add_argument("--regression_tolerance", type=float, default=0.05)
    parser.add_argument("--no_online_patch", action="store_true")
    parser.add_argument("--no_llm_repair", action="store_true")
    parser.add_argument("--llm_generate_patches", action="store_true")
    parser.add_argument("--llm_patch_max_candidates", type=int, default=1)
    parser.add_argument("--no_consolidate", action="store_true")
    parser.add_argument("--compile_min_credit", type=float, default=0.0)
    parser.add_argument("--compile_max_patches", type=int, default=8)
    parser.add_argument("--compile_content_nodes", action="store_true")
    parser.add_argument("--content_repair_families", nargs="+", default="numeric_content,entity_boundary,multi_span,duration")
    parser.add_argument("--content_repair_batches", nargs="+", default="all")
    parser.add_argument("--no_eval_compiled_workflow", action="store_true")
    parser.add_argument("--compiled_eval_max_samples", type=int, default=None)
    parser.add_argument("--adoption_tolerance", type=float, default=0.0)
    parser.add_argument("--adopt_without_compiled_eval", action="store_true")
    parser.add_argument("--min_improvement", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--stop_on_no_improvement", action="store_true")
    parser.add_argument("--stop_when_no_adoption", action="store_true")
    parser.add_argument("--stop_when_no_new_patches", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--no_history", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


async def main_async() -> None:
    loop = ValidationWorkflowLoop(parse_args())
    summary = await loop.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
