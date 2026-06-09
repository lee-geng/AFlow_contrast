import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.evaluator import Evaluator
from scripts.optimizer_utils.experience_guided import (
    CandidateEvaluationSummary,
    ExperienceGuidedEvaluationProtocol,
    ExperienceGuidedValidationSampler,
)


class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    async def evaluate_initial_round(self, optimizer, graph_path, directory, validation_n, data):
        optimizer.graph = optimizer.graph_utils.load_graph(optimizer.round, graph_path)
        evaluator = Evaluator(eval_path=directory)

        for _ in range(validation_n):
            specific_indices = self._validation_subset_indices(optimizer)
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
                specific_indices=specific_indices,
                max_concurrent_tasks=optimizer.eval_max_concurrent_tasks,
            )

            new_data = optimizer.data_utils.create_result_data(optimizer.round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(graph_path)
            optimizer.data_utils.save_results(result_path, data)

        return data

    async def evaluate_graph(
        self,
        optimizer,
        directory,
        validation_n,
        data,
        initial: bool = False,
        parent_round: Optional[int] = None,
    ):
        if not getattr(optimizer, "experience_guided_enabled", False):
            return await self._evaluate_graph_standard(optimizer, directory, validation_n, data, initial)
        if not (
            optimizer.experience_guided_config.adaptive_validation_enabled
            and optimizer.experience_guided_config.use_staged_validation
        ):
            return await self._evaluate_graph_experience_guided_full_only(
                optimizer,
                directory,
                validation_n,
                data,
                initial,
                parent_round,
            )
        return await self._evaluate_graph_experience_guided(optimizer, directory, validation_n, data, initial, parent_round)

    async def evaluate_graph_test(self, optimizer, directory, is_test=True):
        evaluator = Evaluator(eval_path=directory)
        return await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
            directory,
            is_test=is_test,
            max_concurrent_tasks=optimizer.eval_max_concurrent_tasks,
        )

    async def _evaluate_graph_standard(self, optimizer, directory, validation_n, data, initial=False):
        evaluator = Evaluator(eval_path=directory)
        sum_score = 0.0

        for _ in range(validation_n):
            specific_indices = self._validation_subset_indices(optimizer)
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
                specific_indices=specific_indices,
                max_concurrent_tasks=optimizer.eval_max_concurrent_tasks,
            )

            cur_round = optimizer.round + 1 if initial is False else optimizer.round

            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(f"{optimizer.root_path}/workflows")
            optimizer.data_utils.save_results(result_path, data)

            sum_score += score

        return sum_score / validation_n

    async def _evaluate_graph_experience_guided(
        self,
        optimizer,
        directory,
        validation_n,
        data,
        initial=False,
        parent_round: Optional[int] = None,
    ):
        cur_round = optimizer.round + 1 if initial is False else optimizer.round
        workflow_id = f"{optimizer.dataset}_round_{cur_round}"
        workflow_source = (Path(directory) / "graph.py").read_text(encoding="utf-8")
        sampler = ExperienceGuidedValidationSampler(
            optimizer.experience_guided_config,
            embedder=optimizer.experience_controller.embedder,
        )
        protocol = ExperienceGuidedEvaluationProtocol(optimizer.experience_guided_config)
        validation_records = await self._load_validation_records(optimizer)
        parent_workflow_id = f"{optimizer.dataset}_round_{parent_round}" if parent_round else ""
        parent_outcomes = optimizer.experience_controller.parent_outcomes(parent_workflow_id) if parent_workflow_id else {}
        memories = optimizer.experience_controller.query_memories(limit=20)
        plan = sampler.plan(validation_records, memories, parent_outcomes)

        historical_scores = [float(item.get("score", 0.0)) for item in data if isinstance(item, dict)]

        async def subset_runner(indices: List[int], stage_name: str, return_details: bool) -> Dict[str, Any]:
            stage_dir = self._stage_directory(directory, stage_name)
            details = await self._run_subset(optimizer, stage_dir, indices, return_details=return_details)
            optimizer.experience_controller.observe_evaluation(
                workflow_id=workflow_id,
                workflow_version=str(cur_round),
                workflow_source=workflow_source,
                details=details,
                stage=stage_name,
            )
            return {
                "score": details["score"],
                "avg_cost": details["avg_cost"],
                "total_cost": details["total_cost"],
                "details": details,
                "task_ids": [str(row.get("problem", {}).get("_task_id")) for row in details.get("results", [])],
            }

        summary: CandidateEvaluationSummary = await protocol.evaluate_candidate(
            plan=plan,
            subset_runner=subset_runner,
            parent_outcomes=parent_outcomes,
            historical_top_scores=historical_scores,
        )

        if "full" in summary.stage_results:
            self._copy_stage_outputs(self._stage_directory(directory, "full"), directory)
        elif "anchor" in summary.stage_results:
            self._copy_stage_outputs(self._stage_directory(directory, "anchor"), directory)
        else:
            self._copy_stage_outputs(self._stage_directory(directory, "targeted"), directory)

        total_cost = sum(stage.total_cost for stage in summary.stage_results.values())
        avg_cost = sum(stage.avg_cost for stage in summary.stage_results.values())
        new_data = optimizer.data_utils.create_result_data(cur_round, summary.comparable_score, avg_cost, total_cost)
        new_data.update(
            {
                "selection_score": summary.selection_score,
                "score_source": summary.score_source,
                "anchor_score": summary.anchor_score,
                "full_score": summary.full_score,
                "targeted_gain": summary.targeted_gain,
                "regression_loss": summary.regression_loss,
                "stages_run": summary.stages_run,
            }
        )
        data.append(new_data)

        result_path = optimizer.data_utils.get_results_file_path(f"{optimizer.root_path}/workflows")
        optimizer.data_utils.save_results(result_path, data)

        metrics = optimizer.experience_metrics
        metrics["number_of_candidate_workflows"] += 1
        metrics["number_of_targeted_validations"] += 1
        if "anchor" in summary.stage_results:
            metrics["number_of_anchor_validations"] += 1
        if "full" in summary.stage_results:
            metrics["number_of_full_validations"] += 1
        metrics["validation_task_executions_saved"] += max(0, len(validation_records) - len(plan.anchor_set))
        metrics["best_validation_score_so_far"] = max(metrics["best_validation_score_so_far"], summary.comparable_score)
        metrics["token_cost_estimate"] += total_cost
        metrics["patch_acceptance_rate"] = self._safe_div(
            metrics["number_of_anchor_validations"],
            metrics["number_of_candidate_workflows"],
        )
        metrics["regression_rate"] = self._safe_div(
            1 if summary.regression_loss > 0 else 0,
            1,
        )
        metrics["compressed_memory_count"] = len(optimizer.experience_controller.store.load_memories())
        traces = optimizer.experience_controller.store.load_traces()
        metrics["attribution_event_count"] = len(optimizer.experience_controller.store.load_events())
        metrics["memory_compression_ratio"] = self._safe_div(
            metrics["compressed_memory_count"],
            max(1, len(traces)),
        )
        optimizer.experience_controller.log_metrics(
            {
                "round": cur_round,
                "workflow_id": workflow_id,
                **metrics,
            }
        )
        return summary.comparable_score

    async def _evaluate_graph_experience_guided_full_only(
        self,
        optimizer,
        directory,
        validation_n,
        data,
        initial=False,
        parent_round: Optional[int] = None,
    ):
        cur_round = optimizer.round + 1 if initial is False else optimizer.round
        workflow_id = f"{optimizer.dataset}_round_{cur_round}"
        workflow_source = (Path(directory) / "graph.py").read_text(encoding="utf-8")
        validation_records = await self._load_validation_records(optimizer)
        full_indices = list(range(len(validation_records)))
        total_score = 0.0
        total_avg_cost = 0.0
        total_total_cost = 0.0

        for _ in range(validation_n):
            details = await self._run_subset(optimizer, directory, full_indices, return_details=True)
            optimizer.experience_controller.observe_evaluation(
                workflow_id=workflow_id,
                workflow_version=str(cur_round),
                workflow_source=workflow_source,
                details=details,
                stage="full",
            )
            score = details["score"]
            avg_cost = details["avg_cost"]
            total_cost = details["total_cost"]
            total_score += score
            total_avg_cost += avg_cost
            total_total_cost += total_cost

        avg_score = total_score / max(validation_n, 1)
        avg_cost = total_avg_cost / max(validation_n, 1)
        total_cost = total_total_cost / max(validation_n, 1)

        new_data = optimizer.data_utils.create_result_data(cur_round, avg_score, avg_cost, total_cost)
        new_data.update(
            {
                "selection_score": avg_score,
                "score_source": "full",
                "anchor_score": avg_score,
                "full_score": avg_score,
                "targeted_gain": 0.0,
                "regression_loss": 0.0,
                "stages_run": ["full"],
            }
        )
        data.append(new_data)

        result_path = optimizer.data_utils.get_results_file_path(f"{optimizer.root_path}/workflows")
        optimizer.data_utils.save_results(result_path, data)

        metrics = optimizer.experience_metrics
        metrics["number_of_candidate_workflows"] += 1
        metrics["number_of_full_validations"] += 1
        metrics["best_validation_score_so_far"] = max(metrics["best_validation_score_so_far"], avg_score)
        metrics["token_cost_estimate"] += total_cost
        metrics["compressed_memory_count"] = len(optimizer.experience_controller.store.load_memories())
        traces = optimizer.experience_controller.store.load_traces()
        metrics["attribution_event_count"] = len(optimizer.experience_controller.store.load_events())
        metrics["memory_compression_ratio"] = self._safe_div(
            metrics["compressed_memory_count"],
            max(1, len(traces)),
        )
        optimizer.experience_controller.log_metrics(
            {
                "round": cur_round,
                "workflow_id": workflow_id,
                **metrics,
            }
        )
        return avg_score

    async def _run_subset(self, optimizer, directory: str, indices: List[int], return_details: bool = True) -> Dict[str, Any]:
        evaluator = Evaluator(eval_path=directory)
        details = await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
            directory,
            is_test=False,
            specific_indices=indices,
            return_details=return_details,
            max_concurrent_tasks=optimizer.eval_max_concurrent_tasks,
        )
        return details

    async def _load_validation_records(self, optimizer) -> List[Dict[str, Any]]:
        evaluator = Evaluator(eval_path=self.root_path)
        dataset_path = evaluator._get_data_path(optimizer.dataset, False)
        benchmark = evaluator.dataset_configs[optimizer.dataset](name=optimizer.dataset, file_path=dataset_path, log_path=self.root_path)
        subset_indices = self._validation_subset_indices(optimizer)
        return await benchmark.load_data(specific_indices=subset_indices)

    def _validation_subset_indices(self, optimizer) -> Optional[List[int]]:
        subset_size = int(getattr(optimizer, "validation_subset_size", 0) or 0)
        if subset_size <= 0:
            return None
        return list(range(subset_size))

    def _stage_directory(self, directory: str, stage_name: str) -> str:
        stage_dir = Path(directory) / f"_{stage_name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return str(stage_dir)

    def _copy_stage_outputs(self, src: str, dest: str) -> None:
        src_path = Path(src)
        dest_path = Path(dest)
        for name in ["log.json"]:
            candidate = src_path / name
            if candidate.exists():
                shutil.copyfile(candidate, dest_path / name)
        csv_files = sorted(src_path.glob("*.csv"))
        for csv_file in csv_files:
            shutil.copyfile(csv_file, dest_path / csv_file.name)

    def _safe_div(self, x: float, y: float) -> float:
        if not y:
            return 0.0
        return x / y
