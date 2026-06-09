# -*- coding: utf-8 -*-
# @Date    : 8/12/2024 22:00 PM
# @Author  : issac
# @Desc    : optimizer for graph (updated with AsyncLLM integration)

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType
from scripts.formatter import FormatError, XmlFormatter
from scripts.logs import logger
from scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from scripts.optimizer_utils.experience_guided import (
    ExperienceGuidedConfig,
    ExperienceGuidedController,
    MemoryGuidance,
    OptimizationStateQuery,
)
from scripts.optimizer_utils.experience_utils import ExperienceUtils
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.optimizer_utils.retry_utils import RETRYABLE_ERROR_KEYWORDS, is_retryable_runtime_error
from scripts.optimizer_utils.revision_prior import RevisionPriorController, append_prior_log
from scripts.optimizer_utils.trajectory_memory import TrajectoryMemoryController
from scripts.optimizer_utils.workflow_guard import GuardMode, WorkflowGuard, WorkflowGuardResult
from scripts.utils.common import read_json_file

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]
PriorMode = Literal["generate_then_search", "rank_then_search"]


class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        check_convergence: bool = False,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 5,
        use_revision_prior: bool = False,
        revision_prior_path: str = "",
        revision_prior_mode: PriorMode = "generate_then_search",
        revision_rank_candidates: int = 3,
        use_trajectory_memory: bool = False,
        memory_min_support: int = 2,
        memory_max_patterns: int = 80,
        workflow_guard_mode: GuardMode = "repair_then_continue",
        experience_memory_enabled: bool = False,
        experience_store_path: str = "",
        experience_min_support: int = 3,
        experience_min_confidence: float = 0.6,
        experience_compression_interval: int = 10,
        attribution_enabled: bool = True,
        attribution_mode: str = "heuristic",
        adaptive_validation_enabled: bool = False,
        use_staged_validation: bool = True,
        targeted_failure_size: int = 8,
        success_guard_size: int = 8,
        anchor_size: int = 16,
        full_validation_top_k: int = 3,
        adaptive_beta: float = 0.5,
        adaptive_gamma: float = 0.5,
        adaptive_lambda_cost: float = 0.1,
        validation_subset_size: int = 0,
        task_embedding_enabled: bool = True,
        task_embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        task_embedding_base_url: str = "http://localhost:8090/v1",
        task_embedding_api_key: str = "EMPTY",
        task_embedding_dim: int = 64,
        task_similarity_threshold: float = 0.55,
        enable_prompt_budget_diagnostics: bool = True,
        use_failure_cards: bool = True,
        use_optimization_state_memory_query: bool = True,
        use_memory_guided_parent_selection: bool = False,
        use_memory_guided_patch_scope: bool = True,
        use_memory_as_experience_replacement: bool = True,
        use_graph_prompt_summarization: bool = True,
        max_candidate_attempts_per_round: int = 3,
        eval_max_concurrent_tasks: int = 10,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = create_llm_instance(self.optimize_llm_config)
        self.execute_llm_config = exec_llm_config

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds
        self.validation_subset_size = max(0, int(validation_subset_size))
        self.max_candidate_attempts_per_round = max(1, int(max_candidate_attempts_per_round))
        self.eval_max_concurrent_tasks = max(1, int(eval_max_concurrent_tasks))

        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)

        self.use_revision_prior = use_revision_prior and bool(revision_prior_path)
        self.revision_prior_mode = revision_prior_mode
        self.revision_rank_candidates = max(1, revision_rank_candidates)
        self.revision_prior: Optional[RevisionPriorController] = None
        self.use_trajectory_memory = use_trajectory_memory
        self.trajectory_memory: Optional[TrajectoryMemoryController] = None
        self.workflow_guard_mode: GuardMode = workflow_guard_mode
        self.workflow_guard = WorkflowGuard(mode=workflow_guard_mode)
        default_store_path = experience_store_path or f"{self.root_path}/workflows/experience_store"
        self.experience_guided_config = ExperienceGuidedConfig(
            enabled=experience_memory_enabled,
            store_path=default_store_path,
            min_support=experience_min_support,
            min_confidence=experience_min_confidence,
            compression_interval=experience_compression_interval,
            attribution_enabled=attribution_enabled,
            attribution_mode=attribution_mode,
            adaptive_validation_enabled=adaptive_validation_enabled,
            use_staged_validation=use_staged_validation,
            targeted_failure_size=targeted_failure_size,
            success_guard_size=success_guard_size,
            anchor_size=anchor_size,
            full_validation_top_k=full_validation_top_k,
            beta=adaptive_beta,
            gamma=adaptive_gamma,
            lambda_cost=adaptive_lambda_cost,
            task_embedding_enabled=task_embedding_enabled,
            task_embedding_model_name=task_embedding_model_name,
            task_embedding_base_url=task_embedding_base_url,
            task_embedding_api_key=task_embedding_api_key,
            task_embedding_dim=task_embedding_dim,
            task_similarity_threshold=task_similarity_threshold,
            enable_prompt_budget_diagnostics=enable_prompt_budget_diagnostics,
            use_failure_cards=use_failure_cards,
            use_optimization_state_memory_query=use_optimization_state_memory_query,
            use_memory_guided_parent_selection=use_memory_guided_parent_selection,
            use_memory_guided_patch_scope=use_memory_guided_patch_scope,
            use_memory_as_experience_replacement=use_memory_as_experience_replacement,
            use_graph_prompt_summarization=use_graph_prompt_summarization,
        )
        self.experience_controller = ExperienceGuidedController(dataset=self.dataset, config=self.experience_guided_config)
        self.experience_guided_enabled = self.experience_guided_config.enabled
        self.experience_metrics: Dict[str, float] = {
            "number_of_candidate_workflows": 0,
            "number_of_full_validations": 0,
            "number_of_anchor_validations": 0,
            "number_of_targeted_validations": 0,
            "validation_task_executions_saved": 0,
            "attribution_event_count": 0,
            "compressed_memory_count": 0,
            "memory_compression_ratio": 0.0,
            "patch_acceptance_rate": 0.0,
            "regression_rate": 0.0,
            "best_validation_score_so_far": 0.0,
            "token_cost_estimate": 0.0,
            "cost_to_target_score": 0.0,
        }
        self._last_generation_failure_summary: str = ""

        if self.use_revision_prior:
            try:
                self.revision_prior = RevisionPriorController(
                    model_dir=revision_prior_path,
                    mode=revision_prior_mode,
                    rank_candidates=self.revision_rank_candidates,
                )
                logger.info(f"Revision prior enabled: mode={revision_prior_mode}, path={revision_prior_path}")
            except Exception as exc:
                logger.error(f"Failed to load revision prior ({exc}), fallback to vanilla search.")
                self.use_revision_prior = False
                self.revision_prior = None

        if self.use_trajectory_memory:
            self.trajectory_memory = TrajectoryMemoryController(
                min_support=max(1, memory_min_support),
                max_patterns=max(10, memory_max_patterns),
            )

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            test_n = 1
            for _ in range(test_n):
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.test())
                finally:
                    self._close_event_loop(loop)
            return None

        for _ in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 5

            while retry_count < max_retries:
                try:
                    self._last_generation_failure_summary = ""
                    score = loop.run_until_complete(self._optimize_graph())
                    break
                except Exception as e:
                    retry_count += 1
                    retryable = self._is_retryable_runtime_error(e)
                    if retryable and retry_count < max_retries:
                        logger.info(
                            f"Retryable error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})"
                        )
                    else:
                        if retryable:
                            logger.info("Max retries reached. Moving to next round.")
                        else:
                            extra = ""
                            if self._last_generation_failure_summary and self._last_generation_failure_summary not in str(e):
                                extra = f" Root causes: {self._last_generation_failure_summary}"
                            logger.info(f"Non-retryable error occurred: {e}.{extra} Moving to next round.")
                        score = None
                        break

                    wait_time = min(30, 2 ** retry_count)
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    self._close_event_loop(loop)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self._close_event_loop(loop)
            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)
            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
                )
                self.convergence_utils.print_results()
                break

            time.sleep(5)

    @staticmethod
    def _close_event_loop(loop) -> None:
        try:
            if loop is None:
                return
            try:
                pending = asyncio.all_tasks(loop)
            except RuntimeError:
                pending = set()
            for task in pending:
                task.cancel()
            if pending:
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
            asyncio.set_event_loop(None)
            loop.close()
        except Exception:
            pass

    async def _call_optimize_llm(self, graph_optimize_prompt: str) -> Optional[Dict[str, str]]:
        call_retries = 4
        for attempt in range(1, call_retries + 1):
            try:
                graph_formatter = XmlFormatter.from_model(GraphOptimize)
                response = await self.optimize_llm.call_with_format(graph_optimize_prompt, graph_formatter)
                if response and response.get("graph"):
                    response["graph"] = self.workflow_guard.sanitize_graph_text(response.get("graph", ""))
                logger.info("Graph optimization response received successfully")
                return response
            except FormatError as e:
                logger.error(f"Format error in graph optimization: {str(e)}")
                raw_response = await self.optimize_llm(graph_optimize_prompt)
                response = self._extract_fields_from_response(raw_response)
                if not response:
                    logger.error("Failed to extract fields from raw response")
                    return None
                return response
            except Exception as e:
                if not self._is_retryable_runtime_error(e) or attempt == call_retries:
                    raise
                sleep_s = min(20, 2 ** attempt)
                logger.info(
                    f"Retryable optimize-LLM call error: {e}. Backing off {sleep_s}s "
                    f"(Attempt {attempt}/{call_retries})"
                )
                await asyncio.sleep(sleep_s)
        return None

    async def _optimize_graph(self):
        validation_n = self.validation_rounds
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)
        if self.use_trajectory_memory and self.trajectory_memory is not None:
            try:
                self.trajectory_memory.refresh(workflows_dir=graph_path, task_context=self.dataset)
            except Exception as exc:
                logger.warning(f"Trajectory memory refresh skipped: {exc}")

        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=True)

        candidate_attempt = 0
        recent_generation_failures: List[str] = []
        blocked_parent_rounds: List[int] = []
        while True:
            candidate_attempt += 1
            if candidate_attempt > self.max_candidate_attempts_per_round:
                failure_summary = self._summarize_recent_generation_failures(recent_generation_failures)
                self._last_generation_failure_summary = failure_summary
                message = (
                    f"Exceeded max candidate attempts for round {self.round + 1}: "
                    f"{self.max_candidate_attempts_per_round}"
                )
                if failure_summary:
                    message += f". Root causes: {failure_summary}"
                raise RuntimeError(message)
            directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)

            processed_experience = self.experience_utils.load_experience()
            top_rounds = self._select_parent_round_candidates(
                graph_path=graph_path,
                processed_experience=processed_experience,
                recent_generation_failures=recent_generation_failures,
                blocked_parent_rounds=blocked_parent_rounds,
            )
            sample = self.data_utils.select_round(top_rounds)

            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
            graph = self.graph_utils.extract_solve_graph(graph_load)
            if self.experience_guided_config.use_graph_prompt_summarization:
                workflow_summary, prompt_summary = self.graph_utils.summarize_workflow(graph_load, prompt)
            else:
                workflow_summary, prompt_summary = graph[0], prompt

            operator_description = self.graph_utils.load_operators_description(self.operators)
            if self.experience_guided_config.use_failure_cards:
                failure_summaries = self.data_utils.summarize_failures(sample["round"])
                failure_cards = self._load_failure_cards(sample["round"])
            else:
                failure_summaries = self.data_utils.load_log(sample["round"])
                failure_cards = []
            experience = self.experience_utils.format_experience(processed_experience, sample["round"])
            failed_mods, success_mods = self._experience_modifications(processed_experience, sample["round"])
            optimization_state: Optional[OptimizationStateQuery] = None
            memory_guidance = MemoryGuidance()
            memory_guidance_text = ""
            if self.experience_guided_enabled and self.experience_guided_config.use_optimization_state_memory_query:
                optimization_state = self.experience_controller.build_optimization_state_query(
                    parent_round=int(sample["round"]),
                    parent_workflow_id=f"{self.dataset}_round_{sample['round']}",
                    score=float(sample["score"]),
                    failure_cards=failure_cards,
                    graph_summary=workflow_summary,
                    prompt_summary=prompt_summary,
                    recent_failed_modifications=failed_mods,
                    recent_successful_modifications=success_mods,
                )
                memory_guidance = self.experience_controller.build_memory_guidance(optimization_state)
                if self.experience_guided_config.use_memory_guided_patch_scope:
                    memory_guidance_text = self.experience_controller.build_memory_guidance_text(memory_guidance)
                else:
                    memory_guidance_text = ""

            if self.use_trajectory_memory and self.trajectory_memory is not None:
                memory_suggestion = self.trajectory_memory.suggest_for_workflow(
                    task_context=self.dataset,
                    workflow_graph=graph[0],
                    execution_feedback={
                        "failure_summary": failure_summaries[:500],
                        "trace_summary": failure_summaries[:300],
                        "score_mean": float(sample["score"]),
                    },
                )
                trajectory_memory_text = self._render_trajectory_memory_hint(memory_suggestion)
            else:
                trajectory_memory_text = ""

            experience_context, experience_mode = self._build_experience_context(
                sample_round=int(sample["round"]),
                old_experience=experience,
                memory_guidance_text=memory_guidance_text,
                memory_guidance=memory_guidance,
            )
            graph_optimize_prompt = self._build_optimize_prompt(
                sample=sample,
                workflow_summary=workflow_summary,
                prompt_summary=prompt_summary,
                operator_description=operator_description,
                failure_summaries=failure_summaries,
                trajectory_memory_text=trajectory_memory_text,
                memory_guidance_text=memory_guidance_text,
                experience_context=experience_context,
                memory_guidance=memory_guidance,
                guard_feedback_text=self._build_guard_feedback_text(recent_generation_failures),
            )
            logger.info(
                f"[ExperienceContextMode] mode={experience_mode} "
                f"old_experience_chars={len(experience)} memory_guidance_chars={len(memory_guidance_text)}"
            )

            prior_payload: Dict[str, object] = {
                "prior_enabled": bool(self.use_revision_prior),
                "prior_mode": self.revision_prior_mode,
                "parent_round": sample["round"],
                "parent_score": sample["score"],
                "prior_proposals": [],
                "candidate_scores": [],
            }

            prior_context: Optional[Dict[str, object]] = None
            if self.use_revision_prior and self.revision_prior is not None:
                prior_context = self.revision_prior.build_context(
                    dataset=self.dataset,
                    parent_round=sample["round"],
                    parent_workflow=graph[0],
                    parent_score=float(sample["score"]),
                    log_data=failure_summaries,
                    task_summary=(optimization_state.task_query_text if optimization_state else self.dataset),
                )

                if self.revision_prior_mode == "generate_then_search":
                    prior_edit = self.revision_prior.generate_prior_edit(prior_context)
                    prior_payload["prior_proposals"].append(prior_edit)
                    graph_optimize_prompt = self.revision_prior.augment_prompt(graph_optimize_prompt, prior_edit)

            response: Optional[Dict[str, str]] = None
            if (
                self.use_revision_prior
                and self.revision_prior is not None
                and self.revision_prior_mode == "rank_then_search"
                and prior_context is not None
            ):
                best_score = float("-inf")
                for _ in range(self.revision_rank_candidates):
                    candidate = await self._call_optimize_llm(graph_optimize_prompt)
                    if not candidate:
                        continue
                    score = self.revision_prior.score_edit(prior_context, candidate.get("modification", ""))
                    prior_payload["candidate_scores"].append(
                        {"modification": candidate.get("modification", ""), "prior_score": score}
                    )
                    if score > best_score:
                        best_score = score
                        response = candidate
            else:
                response = await self._call_optimize_llm(graph_optimize_prompt)

            if not response:
                continue
            candidate_snapshot_dir = self._persist_candidate_snapshot(
                directory=directory,
                candidate_attempt=candidate_attempt,
                sample_round=int(sample["round"]),
                response=response,
            )

            guard_result = self.workflow_guard.ensure(response.get("graph", ""))
            if guard_result.repaired:
                response["graph"] = guard_result.graph
                logger.info(f"Workflow guard auto-repaired graph: {', '.join(guard_result.issues)}")
                self._update_candidate_snapshot_status(
                    candidate_snapshot_dir,
                    status="guard_repaired",
                    guard_result=guard_result,
                )
            if not guard_result.valid:
                self._update_candidate_snapshot_status(
                    candidate_snapshot_dir,
                    status="rejected",
                    rejection_kind="invalid_graph",
                    rejection_reasons=guard_result.issues,
                    guard_result=guard_result,
                )
                logger.warning(f"Workflow guard blocked invalid graph: {', '.join(guard_result.issues)}")
                logger.info(f"Rejected candidate saved to {candidate_snapshot_dir}")
                recent_generation_failures.append(
                    f"Attempt {candidate_attempt}: invalid graph rejected by guard because {', '.join(guard_result.issues)}."
                )
                recent_generation_failures = recent_generation_failures[-3:]
                if self.workflow_guard_mode == "fail_fast":
                    raise RuntimeError(f"Workflow guard fail-fast: {guard_result.issues}")
                logger.info(
                    f"Rejected candidate attempt {candidate_attempt}/{self.max_candidate_attempts_per_round} "
                    f"for round {self.round + 1}"
                )
                continue

            degeneration_reasons = self._detect_degenerate_candidate(
                response=response,
                parent_graph=graph_load,
                parent_prompt=prompt,
            )
            if degeneration_reasons:
                self._update_candidate_snapshot_status(
                    candidate_snapshot_dir,
                    status="rejected",
                    rejection_kind="degenerate_candidate",
                    rejection_reasons=degeneration_reasons,
                    guard_result=guard_result,
                )
                logger.info(f"Rejected candidate saved to {candidate_snapshot_dir}")
                recent_generation_failures.append(
                    f"Attempt {candidate_attempt}: candidate rejected as degenerate because {', '.join(degeneration_reasons)}."
                )
                recent_generation_failures = recent_generation_failures[-3:]
                logger.warning(f"Rejected degenerate candidate: {', '.join(degeneration_reasons)}")
                continue

            check = self.experience_utils.check_modification(
                processed_experience,
                response["modification"],
                sample["round"],
            )

            if check:
                self._update_candidate_snapshot_status(
                    candidate_snapshot_dir,
                    status="accepted",
                    guard_result=guard_result,
                )
                break
            self._update_candidate_snapshot_status(
                candidate_snapshot_dir,
                status="rejected",
                rejection_kind="duplicate_modification",
                rejection_reasons=["repeats_known_history"],
                guard_result=guard_result,
            )
            recent_generation_failures.append(
                f"Attempt {candidate_attempt}: modification rejected because it repeats known history: "
                f"{response.get('modification', '')[:240]}"
            )
            recent_generation_failures = recent_generation_failures[-3:]
            if self.type == "code" and int(sample["round"]) not in blocked_parent_rounds:
                blocked_parent_rounds.append(int(sample["round"]))
                logger.info(
                    f"[CodeParentDiversification] blocking parent round {sample['round']} "
                    f"after duplicate modification to diversify subsequent attempts"
                )
            logger.info(f"Rejected candidate saved to {candidate_snapshot_dir}")
            logger.info(
                f"Rejected duplicate/known modification attempt {candidate_attempt}/"
                f"{self.max_candidate_attempts_per_round} for round {self.round + 1}"
            )

        self.graph_utils.write_graph_files(directory, response, self.round + 1, self.dataset)

        experience = self.experience_utils.create_experience_data(sample, response["modification"])
        self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)

        logger.info(directory)
        avg_score = await self.evaluation_utils.evaluate_graph(
            self,
            directory,
            validation_n,
            data,
            initial=False,
            parent_round=sample["round"],
        )
        self.experience_utils.update_experience(directory, experience, avg_score)

        prior_payload.update(
            {
                "selected_modification": response.get("modification", ""),
                "selected_graph_written": True,
                "child_round": self.round + 1,
                "child_score": avg_score,
                "delta_score": float(avg_score - sample["score"]),
                "executed": True,
            }
        )
        append_prior_log(directory, prior_payload)
        if self.use_trajectory_memory and self.trajectory_memory is not None:
            try:
                self.trajectory_memory.refresh(workflows_dir=graph_path, task_context=self.dataset)
            except Exception as exc:
                logger.warning(f"Trajectory memory post-update refresh skipped: {exc}")
        if self.experience_guided_enabled:
            self.experience_controller.compress()

        return avg_score

    def _augment_prompt_with_memory(self, prompt: str, suggestion: Dict[str, object]) -> str:
        selected = suggestion.get("selected_edit", {}) if isinstance(suggestion, dict) else {}
        if not isinstance(selected, dict):
            return prompt
        guardrails = selected.get("guardrails", [])
        guardrails_text = ", ".join([str(x) for x in guardrails[:3]]) if isinstance(guardrails, list) else ""
        block = (
            "\n\n[Failure-Aware Memory Hint]\n"
            f"- localized_failure_type: {suggestion.get('localized_failure_type', 'unknown')}\n"
            f"- localized_failure_node: {suggestion.get('localized_failure_node', 'b0')}\n"
            f"- minimal_edit_primitive: {selected.get('candidate_edit', 'unknown')}\n"
            f"- expected_delta_score: {selected.get('expected_delta_score', 0.0)}\n"
            f"- risk_of_regression: {selected.get('risk_of_regression', 0.0)}\n"
            f"- guardrails: {guardrails_text}\n"
            "Apply one minimal local edit unless strong evidence requires otherwise.\n"
        )
        return prompt + block

    def _render_trajectory_memory_hint(self, suggestion: Dict[str, object]) -> str:
        selected = suggestion.get("selected_edit", {}) if isinstance(suggestion, dict) else {}
        if not isinstance(selected, dict):
            return ""
        guardrails = selected.get("guardrails", [])
        guardrails_text = ", ".join([str(x) for x in guardrails[:3]]) if isinstance(guardrails, list) else ""
        return (
            "[Trajectory Memory]\n"
            f"- localized_failure_type: {suggestion.get('localized_failure_type', 'unknown')}\n"
            f"- localized_failure_node: {suggestion.get('localized_failure_node', 'b0')}\n"
            f"- minimal_edit_primitive: {selected.get('candidate_edit', 'unknown')}\n"
            f"- expected_delta_score: {selected.get('expected_delta_score', 0.0)}\n"
            f"- risk_of_regression: {selected.get('risk_of_regression', 0.0)}\n"
            f"- guardrails: {guardrails_text}\n"
            "Apply one minimal local edit unless strong evidence requires otherwise.\n"
        )

    def _build_optimize_prompt(
        self,
        sample: Dict[str, Any],
        workflow_summary: str,
        prompt_summary: str,
        operator_description: str,
        failure_summaries: str,
        trajectory_memory_text: str,
        memory_guidance_text: str,
        experience_context: str,
        memory_guidance: MemoryGuidance,
        guard_feedback_text: str,
    ) -> str:
        objective = (
            f"Parent round: {sample['round']}\n"
            f"Current score: {sample['score']}\n"
            "Goal: improve the workflow with a small targeted patch while preserving validated successful subworkflows."
        )
        workflow_context = (
            f"{workflow_summary}\n"
            f"{prompt_summary}\n"
            f"[Operator Description]\n{operator_description}\n"
            f"[Experience Context]\n{experience_context}"
        )
        failure_evidence = failure_summaries
        memory_block = "\n".join([part for part in [trajectory_memory_text.strip(), memory_guidance_text.strip()] if part]).strip()
        constraints = (
            "- Do not rewrite unrelated parts of the workflow.\n"
            "- Preserve validated successful subworkflows unless directly contradicted by evidence.\n"
            "- Do not repeat rejected modifications.\n"
            "- Prefer small targeted patches over broad prompt expansion.\n"
            "- If the previous candidate failed syntax or guard checks, fix those exact issues before making any new change.\n"
            "- Output pure valid Python code inside <graph> with a complete Workflow class and async __call__.\n"
            "- If changing graph.py, explain why graph-level modification is needed.\n"
            "- If changing prompt.py, edit only the relevant prompt section when possible.\n"
        )
        if memory_guidance.preserve_nodes:
            constraints += f"- Preserve nodes: {', '.join(memory_guidance.preserve_nodes[:4])}.\n"
        if memory_guidance.focus_nodes:
            constraints += f"- Focus edits around: {', '.join(memory_guidance.focus_nodes[:4])}.\n"
        if memory_guidance.avoid_nodes_or_patches:
            constraints += f"- Avoid these patches: {', '.join(memory_guidance.avoid_nodes_or_patches[:4])}.\n"
        output_requirements = (
            "- modification summary\n"
            "- changed files\n"
            "- graph.py if changed\n"
            "- prompt.py if changed\n"
            "- rationale linked to failure evidence and memory guidance\n"
        )
        if guard_feedback_text:
            failure_evidence = (failure_evidence + "\n\n[Recent Candidate Generation Failures]\n" + guard_feedback_text).strip()
        components = {
            "graph": workflow_summary,
            "prompt": prompt_summary,
            "operator description": operator_description,
            "log_data": failure_summaries,
            "old round-level experience": experience_context,
            "trajectory memory hint": trajectory_memory_text,
            "experience-guided memory hint": memory_guidance_text,
            "guard feedback": guard_feedback_text,
        }
        if self.experience_guided_config.enable_prompt_budget_diagnostics:
            self._log_prompt_budget(components)
        prompt = self.graph_utils.create_memory_guided_optimize_prompt(
            objective=objective,
            workflow_context=workflow_context,
            failure_evidence=failure_evidence,
            memory_guidance=memory_block,
            constraints=constraints,
            output_requirements=output_requirements,
            type=self.type,
        )
        if self.experience_guided_config.enable_prompt_budget_diagnostics:
            self._log_prompt_budget({**components, "final assembled optimize prompt": prompt})
        return prompt

    def _log_prompt_budget(self, components: Dict[str, str]) -> None:
        rows = ["[PromptBudget]"]
        total_chars = 0
        for key, value in components.items():
            chars = len(value or "")
            total_chars += chars
            approx_tokens = chars // 4
            slug = key.replace(" ", "_")
            rows.append(f"{slug}_chars={chars} {slug}_approx_tokens={approx_tokens}")
        rows.append(f"total_chars={total_chars} total_approx_tokens={total_chars // 4}")
        logger.info(" ".join(rows))

    def _build_experience_context(
        self,
        sample_round: int,
        old_experience: str,
        memory_guidance_text: str,
        memory_guidance: MemoryGuidance,
    ) -> Tuple[str, str]:
        if self.type == "code":
            if memory_guidance_text.strip():
                return memory_guidance_text[:700] + "\n[Supporting Old Experience]\n" + old_experience[:700], "hybrid"
            return old_experience, "old_experience_fallback"
        if self.experience_guided_enabled and self.experience_guided_config.use_memory_as_experience_replacement:
            if memory_guidance_text.strip():
                return memory_guidance_text, "memory_replacement"
            return old_experience[:900], "old_experience_fallback"
        if self.experience_guided_enabled and memory_guidance_text.strip():
            return memory_guidance_text + "\n[Supporting Old Experience]\n" + old_experience[:700], "hybrid"
        return old_experience, "old_experience_fallback"

    def _build_guard_feedback_text(self, failures: List[str]) -> str:
        if not failures:
            return ""
        rows = []
        summarized = self._summarize_recent_generation_failures(failures)
        if summarized:
            rows.append(f"- Consolidated root causes: {summarized}")
        for item in failures[-3:]:
            rows.append(f"- {item}")
        return "\n".join(rows)

    @staticmethod
    def _summarize_recent_generation_failures(failures: List[str]) -> str:
        if not failures:
            return ""
        normalized: List[str] = []
        for item in failures[-6:]:
            reason = str(item).strip()
            lower = reason.lower()
            if "repeats known history:" in lower:
                idx = lower.find("repeats known history:")
                reason = "repeated modification: " + reason[idx + len("repeats known history:"):].strip()
            elif "because " in lower:
                idx = lower.find("because ")
                reason = reason[idx + len("because "):].strip().rstrip(".")
            elif "invalid graph rejected by guard" in lower:
                reason = reason.split("invalid graph rejected by guard", 1)[-1].strip(" :.")
            if reason and reason not in normalized:
                normalized.append(reason)
        return "; ".join(normalized[:4])

    def _load_failure_cards(self, round_number: int) -> List[Dict[str, Any]]:
        log_dir = self.data_utils.root_path / "workflows" / f"round_{round_number}" / "log.json"
        if not log_dir.exists():
            return []
        try:
            raw = read_json_file(log_dir, encoding="utf-8")
            if isinstance(raw, dict):
                raw = [raw]
            return self.data_utils.build_failure_cards(raw, max_cases=3, max_chars_per_case=900)
        except Exception:
            return []

    def _experience_modifications(
        self,
        processed_experience: Dict[int, Any],
        sample_round: int,
    ) -> Tuple[List[str], List[str]]:
        payload = processed_experience.get(sample_round, {})
        failure = [str(item.get("modification", "")) for item in payload.get("failure", {}).values()]
        success = [str(item.get("modification", "")) for item in payload.get("success", {}).values()]
        return failure, success

    @staticmethod
    def _detect_degenerate_candidate(
        response: Dict[str, str],
        parent_graph: str,
        parent_prompt: str,
    ) -> List[str]:
        reasons: List[str] = []
        graph_text = str(response.get("graph", "") or "")
        prompt_text = str(response.get("prompt", "") or "")
        graph_lower = graph_text.lower()
        prompt_lower = prompt_text.lower().strip()

        await_calls = graph_text.count("await self.")
        custom_calls = graph_text.count("await self.custom(")
        parent_await_calls = str(parent_graph or "").count("await self.")

        generic_prompt_markers = [
            'xxx_prompt = """solve it."""',
            "xxx_prompt = '''solve it.'''",
            'xxx_prompt = "solve it."',
            "solve it.",
        ]
        generic_prompt = any(marker in prompt_lower for marker in generic_prompt_markers)

        if graph_lower.strip() in {"graph.py", "graph"}:
            reasons.append("graph_payload_is_filename_placeholder")
        if prompt_lower.strip() in {"prompt.py", "prompt"}:
            reasons.append("prompt_payload_is_filename_placeholder")
        if await_calls == 1 and custom_calls == 1 and parent_await_calls >= 3:
            reasons.append("collapsed_workflow_to_single_custom_call")
        if generic_prompt and len(prompt_text.strip()) <= 40:
            reasons.append("prompt_replaced_with_generic_solve_it")
        if "final answer:" not in prompt_lower and generic_prompt and "custom(" in graph_lower:
            reasons.append("missing_answer_format_constraints")
        return reasons

    def _persist_candidate_snapshot(
        self,
        directory: str,
        candidate_attempt: int,
        sample_round: int,
        response: Dict[str, str],
    ) -> str:
        candidate_dir = Path(directory) / "candidate_attempts" / f"attempt_{candidate_attempt:02d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        modification_text = str(response.get("modification", "") or "")
        graph_text = str(response.get("graph", "") or "")
        prompt_text = str(response.get("prompt", "") or "")

        (candidate_dir / "modification.txt").write_text(modification_text, encoding="utf-8")
        (candidate_dir / "graph.py").write_text(graph_text, encoding="utf-8")
        (candidate_dir / "prompt.py").write_text(prompt_text, encoding="utf-8")

        metadata = {
            "round": self.round + 1,
            "parent_round": sample_round,
            "candidate_attempt": candidate_attempt,
            "status": "pending",
            "modification_preview": modification_text[:500],
            "graph_chars": len(graph_text),
            "prompt_chars": len(prompt_text),
        }

        (candidate_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(candidate_dir)

    def _update_candidate_snapshot_status(
        self,
        candidate_dir: str,
        status: str,
        rejection_kind: str = "",
        rejection_reasons: Optional[List[str]] = None,
        guard_result: Optional[WorkflowGuardResult] = None,
    ) -> None:
        metadata_path = Path(candidate_dir) / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = {}
        metadata["status"] = status
        if rejection_kind:
            metadata["rejection_kind"] = rejection_kind
        if rejection_reasons is not None:
            metadata["rejection_reasons"] = [str(item) for item in rejection_reasons]
        if guard_result is not None:
            metadata["guard_valid"] = bool(guard_result.valid)
            metadata["guard_repaired"] = bool(guard_result.repaired)
            metadata["guard_issues"] = [str(item) for item in guard_result.issues]
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _slugify(text: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_") or "candidate"

    def _apply_memory_guided_parent_selection(
        self,
        top_rounds: List[Dict[str, Any]],
        graph_path: str,
        processed_experience: Dict[int, Any],
    ) -> List[Dict[str, Any]]:
        if self.type == "code":
            logger.info("[MemoryParentSelection] skipped for code task to preserve stable parent selection")
            return top_rounds
        rescored: List[Tuple[float, Dict[str, Any]]] = []
        for item in top_rounds:
            round_num = int(item["round"])
            prompt, graph_load = self.graph_utils.read_graph_files(round_num, graph_path)
            workflow_summary, prompt_summary = self.graph_utils.summarize_workflow(graph_load, prompt)
            cards = self._load_failure_cards(round_num)
            failed_mods, success_mods = self._experience_modifications(processed_experience, round_num)
            state = self.experience_controller.build_optimization_state_query(
                parent_round=round_num,
                parent_workflow_id=f"{self.dataset}_round_{round_num}",
                score=float(item["score"]),
                failure_cards=cards,
                graph_summary=workflow_summary,
                prompt_summary=prompt_summary,
                recent_failed_modifications=failed_mods,
                recent_successful_modifications=success_mods,
            )
            guidance = self.experience_controller.build_memory_guidance(state, limit=6)
            bonus = 0.05 * len(guidance.recommended_patch_rules) + 0.03 * len(guidance.preserve_nodes)
            penalty = 0.06 * len(guidance.avoid_nodes_or_patches)
            final_score = float(item["score"]) + bonus - penalty
            logger.info(
                f"[MemoryParentSelection] round={round_num} original_score={item['score']} "
                f"memory_bonus={bonus:.3f} memory_penalty={penalty:.3f} final_score={final_score:.3f} "
                f"reasons={guidance.evidence_summary[:3]}"
            )
            updated = dict(item)
            updated["_memory_parent_score"] = final_score
            rescored.append((final_score, updated))
        rescored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in rescored]

    def _select_parent_round_candidates(
        self,
        graph_path: str,
        processed_experience: Dict[int, Any],
        recent_generation_failures: List[str],
        blocked_parent_rounds: List[int],
    ) -> List[Dict[str, Any]]:
        candidate_pool_size = self.sample
        if self.type == "code" and (
            blocked_parent_rounds or self._count_recent_duplicate_failures(recent_generation_failures) > 0
        ):
            candidate_pool_size = max(self.sample, 3)
        top_rounds = self.data_utils.get_top_rounds(candidate_pool_size)
        if self.experience_guided_enabled and self.experience_guided_config.use_memory_guided_parent_selection:
            top_rounds = self._apply_memory_guided_parent_selection(top_rounds, graph_path, processed_experience)
        if self.type == "code" and blocked_parent_rounds:
            filtered = [item for item in top_rounds if int(item["round"]) not in blocked_parent_rounds]
            if filtered:
                logger.info(
                    f"[CodeParentDiversification] avoiding blocked parents={blocked_parent_rounds}; "
                    f"candidate_pool={[int(item['round']) for item in filtered]}"
                )
                return filtered
            logger.info(
                f"[CodeParentDiversification] no alternative parents available after blocking "
                f"{blocked_parent_rounds}; falling back to full candidate pool"
            )
        return top_rounds

    @staticmethod
    def _count_recent_duplicate_failures(failures: List[str]) -> int:
        count = 0
        for item in failures[-6:]:
            if "repeats known history" in str(item).lower() or "repeated modification:" in str(item).lower():
                count += 1
        return count

    def _extract_fields_from_response(self, response: str) -> Optional[Dict[str, str]]:
        try:
            import re

            result = {"modification": "", "graph": "", "prompt": ""}
            for field in result.keys():
                pattern = rf"<{field}>(.*?)</{field}>"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    result[field] = match.group(1).strip()

            if result.get("graph"):
                result["graph"] = self.workflow_guard.sanitize_graph_text(result.get("graph", ""))

            if not any(result.values()):
                logger.error("No fields could be extracted from response")
                return None
            return result
        except Exception as e:
            logger.error(f"Error extracting fields from response: {str(e)}")
            return None

    @staticmethod
    def _is_retryable_runtime_error(exc: Exception) -> bool:
        return is_retryable_runtime_error(exc)

    @staticmethod
    def _build_task_query_from_log(log_data: str) -> str:
        if not log_data:
            return ""
        question_lines = []
        for line in str(log_data).splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if stripped.startswith("Q:") or stripped.startswith("Question:"):
                question_lines.append(stripped)
            elif lowered.startswith("pred:") or lowered.startswith("extracted:"):
                continue
            elif stripped and len(question_lines) < 3:
                question_lines.append(stripped)
        query = " ".join(question_lines[:3]).strip()
        return query[:800] if query else str(log_data)[:800]

    async def test(self):
        rounds = [1]
        data = []

        graph_path = f"{self.root_path}/workflows_test"
        json_file_path = self.data_utils.get_results_file_path(graph_path)

        data = self.data_utils.load_results(graph_path)

        for round in rounds:
            directory = self.graph_utils.create_round_directory(graph_path, round)
            self.graph = self.graph_utils.load_graph(round, graph_path)

            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test(self, directory, is_test=True)

            new_data = self.data_utils.create_result_data(round, score, avg_cost, total_cost)
            data.append(new_data)

            self.data_utils.save_results(json_file_path, data)
