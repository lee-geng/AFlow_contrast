# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 20:00 PM
# @Author  : didi
# @Desc    : Entrance of AFlow.

import argparse
from typing import Dict, List

from data.download_data import download
from scripts.async_llm import LLMsConfig
from scripts.optimizer import Optimizer
from scripts.optimizer_utils.run_paths import (
    ensure_dataset_template_initialized,
    resolve_optimized_path,
    write_run_metadata,
)


class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators


EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "DROP": ExperimentConfig(dataset="DROP", question_type="qa", operators=["Custom", "AnswerGenerate", "ScEnsemble"]),
    "HotpotQA": ExperimentConfig(dataset="HotpotQA", question_type="qa", operators=["Custom", "AnswerGenerate", "ScEnsemble"]),
    "MATH": ExperimentConfig(
        dataset="MATH",
        question_type="math",
        operators=[
            "Custom",
            "Plan",
            "AnswerGenerate",
            "Review",
            "Revise",
            "Verify",
            "ExtractAnswer",
            "ScEnsemble",
            "Programmer",
        ],
    ),
    "GSM8K": ExperimentConfig(dataset="GSM8K", question_type="math", operators=["Custom", "ScEnsemble", "Programmer"]),
    "MBPP": ExperimentConfig(dataset="MBPP", question_type="code", operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]),
    "HumanEval": ExperimentConfig(dataset="HumanEval", question_type="code", operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]),
    "LiveCodeBench": ExperimentConfig(dataset="LiveCodeBench", question_type="code", operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]),
}
def parse_args():
    parser = argparse.ArgumentParser(description="AFlow Optimizer")
    parser.add_argument("--dataset", type=str, choices=list(EXPERIMENT_CONFIGS.keys()), required=True, help="Dataset type")
    parser.add_argument("--sample", type=int, default=4, help="Sample count")
    parser.add_argument("--optimized_path", type=str, default="workspace", help="Optimized result save path")
    parser.add_argument(
        "--separate_model_artifacts",
        action="store_true",
        help="Store outputs under a model-scoped subdirectory so runs from different models do not mix",
    )
    parser.add_argument(
        "--artifact_tag",
        type=str,
        default="",
        help="Optional extra tag appended to the model-scoped artifact directory name",
    )
    parser.add_argument("--initial_round", type=int, default=1, help="Initial round")
    parser.add_argument("--max_rounds", type=int, default=20, help="Max iteration rounds")
    parser.add_argument(
        "--check_convergence",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Enable early stop (true/false)",
    )
    parser.add_argument("--validation_rounds", type=int, default=1, help="Validation rounds")
    parser.add_argument(
        "--validation_subset_size",
        type=int,
        default=0,
        help="If > 0, only evaluate the first N validation tasks per round",
    )
    parser.add_argument(
        "--eval_max_concurrent_tasks",
        type=int,
        default=10,
        help="Maximum number of validation/test tasks evaluated concurrently",
    )
    parser.add_argument(
        "--if_force_download",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether enforce dataset download.",
    )
    parser.add_argument(
        "--opt_model_name",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Optimization model name.",
    )
    parser.add_argument(
        "--exec_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Execution model name.",
    )

    parser.add_argument("--use_revision_prior", action="store_true", help="Enable amortized workflow revision prior")
    parser.add_argument("--revision_prior_path", type=str, default="", help="Path to trained revision model checkpoint")
    parser.add_argument(
        "--revision_prior_mode",
        type=str,
        choices=["generate_then_search", "rank_then_search"],
        default="generate_then_search",
        help="Integration mode for revision prior",
    )
    parser.add_argument(
        "--revision_rank_candidates",
        type=int,
        default=3,
        help="Number of optimize candidates for rank_then_search mode",
    )
    parser.add_argument(
        "--use_trajectory_memory",
        action="store_true",
        help="Enable case/contrast memory distillation and failure-aware minimal edit hints",
    )
    parser.add_argument(
        "--memory_min_support",
        type=int,
        default=2,
        help="Minimum support for keeping distilled signatures/patterns",
    )
    parser.add_argument(
        "--memory_max_patterns",
        type=int,
        default=80,
        help="Maximum number of distilled signatures/patterns/rules retained",
    )
    parser.add_argument(
        "--workflow_guard_mode",
        type=str,
        choices=["off", "repair_then_continue", "fail_fast"],
        default="repair_then_continue",
        help="Workflow pre-eval guard mode: auto-repair+continue, fail-fast, or off",
    )
    parser.add_argument("--experience_memory_enabled", action="store_true", help="Enable experience-guided workflow search")
    parser.add_argument(
        "--experience_store_path",
        type=str,
        default="",
        help="Directory for attribution events, compressed memories, and metrics",
    )
    parser.add_argument("--experience_min_support", type=int, default=3, help="Minimum support for memory promotion")
    parser.add_argument(
        "--experience_min_confidence",
        type=float,
        default=0.6,
        help="Minimum confidence for memory promotion",
    )
    parser.add_argument(
        "--experience_compression_interval",
        type=int,
        default=10,
        help="Compress buffered attribution events every N new events",
    )
    parser.add_argument("--attribution_enabled", action="store_true", help="Enable attribution event generation")
    parser.add_argument(
        "--attribution_mode",
        type=str,
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Attribution analyzer mode",
    )
    parser.add_argument(
        "--adaptive_validation_enabled",
        action="store_true",
        help="Enable experience-guided staged validation",
    )
    parser.add_argument(
        "--disable_staged_validation",
        action="store_true",
        help="When experience-guided mode is enabled, skip targeted/anchor stages and evaluate only full validation",
    )
    parser.add_argument("--targeted_failure_size", type=int, default=8, help="Targeted failure mini-validation size")
    parser.add_argument("--success_guard_size", type=int, default=8, help="Success guard mini-validation size")
    parser.add_argument("--anchor_size", type=int, default=16, help="Shared anchor validation size")
    parser.add_argument(
        "--full_validation_top_k",
        type=int,
        default=3,
        help="Only anchor-promising candidates near top-k history get full validation",
    )
    parser.add_argument("--adaptive_beta", type=float, default=0.5, help="Targeted gain weight in selection score")
    parser.add_argument("--adaptive_gamma", type=float, default=0.5, help="Regression loss weight in selection score")
    parser.add_argument("--adaptive_lambda_cost", type=float, default=0.1, help="Cost weight in selection score")
    parser.add_argument(
        "--task_embedding_enabled",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Enable embedding-based task similarity for experience-guided search",
    )
    parser.add_argument(
        "--task_embedding_model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model name for task similarity",
    )
    parser.add_argument(
        "--task_embedding_base_url",
        type=str,
        default="http://localhost:8090/v1",
        help="OpenAI-compatible embedding endpoint base URL",
    )
    parser.add_argument(
        "--task_embedding_api_key",
        type=str,
        default="EMPTY",
        help="API key for the embedding endpoint",
    )
    parser.add_argument("--task_embedding_dim", type=int, default=64, help="Fallback embedding dimension")
    parser.add_argument(
        "--task_similarity_threshold",
        type=float,
        default=0.55,
        help="Minimum cosine similarity for memory retrieval",
    )
    parser.add_argument("--enable_prompt_budget_diagnostics", action="store_true", help="Log prompt component sizes")
    parser.add_argument("--use_failure_cards", action="store_true", help="Use compact failure cards instead of raw logs")
    parser.add_argument(
        "--use_optimization_state_memory_query",
        action="store_true",
        help="Retrieve memories using optimization-state similarity",
    )
    parser.add_argument(
        "--use_memory_guided_parent_selection",
        action="store_true",
        help="Adjust parent round selection with memory-aware scoring",
    )
    parser.add_argument(
        "--use_memory_guided_patch_scope",
        action="store_true",
        help="Derive patch scope from retrieved memory guidance",
    )
    parser.add_argument(
        "--use_memory_as_experience_replacement",
        action="store_true",
        help="Use compressed memory as the primary experience context",
    )
    parser.add_argument(
        "--use_graph_prompt_summarization",
        action="store_true",
        help="Summarize graph/prompt instead of always sending full files",
    )
    parser.add_argument(
        "--max_candidate_attempts_per_round",
        type=int,
        default=3,
        help="Maximum optimize candidates tried inside a single round before moving on",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = EXPERIMENT_CONFIGS[args.dataset]

    models_config = LLMsConfig.default()
    opt_llm_config = models_config.get(args.opt_model_name)
    if opt_llm_config is None:
        raise ValueError(
            f"The optimization model '{args.opt_model_name}' was not found in config. "
            "Please add it or specify a valid model with --opt_model_name."
        )

    exec_llm_config = models_config.get(args.exec_model_name)
    if exec_llm_config is None:
        raise ValueError(
            f"The execution model '{args.exec_model_name}' was not found in config. "
            "Please add it or specify a valid model with --exec_model_name."
        )

    resolved_optimized_path = resolve_optimized_path(
        optimized_path=args.optimized_path,
        opt_model_name=args.opt_model_name,
        exec_model_name=args.exec_model_name,
        separate_model_artifacts=args.separate_model_artifacts,
        artifact_tag=args.artifact_tag,
    )

    download(["datasets"], force_download=args.if_force_download)
    ensure_dataset_template_initialized(
        optimized_path=resolved_optimized_path,
        dataset=config.dataset,
    )
    write_run_metadata(
        optimized_path=resolved_optimized_path,
        dataset=config.dataset,
        opt_model_name=args.opt_model_name,
        exec_model_name=args.exec_model_name,
        artifact_tag=args.artifact_tag,
    )

    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        check_convergence=args.check_convergence,
        operators=config.operators,
        optimized_path=resolved_optimized_path,
        sample=args.sample,
        initial_round=args.initial_round,
        max_rounds=args.max_rounds,
        validation_rounds=args.validation_rounds,
        validation_subset_size=args.validation_subset_size,
        use_revision_prior=args.use_revision_prior,
        revision_prior_path=args.revision_prior_path,
        revision_prior_mode=args.revision_prior_mode,
        revision_rank_candidates=args.revision_rank_candidates,
        use_trajectory_memory=args.use_trajectory_memory,
        memory_min_support=args.memory_min_support,
        memory_max_patterns=args.memory_max_patterns,
        workflow_guard_mode=args.workflow_guard_mode,
        experience_memory_enabled=args.experience_memory_enabled,
        experience_store_path=args.experience_store_path,
        experience_min_support=args.experience_min_support,
        experience_min_confidence=args.experience_min_confidence,
        experience_compression_interval=args.experience_compression_interval,
        attribution_enabled=args.attribution_enabled or args.experience_memory_enabled,
        attribution_mode=args.attribution_mode,
        adaptive_validation_enabled=args.adaptive_validation_enabled,
        use_staged_validation=not args.disable_staged_validation,
        targeted_failure_size=args.targeted_failure_size,
        success_guard_size=args.success_guard_size,
        anchor_size=args.anchor_size,
        full_validation_top_k=args.full_validation_top_k,
        adaptive_beta=args.adaptive_beta,
        adaptive_gamma=args.adaptive_gamma,
        adaptive_lambda_cost=args.adaptive_lambda_cost,
        task_embedding_enabled=args.task_embedding_enabled,
        task_embedding_model_name=args.task_embedding_model_name,
        task_embedding_base_url=args.task_embedding_base_url,
        task_embedding_api_key=args.task_embedding_api_key,
        task_embedding_dim=args.task_embedding_dim,
        task_similarity_threshold=args.task_similarity_threshold,
        enable_prompt_budget_diagnostics=args.enable_prompt_budget_diagnostics,
        use_failure_cards=args.use_failure_cards,
        use_optimization_state_memory_query=args.use_optimization_state_memory_query,
        use_memory_guided_parent_selection=args.use_memory_guided_parent_selection,
        use_memory_guided_patch_scope=args.use_memory_guided_patch_scope,
        use_memory_as_experience_replacement=args.use_memory_as_experience_replacement,
        use_graph_prompt_summarization=args.use_graph_prompt_summarization,
        max_candidate_attempts_per_round=args.max_candidate_attempts_per_round,
        eval_max_concurrent_tasks=args.eval_max_concurrent_tasks,
    )

    optimizer.optimize("Graph")
