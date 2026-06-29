import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ["DROP", "HotpotQA", "MATH", "GSM8K", "MBPP", "HumanEval", "LiveCodeBench"]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect operator traces for an existing workflow round")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--optimized_path", type=str, default="workspace")
    parser.add_argument("--opt_model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--exec_model_name", type=str, default="qwen2.5-coder:7b")
    parser.add_argument("--trace_dir", type=str, default="traces")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--trace_full_io", action="store_true")
    parser.add_argument("--trace_preview_chars", type=int, default=512)
    parser.add_argument("--success_threshold", type=float, default=None)
    parser.add_argument("--test", action="store_true", help="Use test split instead of validation split")
    return parser.parse_args()


async def main_async():
    args = parse_args()

    from run import EXPERIMENT_CONFIGS
    from scripts.async_llm import LLMsConfig
    from scripts.evaluator import Evaluator
    from scripts.optimizer import Optimizer
    from contrastive_experience.trace_context import TraceConfig
    from contrastive_experience.trace_logger import default_run_id

    config = EXPERIMENT_CONFIGS[args.dataset]
    models_config = LLMsConfig.default()
    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=models_config.get(args.opt_model_name),
        exec_llm_config=models_config.get(args.exec_model_name),
        operators=config.operators,
        sample=1,
        optimized_path=args.optimized_path,
        initial_round=args.round,
        max_rounds=1,
        validation_rounds=1,
        trace_enabled=True,
        trace_dir=args.trace_dir,
        trace_full_io=args.trace_full_io,
        trace_preview_chars=args.trace_preview_chars,
        success_threshold=args.success_threshold,
        trace_run_id=args.run_id or default_run_id(),
    )
    graph_path = f"{optimizer.root_path}/workflows"
    graph_class = optimizer.graph_utils.load_graph(args.round, graph_path)
    graph = graph_class(name=args.dataset, llm_config=optimizer.execute_llm_config, dataset=args.dataset)
    trace_config = TraceConfig.create(
        enabled=True,
        dataset=args.dataset,
        trace_dir=args.trace_dir,
        run_id=optimizer.trace_run_id,
        round_id=args.round,
        trace_full_io=args.trace_full_io,
        trace_preview_chars=args.trace_preview_chars,
        success_threshold=args.success_threshold,
    )
    evaluator = Evaluator(eval_path=f"{graph_path}/round_{args.round}")
    score, avg_cost, total_cost = await evaluator.graph_evaluate(
        args.dataset,
        lambda name, llm_config, dataset: graph,
        {"dataset": args.dataset, "llm_config": optimizer.execute_llm_config},
        f"{graph_path}/round_{args.round}",
        is_test=args.test,
        trace_config=trace_config,
    )
    print(f"score={score:.5f} avg_cost={avg_cost:.5f} total_cost={total_cost:.5f}")


if __name__ == "__main__":
    asyncio.run(main_async())
