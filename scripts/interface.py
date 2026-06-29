# -*- coding: utf-8 -*-
# @Date    : 2024-03-21
# @Author  : all
# @Desc    : Interface for AFLOW

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

from scripts.async_llm import LLMsConfig
from scripts.evaluator import DatasetType
from scripts.logs import logger
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.path_utils import get_model_run_root


def load_best_round(
    dataset: str,
    optimized_path: str = "workspace",
    opt_model_name: Optional[str] = None,
    exec_model_name: Optional[str] = None,
) -> int:
    if opt_model_name and exec_model_name:
        result_root = get_model_run_root(optimized_path, dataset, opt_model_name, exec_model_name)
    else:
        result_root = Path(optimized_path) / dataset

    data_utils = DataUtils(str(result_root))
    top_rounds = data_utils.get_top_rounds(sample=2, mode="Graph")
    if len(top_rounds) < 2 or not top_rounds[1]:
        return 1

    return top_rounds[1]["round"]


def load_workflow_class(graph_path: str):
    spec = importlib.util.spec_from_file_location("workflow_module", graph_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["workflow_module"] = module
    spec.loader.exec_module(module)
    return module.Workflow


async def aflow_inference(
    dataset: DatasetType,
    question: str,
    entry_point: Optional[str] = None,
    round: Optional[int] = None,
    llm_name: str = "gpt-4o-mini",
    optimized_path: str = "workspace",
    opt_model_name: Optional[str] = None,
    exec_model_name: Optional[str] = None,
) -> Tuple[str, float]:
    if round is None:
        round = load_best_round(dataset, optimized_path, opt_model_name, exec_model_name)

    logger.info(f"Using round {round} for inference")

    if opt_model_name and exec_model_name:
        workflow_root = get_model_run_root(optimized_path, dataset, opt_model_name, exec_model_name)
    else:
        workflow_root = Path(optimized_path) / dataset

    graph_path = workflow_root / "workflows" / f"round_{round}" / "graph.py"
    if not graph_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {graph_path}")

    workflow_class = load_workflow_class(str(graph_path))
    llm_config = LLMsConfig.default().get(llm_name)
    workflow = workflow_class(
        name=f"{dataset}_workflow",
        llm_config=llm_config,
        dataset=dataset,
    )

    if dataset in ["MBPP", "HumanEval"]:
        answer, cost = await workflow(question, entry_point=entry_point)
    else:
        answer, cost = await workflow(question)

    return answer, cost


if __name__ == "__main__":
    asyncio.run(
        aflow_inference(
            dataset="MBPP",
            question="write a function named add_two_numbers to calculate the sum of two numbers",
            entry_point="add_two_numbers",
        )
    )
