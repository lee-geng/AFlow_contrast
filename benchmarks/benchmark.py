import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from scripts.logs import logger
from scripts.utils.common import write_json_file
from contrastive_experience.grouping import dataset_success_threshold
from contrastive_experience.trace_context import (
    finish_sample_trace,
    make_sample_id,
    set_trace_config,
    start_sample_trace,
    reset_trace_config,
)
from patch_evolution.guarded_runtime import reset_patch_runtime, set_patch_runtime


class BaseBenchmark(ABC):
    def __init__(self, name: str, file_path: str, log_path: str):
        self.name = name
        self.file_path = file_path
        self.log_path = log_path

    PASS = "PASS"
    FAIL = "FAIL"

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        avg_score = df["score"].mean()
        t_cost = df["cost"].max()
        a_cost = t_cost / len(df) if len(df) > 0 else 0
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score, a_cost, t_cost

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ):
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)

    @abstractmethod
    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        pass

    @abstractmethod
    def get_result_columns(self) -> List[str]:
        pass

    def _trace_result_from_tuple(self, result: Tuple[Any, ...], trace_config=None):
        columns = self.get_result_columns()
        mapped = {column: result[index] for index, column in enumerate(columns) if index < len(result)}
        score = mapped.get("score", 0.0)
        threshold = dataset_success_threshold(self.name, getattr(trace_config, "success_threshold", None))
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = 0.0
        return {
            "final_result": "success" if numeric_score >= threshold else "failure",
            "prediction": mapped.get("prediction") or mapped.get("model_output") or mapped.get("output"),
            "expected": mapped.get("expected_output") or mapped.get("right_answer") or mapped.get("answer"),
            "sample_score": numeric_score,
            "cost": mapped.get("cost"),
        }

    async def evaluate_all_problems(
        self,
        data: List[dict],
        agent: Callable,
        max_concurrent_tasks: int = 50,
        trace_config=None,
    ):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def sem_evaluate(index, problem):
            async with semaphore:
                sample_token = start_sample_trace(make_sample_id(index, problem), problem)
                try:
                    result = await self.evaluate_problem(problem, agent)
                except Exception as e:
                    finish_sample_trace(
                        sample_token,
                        {
                            "final_result": "failure",
                            "prediction": str(e),
                            "expected": None,
                            "sample_score": 0.0,
                            "cost": 0.0,
                        },
                    )
                    raise
                finish_sample_trace(sample_token, self._trace_result_from_tuple(result, trace_config=trace_config))
                return result

        tasks = [sem_evaluate(index, problem) for index, problem in enumerate(data)]
        return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))

    async def run_evaluation(
        self,
        agent: Callable,
        va_list: List[int],
        max_concurrent_tasks: int = 50,
        trace_config=None,
        patch_runtime_config=None,
    ):
        data = await self.load_data(va_list)
        token = set_trace_config(trace_config) if trace_config and trace_config.enabled else None
        patch_token = set_patch_runtime(patch_runtime_config) if patch_runtime_config and patch_runtime_config.enabled else None
        try:
            results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks, trace_config=trace_config)
        finally:
            if token is not None:
                reset_trace_config(token)
            if patch_token is not None:
                reset_patch_runtime(patch_token)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        return average_score, average_cost, total_cost
    

    async def run_baseline(self, agent: Callable, max_concurrent_tasks: int = 50):
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        logger.info(f"Avg Cost:{average_cost:.5f}")
        return average_score, average_cost, total_cost

