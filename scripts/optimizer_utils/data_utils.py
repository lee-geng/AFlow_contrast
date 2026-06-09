import datetime
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.logs import logger
from scripts.utils.common import read_json_file, write_json_file


class DataUtils:
    
    DEFAULT_ALPHA = 0.2
    DEFAULT_LAMBDA = 0.3
    DEFAULT_LOG_SAMPLES = 2
    DEFAULT_LOG_FIELD_CHAR_LIMIT = 280
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.top_scores: List[Dict[str, Any]] = []

    def load_results(self, path: str) -> list:
        result_path = Path(path) / "results.json"
        
        if not result_path.exists():
            return []
        
        try:
            with open(result_path, "r", encoding="utf-8") as json_file:
                return json.load(json_file)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from {result_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading results from {result_path}: {e}")
            return []

    def get_top_rounds(self, sample: int, path: Optional[str] = None, mode: str = "Graph") -> List[Dict]:

        self._load_scores(path, mode)
        unique_rounds: Dict[int, Dict] = {}
        
        for item in self.top_scores:
            round_num = item["round"]
            if round_num not in unique_rounds:
                unique_rounds[round_num] = item
                if len(unique_rounds) >= sample:
                    break
        
        result = []
        if 1 in unique_rounds:
            result.append(unique_rounds[1])
            del unique_rounds[1]
        
        result.extend(unique_rounds.values())
        
        return result[:sample]

    def select_round(self, items: List[Dict]) -> Dict:

        if not items:
            raise ValueError("Item list is empty.")

        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        scores = [item["score"] * 100 for item in sorted_items]

        probabilities = self._compute_probabilities(scores)
        logger.info(f"\nMixed probability distribution: {probabilities}")
        logger.info(f"\nSorted rounds: {sorted_items}")

        selected_index = np.random.choice(len(sorted_items), p=probabilities)
        logger.info(f"\nSelected index: {selected_index}, Selected item: {sorted_items[selected_index]}")

        return sorted_items[selected_index]

    def _compute_probabilities(
        self, 
        scores: List[float], 
        alpha: float = DEFAULT_ALPHA, 
        lambda_: float = DEFAULT_LAMBDA
    ) -> np.ndarray:

        scores = np.array(scores, dtype=np.float64)
        n = len(scores)

        if n == 0:
            raise ValueError("Score list is empty.")

        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

        max_score = np.max(scores)
        shifted_scores = scores - max_score
        exp_weights = np.exp(alpha * shifted_scores)

        sum_exp_weights = np.sum(exp_weights)
        if sum_exp_weights == 0:
            raise ValueError("Sum of exponential weights is 0, cannot normalize.")

        score_prob = exp_weights / sum_exp_weights

        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0):
            mixed_prob = mixed_prob / total_prob

        return mixed_prob

    def load_log(self, cur_round: int, path: Optional[str] = None, mode: str = "Graph") -> str:
        if mode == "Graph":
            log_dir = self.root_path / "workflows" / f"round_{cur_round}" / "log.json"
        else:
            log_dir = Path(path)

        if not log_dir.exists():
            logger.warning(f"Log file not found: {log_dir}")
            return ""
        
        logger.info(f"Loading log from: {log_dir}")
        
        try:
            data = read_json_file(log_dir, encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading log file {log_dir}: {e}")
            return ""

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data)

        if not data:
            return ""

        return self._format_failure_entries(data, top_k=self.DEFAULT_LOG_SAMPLES)

    def summarize_failures(
        self,
        cur_round: int,
        path: Optional[str] = None,
        mode: str = "Graph",
        top_k: int = DEFAULT_LOG_SAMPLES,
    ) -> str:
        if mode == "Graph":
            log_dir = self.root_path / "workflows" / f"round_{cur_round}" / "log.json"
        else:
            log_dir = Path(path)

        if not log_dir.exists():
            logger.warning(f"Log file not found: {log_dir}")
            return ""

        try:
            data = read_json_file(log_dir, encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading log file {log_dir}: {e}")
            return ""

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data)

        return self._format_failure_entries(data, top_k=top_k)

    def build_failure_cards(
        self,
        log_data: List[Dict[str, Any]],
        max_cases: int = 3,
        max_chars_per_case: int = 1200,
    ) -> List[Dict[str, Any]]:
        if not log_data:
            return []
        sample_size = min(max(1, max_cases), len(log_data))
        samples = random.sample(log_data, sample_size)
        cards: List[Dict[str, Any]] = []
        for sample in samples:
            expected = self._truncate(sample.get("right_answer", ""), 220)
            extracted = self._truncate(sample.get("extracted_output", ""), 180)
            model_output = self._truncate(sample.get("model_output", ""), 240)
            failure_type = self._infer_failure_type(sample)
            mismatch_info = self._build_mismatch_info(expected, extracted)
            card = {
                "task_id": sample.get("task_id") or sample.get("_task_id") or "",
                "node_name": sample.get("node_name") or sample.get("failed_node") or "",
                "error_type": sample.get("error_type") or failure_type,
                "failure_type": sample.get("failure_type") or failure_type,
                "extract_result": extracted,
                "mismatch_info": mismatch_info,
                "expected_summary": expected,
                "suspected_cause": sample.get("suspected_cause") or self._infer_suspected_cause(sample, failure_type),
                "input_summary": self._truncate(sample.get("question", ""), 200),
            }
            if not extracted or "runtime" in failure_type or "error" in failure_type:
                card["output_snippet"] = model_output
            serialized = json.dumps(card, ensure_ascii=False)
            if len(serialized) > max_chars_per_case:
                card["output_snippet"] = self._truncate(card.get("output_snippet", ""), 80)
                card["input_summary"] = self._truncate(card.get("input_summary", ""), 120)
            cards.append(card)
        return cards

    def _format_failure_entries(self, entries: List[Dict[str, Any]], top_k: int) -> str:
        cards = self.build_failure_cards(entries, max_cases=top_k, max_chars_per_case=900)
        if not cards:
            return ""
        rows = []
        for idx, card in enumerate(cards, start=1):
            rows.append(
                f"[FailureCard {idx}]\n"
                f"task_id={card.get('task_id') or 'n/a'} node={card.get('node_name') or 'n/a'} "
                f"failure_type={card.get('failure_type') or card.get('error_type') or 'unknown'}\n"
                f"extract_result={card.get('extract_result', '')}\n"
                f"mismatch_info={card.get('mismatch_info', '')}\n"
                f"expected_summary={card.get('expected_summary', '')}\n"
                f"suspected_cause={card.get('suspected_cause', '')}\n"
                f"input_summary={card.get('input_summary', '')}"
                + (f"\noutput_snippet={card.get('output_snippet', '')}" if card.get("output_snippet") else "")
            )
        return "\n\n".join(rows)

    def _truncate(self, value: Any, limit: int = DEFAULT_LOG_FIELD_CHAR_LIMIT) -> str:
        text = " ".join(str(value or "").split())
        return text[:limit]

    def _build_mismatch_info(self, expected: str, extracted: str) -> str:
        if expected and extracted:
            return f"expected={expected} vs extracted={extracted}"
        if expected:
            return f"expected={expected} vs extracted=<empty>"
        return "mismatch_detected"

    def _infer_failure_type(self, sample: Dict[str, Any]) -> str:
        text = " ".join(
            [
                str(sample.get("model_output", "")),
                str(sample.get("extracted_output", "")),
                str(sample.get("question", "")),
            ]
        ).lower()
        if "traceback" in text or "runtime error" in text or "attributeerror" in text:
            return "runtime_error"
        if "\\boxed" not in str(sample.get("model_output", "")) and sample.get("extracted_output", "") == "":
            return "answer_extraction_failure"
        return "answer_mismatch"

    def _infer_suspected_cause(self, sample: Dict[str, Any], failure_type: str) -> str:
        if failure_type == "runtime_error":
            return "workflow execution error"
        if failure_type == "answer_extraction_failure":
            return "final answer formatting or extraction failure"
        return "predicted answer does not align with ground truth"

    def get_results_file_path(self, graph_path: str) -> str:
        
        return str(Path(graph_path) / "results.json")

    def create_result_data(
        self, 
        round: int, 
        score: float, 
        avg_cost: float, 
        total_cost: float
    ) -> dict:

        now = datetime.datetime.now()
        return {
            "round": round,
            "score": score,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "time": now
        }

    def save_results(self, json_file_path: str, data: list) -> None:
        write_json_file(json_file_path, data, encoding="utf-8", indent=4)

    def _load_scores(self, path: Optional[str] = None, mode: str = "Graph") -> List[Dict]:
        if mode == "Graph":
            rounds_dir = self.root_path / "workflows"
        else:
            rounds_dir = Path(path)

        result_file = rounds_dir / "results.json"
        self.top_scores = []

        try:
            data = read_json_file(result_file, encoding="utf-8")
            df = pd.DataFrame(data)

            scores_per_round = df.groupby("round")["score"].mean().to_dict()

            self.top_scores = [
                {"round": round_number, "score": average_score}
                for round_number, average_score in scores_per_round.items()
            ]

            self.top_scores.sort(key=lambda x: x["score"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error loading scores from {result_file}: {e}")
            self.top_scores = []

        return self.top_scores
