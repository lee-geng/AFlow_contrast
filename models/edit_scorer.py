import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class TextScorerConfig:
    feature_dim: int = 2048


class EditScorer:
    """A lightweight hashed-BOW edit scorer baseline.

    Input tuple: (task_summary, parent_workflow_summary, candidate_edit)
    Output: scalar score.
    """

    def __init__(self, config: TextScorerConfig):
        self.config = config
        self.weight = np.zeros(config.feature_dim, dtype=np.float32)
        self.bias = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def encode(self, task_summary: str, parent_workflow_summary: str, candidate_edit: str) -> np.ndarray:
        vec = np.zeros(self.config.feature_dim, dtype=np.float32)
        merged = f"task: {task_summary}\nparent: {parent_workflow_summary}\nedit: {candidate_edit}"
        tokens = self._tokenize(merged)
        if not tokens:
            return vec

        for token in tokens:
            idx = hash(token) % self.config.feature_dim
            vec[idx] += 1.0

        vec /= float(len(tokens))
        return vec

    def score(self, task_summary: str, parent_workflow_summary: str, candidate_edit: str) -> float:
        x = self.encode(task_summary, parent_workflow_summary, candidate_edit)
        return float(np.dot(self.weight, x) + self.bias)

    def score_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        return np.matmul(feature_matrix, self.weight) + self.bias

    def save(self, output_dir: str) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "edit_scorer.npz", weight=self.weight, bias=np.array([self.bias], dtype=np.float32))
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({"feature_dim": self.config.feature_dim}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "EditScorer":
        model_path = Path(model_dir)
        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        scorer = cls(TextScorerConfig(feature_dim=int(cfg["feature_dim"])))
        weights = np.load(model_path / "edit_scorer.npz")
        scorer.weight = weights["weight"].astype(np.float32)
        scorer.bias = float(weights["bias"][0])
        return scorer
