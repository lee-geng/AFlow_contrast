import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.edit_scorer import EditScorer, TextScorerConfig


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} in {path}: {exc}") from exc
    return records


def build_feature_matrices(samples: List[Dict], scorer: EditScorer) -> Tuple[np.ndarray, np.ndarray]:
    pos_vectors: List[np.ndarray] = []
    neg_vectors: List[np.ndarray] = []

    for sample in samples:
        task_summary = str(sample.get("task_summary") or sample.get("task_family") or "")
        parent_summary = str(sample.get("parent_workflow_summary") or sample.get("parent_workflow") or "")

        pos_edit = str(sample.get("positive_edit") or "")
        neg_edit = str(sample.get("negative_edit") or "")

        pos_vectors.append(scorer.encode(task_summary, parent_summary, pos_edit))
        neg_vectors.append(scorer.encode(task_summary, parent_summary, neg_edit))

    if not pos_vectors:
        return (
            np.zeros((0, scorer.config.feature_dim), dtype=np.float32),
            np.zeros((0, scorer.config.feature_dim), dtype=np.float32),
        )

    return np.stack(pos_vectors, axis=0), np.stack(neg_vectors, axis=0)


def evaluate_pairwise(scorer: EditScorer, pos_x: np.ndarray, neg_x: np.ndarray, margin: float) -> Dict[str, float]:
    pos_scores = scorer.score_batch(pos_x)
    neg_scores = scorer.score_batch(neg_x)
    losses = np.maximum(0.0, margin - pos_scores + neg_scores)
    pair_acc = float(np.mean(pos_scores > neg_scores)) if len(pos_scores) else 0.0
    return {
        "loss": float(np.mean(losses)) if len(losses) else 0.0,
        "pair_acc": pair_acc,
    }


def train(
    train_samples: List[Dict],
    val_samples: List[Dict],
    output_dir: str,
    margin: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    feature_dim: int,
    seed: int,
) -> None:
    if not train_samples:
        raise ValueError("Training dataset is empty")

    rng = np.random.default_rng(seed)
    scorer = EditScorer(TextScorerConfig(feature_dim=feature_dim))

    train_pos, train_neg = build_feature_matrices(train_samples, scorer)
    val_pos, val_neg = build_feature_matrices(val_samples, scorer) if val_samples else (
        np.zeros((0, feature_dim), dtype=np.float32),
        np.zeros((0, feature_dim), dtype=np.float32),
    )

    history: List[Dict[str, float]] = []

    n = train_pos.shape[0]
    for epoch in range(1, epochs + 1):
        indices = rng.permutation(n)
        epoch_loss = 0.0
        active_count = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            pos_batch = train_pos[batch_idx]
            neg_batch = train_neg[batch_idx]

            pos_scores = scorer.score_batch(pos_batch)
            neg_scores = scorer.score_batch(neg_batch)
            hinge = margin - pos_scores + neg_scores
            active = hinge > 0

            if np.any(active):
                grad = (pos_batch[active] - neg_batch[active]).mean(axis=0)
                scorer.weight += learning_rate * grad.astype(np.float32)
                active_count += int(np.sum(active))

            loss = np.maximum(0.0, hinge)
            epoch_loss += float(np.sum(loss))

        train_metrics = evaluate_pairwise(scorer, train_pos, train_neg, margin)
        val_metrics = evaluate_pairwise(scorer, val_pos, val_neg, margin) if len(val_samples) else {"loss": 0.0, "pair_acc": 0.0}

        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_pair_acc": train_metrics["pair_acc"],
            "val_loss": val_metrics["loss"],
            "val_pair_acc": val_metrics["pair_acc"],
            "active_pairs": float(active_count),
            "avg_batch_loss": epoch_loss / max(n, 1),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    scorer.save(output_dir)
    with open(Path(output_dir) / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pairwise edit ranker")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_samples = load_jsonl(args.train_path)
    val_samples = load_jsonl(args.val_path) if args.val_path else []

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(
        train_samples=train_samples,
        val_samples=val_samples,
        output_dir=args.output_dir,
        margin=args.margin,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        feature_dim=args.feature_dim,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
