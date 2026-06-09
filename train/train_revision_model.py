import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.workflow_revision_model import RevisionModelConfig, WorkflowRevisionModel


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def evaluate_format_validity(model: WorkflowRevisionModel, rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {"format_valid_rate": 0.0, "avg_preference_score": 0.0, "target_block_hit_rate": 0.0}

    valid_count = 0
    pref_scores: List[float] = []
    target_hits = 0

    for row in rows:
        proposal = model.generate_proposal(row)
        valid_count += int(model.valid_proposal_format(proposal))

        chosen = row.get("chosen_edit") or {}
        rejected = row.get("rejected_edit") or {}
        pref = model.score_candidate(row, chosen) - model.score_candidate(row, rejected)
        pref_scores.append(pref)

        target_hits += int(str(proposal.get("target_block", "")) == str(chosen.get("target_block", "")))

    return {
        "format_valid_rate": valid_count / len(rows),
        "avg_preference_score": sum(pref_scores) / len(pref_scores),
        "target_block_hit_rate": target_hits / len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train workflow revision model (SFT baseline with LoRA-like adapters)")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.train_path)
    val_rows = load_jsonl(args.val_path) if args.val_path else []

    model = WorkflowRevisionModel(
        RevisionModelConfig(feature_dim=args.feature_dim, lora_rank=args.lora_rank)
    )

    history = model.train_sft(
        rows=train_rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    metrics = evaluate_format_validity(model, val_rows if val_rows else train_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))

    with open(out_dir / "sft_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({"history": history, "metrics": metrics, "output_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
