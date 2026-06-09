import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.workflow_revision_model import WorkflowRevisionModel


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(model: WorkflowRevisionModel, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "num_samples": 0,
            "format_valid_rate": 0.0,
            "target_block_hit_rate": 0.0,
            "avg_preference_margin": 0.0,
            "fuzzy_match_mean": 0.0,
            "blame_coverage": 0.0,
            "edit_type_distribution": {},
            "task_family_distribution": {},
        }

    valid = 0
    target_hits = 0
    pref_margins: List[float] = []
    fuzzy_scores: List[float] = []
    blame_known = 0

    edit_types = Counter()
    task_families = Counter()

    for row in rows:
        proposal = model.generate_proposal(row)
        valid += int(model.valid_proposal_format(proposal))

        chosen = row.get("chosen_edit") or {}
        rejected = row.get("rejected_edit") or {}

        target_hits += int(str(proposal.get("target_block", "")) == str(chosen.get("target_block", "")))

        score_margin = model.score_candidate(row, chosen) - model.score_candidate(row, rejected)
        pref_margins.append(score_margin)

        fuzzy_scores.append(
            SequenceMatcher(
                a=str(proposal.get("edit_text", "")).lower(),
                b=str(chosen.get("edit_text", "")).lower(),
            ).ratio()
        )

        blame = row.get("blame") or {}
        blame_known += int(str(blame.get("blame_type", "unknown")) != "unknown")

        edit_types[str(chosen.get("edit_type", "unknown"))] += 1
        task_families[str(row.get("task_family", "unknown"))] += 1

    n = len(rows)
    return {
        "num_samples": n,
        "format_valid_rate": valid / n,
        "target_block_hit_rate": target_hits / n,
        "avg_preference_margin": sum(pref_margins) / n,
        "fuzzy_match_mean": sum(fuzzy_scores) / n,
        "blame_coverage": blame_known / n,
        "edit_type_distribution": dict(edit_types),
        "task_family_distribution": dict(task_families),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate workflow revision model")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = WorkflowRevisionModel.load(args.model_dir)
    rows = load_jsonl(args.data_path)
    metrics = compute_metrics(model, rows)

    if args.output_path:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
