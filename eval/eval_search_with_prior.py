import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimizer_utils.revision_prior import load_prior_usage_logs


def load_results(workflows_dir: Path) -> List[Dict[str, Any]]:
    result_file = workflows_dir / "results.json"
    if not result_file.exists() or result_file.stat().st_size == 0:
        return []
    payload = json.loads(result_file.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else []


def summarize(workflows_dir: str) -> Dict[str, Any]:
    wf_dir = Path(workflows_dir)
    results = load_results(wf_dir)
    prior_logs = load_prior_usage_logs(str(wf_dir))

    rounds = sorted({int(item.get("round")) for item in results if isinstance(item, dict) and item.get("round") is not None})
    best_score = max([float(item.get("score", 0.0)) for item in results], default=0.0)
    total_cost = sum(float(item.get("total_cost", 0.0)) for item in results)

    best_round = None
    for r in rounds:
        r_scores = [float(item.get("score", 0.0)) for item in results if int(item.get("round", -1)) == r]
        if r_scores and max(r_scores) == best_score:
            best_round = r
            break

    executed = [x for x in prior_logs if x.get("executed")]
    deltas = [float(x.get("delta_score", 0.0)) for x in executed]

    return {
        "search_steps": len(rounds),
        "executed_candidates": len(executed),
        "token_or_cost_proxy": total_cost,
        "best_workflow_score": best_score,
        "round_of_best_score": best_round,
        "avg_delta_after_prior": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "prior_call_count": len(prior_logs),
        "same_budget_performance_proxy": best_score,
        "faster_convergence_proxy": best_round if best_round is not None else -1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AFlow search with amortized revision prior")
    parser.add_argument("--workflows_dir", type=str, required=True, help="Path to workspace/<dataset>/workflows")
    parser.add_argument("--output_path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = summarize(args.workflows_dir)

    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
