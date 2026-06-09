import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experience-guided workflow search metrics")
    parser.add_argument("--store_path", type=str, required=True, help="Path to workflows/experience_store")
    args = parser.parse_args()

    root = Path(args.store_path)
    metrics = _read_jsonl(root / "experience_metrics.jsonl")
    memories = _read_jsonl(root / "workflow_memories.jsonl")
    events = _read_jsonl(root / "attribution_events.jsonl")

    if not metrics:
        print("No experience-guided metrics found.")
        return

    best_score = max(float(row.get("best_validation_score_so_far", 0.0)) for row in metrics)
    final = metrics[-1]
    print("Experience-Guided Search Summary")
    print(f"- rounds_logged: {len(metrics)}")
    print(f"- best_validation_score: {best_score:.5f}")
    print(f"- candidate_workflows: {final.get('number_of_candidate_workflows', 0)}")
    print(f"- targeted_validations: {final.get('number_of_targeted_validations', 0)}")
    print(f"- anchor_validations: {final.get('number_of_anchor_validations', 0)}")
    print(f"- full_validations: {final.get('number_of_full_validations', 0)}")
    print(f"- attribution_events: {len(events)}")
    print(f"- compressed_memories: {len(memories)}")
    print(f"- memory_compression_ratio: {final.get('memory_compression_ratio', 0.0):.4f}")
    print(f"- token_cost_estimate: {final.get('token_cost_estimate', 0.0):.5f}")


if __name__ == "__main__":
    main()
