import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimizer_utils.trajectory_memory import TrajectoryMemoryController


def summarize(workflows_dir: str, task_context: str, min_support: int, max_patterns: int) -> Dict[str, Any]:
    controller = TrajectoryMemoryController(min_support=min_support, max_patterns=max_patterns)
    artifacts = controller.refresh(workflows_dir=workflows_dir, task_context=task_context)
    return {
        "workflows_dir": workflows_dir,
        "task_context": task_context,
        "num_success_cases": len(artifacts.success_cases),
        "num_failure_cases": len(artifacts.failure_cases),
        "num_contrast_records": len(artifacts.contrast_records),
        "num_failure_signatures": len(artifacts.failure_signatures),
        "num_success_patterns": len(artifacts.success_patterns),
        "num_edit_rules": len(artifacts.edit_rules),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile+distill AFlow trajectory memory and print summary metrics")
    parser.add_argument("--workflows_dir", type=str, required=True, help="Path to workspace/<dataset>/workflows")
    parser.add_argument("--task_context", type=str, default="unknown")
    parser.add_argument("--min_support", type=int, default=2)
    parser.add_argument("--max_patterns", type=int, default=80)
    parser.add_argument("--output_path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = summarize(
        workflows_dir=args.workflows_dir,
        task_context=args.task_context,
        min_support=args.min_support,
        max_patterns=args.max_patterns,
    )
    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
