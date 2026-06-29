import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from patch_evolution.workflow_compiler import WorkflowPatchCompiler


def parse_csv(raw: Optional[Any]) -> Optional[List[str]]:
    if not raw:
        return None
    if isinstance(raw, (list, tuple)):
        values = []
        for item in raw:
            values.extend(str(item).split(","))
    else:
        values = str(raw).split(",")
    return [part.strip() for part in values if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Compile validated runtime patches into an explicit workflow edit")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workflow_dir", required=True)
    parser.add_argument("--patch_registry_dir", required=True)
    parser.add_argument("--output_dir", default="compiled_workflows")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--patch_ids", default=None, help="Optional comma-separated patch ids to compile")
    parser.add_argument("--compile_min_credit", type=float, default=0.0)
    parser.add_argument("--compile_max_patches", type=int, default=8)
    parser.add_argument(
        "--compile_content_nodes",
        action="store_true",
        help="Insert guarded content-level repair nodes into the compiled workflow",
    )
    parser.add_argument(
        "--content_repair_families",
        nargs="+",
        default="numeric_content,entity_boundary,multi_span,duration",
        help="Content repair families to enable. Accepts comma-separated or space-separated values.",
    )
    parser.add_argument(
        "--content_repair_batches",
        nargs="+",
        default=None,
        help=(
            "Optional explicit content repair node batches to insert. "
            "Choices/aliases include: all, numeric, multispan, entity, duration."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    content_repair_batches = parse_csv(args.content_repair_batches)
    compiler = WorkflowPatchCompiler(
        min_credit=args.compile_min_credit,
        max_patches=args.compile_max_patches,
        compile_content_nodes=args.compile_content_nodes or bool(content_repair_batches),
        content_repair_families=parse_csv(args.content_repair_families),
        content_repair_batches=content_repair_batches,
    )
    result = compiler.compile_from_registry(
        workflow_dir=args.workflow_dir,
        patch_registry_dir=args.patch_registry_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        run_id=args.run_id,
        patch_ids=parse_csv(args.patch_ids),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
