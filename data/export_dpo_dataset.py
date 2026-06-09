import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.workflow_revision_model import build_revision_prompt


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


def export_dpo(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        chosen = row.get("chosen_edit") or {}
        rejected = row.get("rejected_edit") or {}

        chosen_payload = {
            "edit_type": chosen.get("edit_type", "unknown"),
            "target_block": chosen.get("target_block", "b0"),
            "edit_text": chosen.get("edit_text", ""),
            "delta_score": chosen.get("delta_score", 0.0),
        }
        rejected_payload = {
            "edit_type": rejected.get("edit_type", "unknown"),
            "target_block": rejected.get("target_block", "b0"),
            "edit_text": rejected.get("edit_text", ""),
            "delta_score": rejected.get("delta_score", 0.0),
        }

        out.append(
            {
                "prompt": build_revision_prompt(row),
                "chosen": json.dumps(chosen_payload, ensure_ascii=False),
                "rejected": json.dumps(rejected_payload, ensure_ascii=False),
                "meta": {
                    "task_family": row.get("task_family"),
                    "parent_id": row.get("parent_id"),
                    "parent_score": row.get("parent_score"),
                    "blame_type": (row.get("blame") or {}).get("blame_type", "unknown"),
                },
            }
        )
    return out


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export revision dataset to DPO preference format")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_path)
    dpo_rows = export_dpo(rows)
    write_jsonl(dpo_rows, Path(args.output_path))
    print(json.dumps({"input_rows": len(rows), "output_rows": len(dpo_rows), "output_path": args.output_path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
