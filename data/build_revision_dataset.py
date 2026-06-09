import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.blame_attributor import RuleBasedBlameAttributor
from data.aflow_adapters import AFlowTreeAdapter, WorkflowNode


def summarize_workflow(workflow: str, max_len: int = 400) -> str:
    compact = re.sub(r"\s+", " ", workflow).strip()
    return compact if len(compact) <= max_len else compact[: max_len - 3] + "..."


def infer_edit_type(edit_text: str) -> str:
    t = (edit_text or "").lower()
    if any(k in t for k in ["insert", "add", "append"]):
        return "insert_block"
    if any(k in t for k in ["delete", "remove"]):
        return "delete_block"
    if any(k in t for k in ["prompt", "instruction", "wording"]):
        return "prompt_change"
    if any(k in t for k in ["tool", "api"]):
        return "tool_change"
    if any(k in t for k in ["verify", "check", "review", "test"]):
        return "verification_patch"
    return "unknown"


def near_no_change(parent_workflow: str, child_workflow: str, edit_text: str, threshold: float = 0.985) -> bool:
    if not parent_workflow or not child_workflow:
        return False
    pa = re.sub(r"\s+", "", parent_workflow)
    ch = re.sub(r"\s+", "", child_workflow)
    if not pa or not ch:
        return False
    common = sum(1 for i, c in enumerate(ch[: len(pa)]) if pa[i] == c)
    ratio = common / max(len(pa), 1)
    return ratio > threshold and len((edit_text or "").strip()) < 8


def build_execution_feedback(node: WorkflowNode, parent_score: float) -> Dict[str, Any]:
    fb = node.log_feedback or {}
    return {
        "score_mean": parent_score,
        "score_std": node.score_std if node.score_std is not None else fb.get("score_std", 0.0),
        "failure_summary": str(fb.get("failure_summary") or ""),
        "trace_summary": str(fb.get("trace_summary") or ""),
    }


def build_samples(
    nodes: Dict[str, WorkflowNode],
    pos_threshold: float,
    neg_threshold: float,
    max_score_std: float,
    drop_unknown_blame_if_weak_feedback: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    attributor = RuleBasedBlameAttributor()
    children_by_parent: Dict[str, List[WorkflowNode]] = defaultdict(list)
    for node in nodes.values():
        if node.parent_id:
            children_by_parent[node.parent_id].append(node)

    stats = {
        "num_nodes": len(nodes),
        "num_edges": sum(len(v) for v in children_by_parent.values()),
        "num_sibling_pairs": 0,
        "num_filtered": 0,
        "filtered_reasons": Counter(),
        "edit_type_distribution": Counter(),
        "blame_type_distribution": Counter(),
    }

    samples: List[Dict[str, Any]] = []

    for parent_id, children in children_by_parent.items():
        parent = nodes.get(parent_id)
        if parent is None or parent.score_mean is None or not parent.workflow:
            continue

        positives: List[Tuple[WorkflowNode, float]] = []
        negatives: List[Tuple[WorkflowNode, float]] = []

        for child in children:
            if child.score_mean is None:
                stats["num_filtered"] += 1
                stats["filtered_reasons"]["missing_child_score"] += 1
                continue
            if child.score_std is not None and child.score_std > max_score_std:
                stats["num_filtered"] += 1
                stats["filtered_reasons"]["score_std_too_large"] += 1
                continue
            if not child.workflow or not child.edit_text:
                stats["num_filtered"] += 1
                stats["filtered_reasons"]["empty_workflow_or_edit"] += 1
                continue
            if near_no_change(parent.workflow, child.workflow, child.edit_text):
                stats["num_filtered"] += 1
                stats["filtered_reasons"]["child_parent_nearly_same"] += 1
                continue

            delta = child.score_mean - parent.score_mean
            if delta > pos_threshold:
                positives.append((child, delta))
            elif delta <= neg_threshold:
                negatives.append((child, delta))
            else:
                stats["num_filtered"] += 1
                stats["filtered_reasons"]["delta_in_ambiguous_zone"] += 1

        positives.sort(key=lambda x: x[1], reverse=True)
        negatives.sort(key=lambda x: x[1])
        pair_n = min(len(positives), len(negatives))
        stats["num_sibling_pairs"] += pair_n

        for idx in range(pair_n):
            chosen, chosen_delta = positives[idx]
            rejected, rejected_delta = negatives[idx]

            feedback = build_execution_feedback(parent, parent.score_mean)
            blame = attributor.attribute(
                parent_workflow=parent.workflow,
                execution_feedback=feedback,
                edit_text=chosen.edit_text,
                child_workflow=chosen.workflow,
            )

            if drop_unknown_blame_if_weak_feedback:
                weak_feedback = not feedback.get("failure_summary") and not feedback.get("trace_summary")
                if blame.blame_type == "unknown" and weak_feedback:
                    stats["num_filtered"] += 1
                    stats["filtered_reasons"]["unknown_blame_with_weak_feedback"] += 1
                    continue

            chosen_type = infer_edit_type(chosen.edit_text)
            rejected_type = infer_edit_type(rejected.edit_text)
            stats["edit_type_distribution"][chosen_type] += 1
            stats["edit_type_distribution"][rejected_type] += 1
            stats["blame_type_distribution"][blame.blame_type] += 1

            row = {
                "task_family": chosen.task_family or parent.task_family,
                "task_summary": chosen.task_summary or parent.task_summary,
                "parent_id": parent_id,
                "parent_workflow": parent.workflow,
                "parent_score": parent.score_mean,
                "execution_feedback": feedback,
                "blame": asdict(blame),
                "chosen_edit": {
                    "edit_text": chosen.edit_text,
                    "edit_type": chosen_type,
                    "target_block": blame.target_block,
                    "child_workflow": chosen.workflow,
                    "delta_score": chosen_delta,
                },
                "rejected_edit": {
                    "edit_text": rejected.edit_text,
                    "edit_type": rejected_type,
                    "target_block": blame.target_block,
                    "child_workflow": rejected.workflow,
                    "delta_score": rejected_delta,
                },
            }
            samples.append(row)

    stats["filtered_reasons"] = dict(stats["filtered_reasons"])
    stats["edit_type_distribution"] = dict(stats["edit_type_distribution"])
    stats["blame_type_distribution"] = dict(stats["blame_type_distribution"])
    return samples, stats


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_val(rows: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rnd = random.Random(seed)
    shuffled = list(rows)
    rnd.shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    val_rows = shuffled[:val_size]
    train_rows = shuffled[val_size:]
    return train_rows, val_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build parent-conditioned, blame-aware revision dataset")
    parser.add_argument("--input_path", type=str, required=True, help="AFlow tree JSON or workspace/workflows directory")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_family", type=str, default="")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos_threshold", type=float, default=0.03)
    parser.add_argument("--neg_threshold", type=float, default=0.0)
    parser.add_argument("--max_score_std", type=float, default=0.2)
    parser.add_argument("--keep_unknown_blame", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = AFlowTreeAdapter(task_family=args.task_family or None)
    bundle = adapter.load(args.input_path)

    rows, stats = build_samples(
        nodes=bundle.nodes,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        max_score_std=args.max_score_std,
        drop_unknown_blame_if_weak_feedback=not args.keep_unknown_blame,
    )

    train_rows, val_rows = split_train_val(rows, args.val_ratio, args.seed)
    out_dir = Path(args.output_dir)

    train_path = out_dir / "train_revision.jsonl"
    val_path = out_dir / "val_revision.jsonl"
    stats_path = out_dir / "revision_stats.json"

    write_jsonl(train_rows, train_path)
    write_jsonl(val_rows, val_path)

    full_stats = {
        **stats,
        "num_total_samples": len(rows),
        "num_train_samples": len(train_rows),
        "num_val_samples": len(val_rows),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(full_stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(full_stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
