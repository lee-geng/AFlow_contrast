import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


POSITIVE_THRESHOLD = 0.03
NEGATIVE_THRESHOLD = 0.0
NO_EDIT_TOKEN = "[NO_EDIT]"


@dataclass
class TreeNode:
    node_id: str
    parent_id: Optional[str]
    workflow: str
    edit: str
    score_mean: Optional[float]
    score_std: Optional[float]
    task_family: str
    task_summary: str


@dataclass
class ContrastiveSample:
    task_family: str
    task_summary: str
    parent_id: str
    parent_workflow: str
    parent_workflow_summary: str
    positive_edit: str
    positive_child_workflow: str
    positive_delta_score: float
    negative_edit: str
    negative_child_workflow: str
    negative_delta_score: float
    sample_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_family": self.task_family,
            "task_summary": self.task_summary,
            "parent_id": self.parent_id,
            "parent_workflow": self.parent_workflow,
            "parent_workflow_summary": self.parent_workflow_summary,
            "positive_edit": self.positive_edit,
            "positive_child_workflow": self.positive_child_workflow,
            "positive_delta_score": self.positive_delta_score,
            "negative_edit": self.negative_edit,
            "negative_child_workflow": self.negative_child_workflow,
            "negative_delta_score": self.negative_delta_score,
            "sample_type": self.sample_type,
        }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _extract_workflow_summary(workflow: str, max_chars: int = 400) -> str:
    compact = re.sub(r"\s+", " ", workflow).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _node_from_dict(raw: Dict[str, Any], default_task_family: str = "unknown") -> TreeNode:
    node_id = _stringify(raw.get("id") or raw.get("node_id") or raw.get("round") or raw.get("child_id"))
    if not node_id:
        raise ValueError("Node missing id-like field")

    parent_raw = raw.get("parent_id")
    if parent_raw is None:
        parent_raw = raw.get("parent")
    if parent_raw is None:
        parent_raw = raw.get("father node")

    return TreeNode(
        node_id=node_id,
        parent_id=_stringify(parent_raw) or None,
        workflow=_stringify(raw.get("workflow") or raw.get("workflow_text") or raw.get("graph") or raw.get("child_workflow")),
        edit=_stringify(raw.get("edit") or raw.get("modification") or raw.get("operation") or raw.get("change")),
        score_mean=_safe_float(raw.get("score_mean") if raw.get("score_mean") is not None else raw.get("score")),
        score_std=_safe_float(raw.get("score_std") if raw.get("score_std") is not None else raw.get("std")),
        task_family=_stringify(raw.get("task_family") or raw.get("dataset") or raw.get("task") or default_task_family),
        task_summary=_stringify(raw.get("task_summary") or raw.get("task_description") or raw.get("task") or default_task_family),
    )


def _parse_tree_json(tree_path: Path, default_task_family: str) -> Dict[str, TreeNode]:
    payload = _read_json(tree_path)
    nodes: Dict[str, TreeNode] = {}

    if isinstance(payload, dict) and isinstance(payload.get("nodes"), list):
        for raw_node in payload["nodes"]:
            node = _node_from_dict(raw_node, default_task_family=default_task_family)
            nodes[node.node_id] = node

        edges = payload.get("edges") or []
        for edge in edges:
            parent_id = _stringify(edge.get("parent_id") or edge.get("source") or edge.get("from"))
            child_id = _stringify(edge.get("child_id") or edge.get("target") or edge.get("to"))
            if parent_id and child_id and child_id in nodes:
                nodes[child_id].parent_id = parent_id
                if not nodes[child_id].edit:
                    nodes[child_id].edit = _stringify(edge.get("edit") or edge.get("modification") or edge.get("operation"))

    elif isinstance(payload, list):
        for raw_node in payload:
            node = _node_from_dict(raw_node, default_task_family=default_task_family)
            nodes[node.node_id] = node
    else:
        raise ValueError(f"Unsupported tree JSON format: {tree_path}")

    return nodes


def _collect_round_workflows(workflows_dir: Path) -> Dict[str, str]:
    workflow_map: Dict[str, str] = {}
    for round_dir in workflows_dir.glob("round_*"):
        if not round_dir.is_dir():
            continue
        round_id = round_dir.name.split("_", 1)[-1]
        graph_file = round_dir / "graph.py"
        if graph_file.exists():
            workflow_map[round_id] = graph_file.read_text(encoding="utf-8")
    return workflow_map


def _collect_round_scores(workflows_dir: Path) -> Dict[str, float]:
    result_file = workflows_dir / "results.json"
    if not result_file.exists() or result_file.stat().st_size == 0:
        return {}

    payload = _read_json(result_file)
    if not isinstance(payload, list):
        return {}

    grouped: Dict[str, List[float]] = {}
    for item in payload:
        round_id = _stringify(item.get("round"))
        score = _safe_float(item.get("score"))
        if round_id and score is not None:
            grouped.setdefault(round_id, []).append(score)

    return {round_id: sum(scores) / len(scores) for round_id, scores in grouped.items() if scores}


def _parse_workspace_dir(input_path: Path, default_task_family: str) -> Dict[str, TreeNode]:
    workflows_dir = input_path / "workflows" if (input_path / "workflows").exists() else input_path
    if not workflows_dir.exists():
        raise FileNotFoundError(f"Cannot find workflows directory under: {input_path}")

    score_map = _collect_round_scores(workflows_dir)
    workflow_map = _collect_round_workflows(workflows_dir)

    nodes: Dict[str, TreeNode] = {}
    for round_id, workflow_text in workflow_map.items():
        nodes[round_id] = TreeNode(
            node_id=round_id,
            parent_id=None,
            workflow=workflow_text,
            edit="",
            score_mean=score_map.get(round_id),
            score_std=None,
            task_family=default_task_family,
            task_summary=default_task_family,
        )

    for exp_file in workflows_dir.glob("round_*/experience.json"):
        if exp_file.stat().st_size == 0:
            continue
        data = _read_json(exp_file)
        child_id = exp_file.parent.name.split("_", 1)[-1]
        child_node = nodes.setdefault(
            child_id,
            TreeNode(
                node_id=child_id,
                parent_id=None,
                workflow=workflow_map.get(child_id, ""),
                edit="",
                score_mean=score_map.get(child_id),
                score_std=None,
                task_family=default_task_family,
                task_summary=default_task_family,
            ),
        )
        child_node.parent_id = _stringify(data.get("father node")) or child_node.parent_id
        child_node.edit = _stringify(data.get("modification")) or child_node.edit
        child_node.score_mean = _safe_float(data.get("after")) if data.get("after") is not None else child_node.score_mean

        parent_id = _stringify(data.get("father node"))
        if parent_id and parent_id not in nodes:
            nodes[parent_id] = TreeNode(
                node_id=parent_id,
                parent_id=None,
                workflow=workflow_map.get(parent_id, ""),
                edit="",
                score_mean=_safe_float(data.get("before")) if data.get("before") is not None else score_map.get(parent_id),
                score_std=None,
                task_family=default_task_family,
                task_summary=default_task_family,
            )

    for log_file in workflows_dir.glob("round_*/log.json"):
        if log_file.stat().st_size == 0:
            continue
        payload = _read_json(log_file)
        entries: Iterable[Dict[str, Any]]
        if isinstance(payload, dict):
            entries = [payload]
        elif isinstance(payload, list):
            entries = [item for item in payload if isinstance(item, dict)]
        else:
            continue

        for item in entries:
            child_id = _stringify(item.get("child_id") or item.get("round") or item.get("node_id") or item.get("id"))
            if not child_id:
                continue
            child_node = nodes.setdefault(
                child_id,
                TreeNode(
                    node_id=child_id,
                    parent_id=None,
                    workflow=workflow_map.get(child_id, ""),
                    edit="",
                    score_mean=score_map.get(child_id),
                    score_std=None,
                    task_family=default_task_family,
                    task_summary=default_task_family,
                ),
            )
            parent_id = _stringify(item.get("parent_id") or item.get("father node") or item.get("parent") or item.get("source_round"))
            if parent_id:
                child_node.parent_id = parent_id
            child_node.edit = _stringify(item.get("edit") or item.get("modification") or item.get("operation")) or child_node.edit
            score_after = _safe_float(item.get("score_mean") if item.get("score_mean") is not None else item.get("after"))
            if score_after is not None:
                child_node.score_mean = score_after
            score_std = _safe_float(item.get("score_std"))
            if score_std is not None:
                child_node.score_std = score_std

            if parent_id and parent_id not in nodes:
                parent_score = _safe_float(item.get("before"))
                nodes[parent_id] = TreeNode(
                    node_id=parent_id,
                    parent_id=None,
                    workflow=workflow_map.get(parent_id, ""),
                    edit="",
                    score_mean=parent_score if parent_score is not None else score_map.get(parent_id),
                    score_std=None,
                    task_family=default_task_family,
                    task_summary=default_task_family,
                )

    return nodes


def load_tree_nodes(input_path: Path, task_family: Optional[str]) -> Dict[str, TreeNode]:
    default_task_family = task_family or input_path.stem
    if input_path.is_file():
        return _parse_tree_json(input_path, default_task_family=default_task_family)
    if input_path.is_dir():
        return _parse_workspace_dir(input_path, default_task_family=default_task_family)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _build_sibling_pairs(nodes: Dict[str, TreeNode]) -> Tuple[List[ContrastiveSample], set[str]]:
    children_by_parent: Dict[str, List[TreeNode]] = {}
    for node in nodes.values():
        if node.parent_id:
            children_by_parent.setdefault(node.parent_id, []).append(node)

    samples: List[ContrastiveSample] = []
    used_children: set[str] = set()

    for parent_id, children in children_by_parent.items():
        parent = nodes.get(parent_id)
        if parent is None or parent.score_mean is None:
            continue

        positives: List[Tuple[TreeNode, float]] = []
        negatives: List[Tuple[TreeNode, float]] = []

        for child in children:
            if child.score_mean is None:
                continue
            delta = child.score_mean - parent.score_mean
            if delta > POSITIVE_THRESHOLD:
                positives.append((child, delta))
            elif delta <= NEGATIVE_THRESHOLD:
                negatives.append((child, delta))

        if not positives or not negatives:
            continue

        positives.sort(key=lambda x: x[1], reverse=True)
        negatives.sort(key=lambda x: x[1])
        pair_count = min(len(positives), len(negatives))

        for idx in range(pair_count):
            pos_child, pos_delta = positives[idx]
            neg_child, neg_delta = negatives[idx]
            samples.append(
                ContrastiveSample(
                    task_family=pos_child.task_family or parent.task_family,
                    task_summary=pos_child.task_summary or parent.task_summary,
                    parent_id=parent_id,
                    parent_workflow=parent.workflow,
                    parent_workflow_summary=_extract_workflow_summary(parent.workflow),
                    positive_edit=pos_child.edit or "",
                    positive_child_workflow=pos_child.workflow,
                    positive_delta_score=pos_delta,
                    negative_edit=neg_child.edit or "",
                    negative_child_workflow=neg_child.workflow,
                    negative_delta_score=neg_delta,
                    sample_type="sibling_pair",
                )
            )
            used_children.add(pos_child.node_id)
            used_children.add(neg_child.node_id)

    return samples, used_children


def _build_fallback_pairs(nodes: Dict[str, TreeNode], used_children: set[str]) -> List[ContrastiveSample]:
    samples: List[ContrastiveSample] = []
    for child in nodes.values():
        if child.node_id in used_children or not child.parent_id:
            continue

        parent = nodes.get(child.parent_id)
        if parent is None or parent.score_mean is None or child.score_mean is None:
            continue

        delta = child.score_mean - parent.score_mean
        if not (delta > POSITIVE_THRESHOLD or delta <= NEGATIVE_THRESHOLD):
            continue

        if delta > POSITIVE_THRESHOLD:
            positive_edit = child.edit or ""
            positive_workflow = child.workflow
            positive_delta = delta
            negative_edit = NO_EDIT_TOKEN
            negative_workflow = parent.workflow
            negative_delta = 0.0
        else:
            positive_edit = NO_EDIT_TOKEN
            positive_workflow = parent.workflow
            positive_delta = 0.0
            negative_edit = child.edit or ""
            negative_workflow = child.workflow
            negative_delta = delta

        samples.append(
            ContrastiveSample(
                task_family=child.task_family or parent.task_family,
                task_summary=child.task_summary or parent.task_summary,
                parent_id=child.parent_id,
                parent_workflow=parent.workflow,
                parent_workflow_summary=_extract_workflow_summary(parent.workflow),
                positive_edit=positive_edit,
                positive_child_workflow=positive_workflow,
                positive_delta_score=positive_delta,
                negative_edit=negative_edit,
                negative_child_workflow=negative_workflow,
                negative_delta_score=negative_delta,
                sample_type="parent_child_fallback",
            )
        )

    return samples


def build_contrastive_dataset(nodes: Dict[str, TreeNode]) -> List[ContrastiveSample]:
    sibling_samples, used_children = _build_sibling_pairs(nodes)
    fallback_samples = _build_fallback_pairs(nodes, used_children)
    return sibling_samples + fallback_samples


def write_jsonl(samples: List[ContrastiveSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build contrastive dataset from AFlow search tree.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to AFlow tree JSON or logs/workflows dir")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--task_family", type=str, default=None, help="Optional task family override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    nodes = load_tree_nodes(input_path, task_family=args.task_family)
    samples = build_contrastive_dataset(nodes)
    write_jsonl(samples, output_path)

    summary = {
        "num_nodes": len(nodes),
        "num_samples": len(samples),
        "output_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
