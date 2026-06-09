import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class WorkflowNode:
    node_id: str
    parent_id: Optional[str]
    workflow: str
    edit_text: str
    score_mean: Optional[float]
    score_std: Optional[float]
    task_family: str
    task_summary: str
    log_feedback: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowEdge:
    parent_id: str
    child_id: str
    edit_text: str


@dataclass
class AFlowTreeBundle:
    nodes: Dict[str, WorkflowNode]
    edges: List[WorkflowEdge]


class AFlowTreeAdapter:
    """Adapter layer that normalizes heterogeneous AFlow tree/log formats."""

    def __init__(self, task_family: Optional[str] = None):
        self.task_family = task_family or "unknown"

    def load(self, input_path: str) -> AFlowTreeBundle:
        path = Path(input_path)
        if path.is_file():
            return self._load_json_tree(path)
        if path.is_dir():
            return self._load_workspace_tree(path)
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    def _load_json_tree(self, path: Path) -> AFlowTreeBundle:
        payload = self._read_json(path)
        nodes: Dict[str, WorkflowNode] = {}
        edges: List[WorkflowEdge] = []

        if isinstance(payload, dict) and isinstance(payload.get("nodes"), list):
            for raw in payload["nodes"]:
                node = self._node_from_raw(raw)
                nodes[node.node_id] = node

            for edge in payload.get("edges", []):
                parent_id = self._to_str(edge.get("parent_id") or edge.get("source") or edge.get("from"))
                child_id = self._to_str(edge.get("child_id") or edge.get("target") or edge.get("to"))
                if not parent_id or not child_id:
                    continue
                edges.append(
                    WorkflowEdge(
                        parent_id=parent_id,
                        child_id=child_id,
                        edit_text=self._to_str(edge.get("edit") or edge.get("modification") or ""),
                    )
                )
                if child_id in nodes:
                    nodes[child_id].parent_id = parent_id
                    if not nodes[child_id].edit_text:
                        nodes[child_id].edit_text = self._to_str(edge.get("edit") or edge.get("modification") or "")
        elif isinstance(payload, list):
            for raw in payload:
                if not isinstance(raw, dict):
                    continue
                node = self._node_from_raw(raw)
                nodes[node.node_id] = node
        else:
            raise ValueError(f"Unsupported JSON tree format: {path}")

        self._derive_edges(nodes, edges)
        return AFlowTreeBundle(nodes=nodes, edges=edges)

    def _load_workspace_tree(self, path: Path) -> AFlowTreeBundle:
        workflows_dir = path / "workflows" if (path / "workflows").exists() else path
        if not workflows_dir.exists():
            raise FileNotFoundError(f"Cannot locate workflows directory in: {path}")

        scores = self._load_scores(workflows_dir)
        workflows = self._load_workflows(workflows_dir)
        logs = self._load_logs(workflows_dir)

        nodes: Dict[str, WorkflowNode] = {}
        edges: List[WorkflowEdge] = []

        for node_id, wf in workflows.items():
            nodes[node_id] = WorkflowNode(
                node_id=node_id,
                parent_id=None,
                workflow=wf,
                edit_text="",
                score_mean=scores.get(node_id),
                score_std=None,
                task_family=self.task_family,
                task_summary=self.task_family,
                log_feedback=logs.get(node_id, {}),
            )

        # Primary source for parent->child edges.
        for exp_file in workflows_dir.glob("round_*/experience.json"):
            if exp_file.stat().st_size == 0:
                continue
            item = self._read_json(exp_file)
            if not isinstance(item, dict):
                continue

            child_id = exp_file.parent.name.split("_", 1)[-1]
            parent_id = self._to_str(item.get("father node"))
            edit_text = self._to_str(item.get("modification"))
            before = self._to_float(item.get("before"))
            after = self._to_float(item.get("after"))

            child = nodes.setdefault(
                child_id,
                WorkflowNode(child_id, None, workflows.get(child_id, ""), edit_text, scores.get(child_id), None, self.task_family, self.task_family, logs.get(child_id, {})),
            )
            child.parent_id = parent_id or child.parent_id
            child.edit_text = edit_text or child.edit_text
            if after is not None:
                child.score_mean = after

            if parent_id:
                parent = nodes.setdefault(
                    parent_id,
                    WorkflowNode(parent_id, None, workflows.get(parent_id, ""), "", scores.get(parent_id), None, self.task_family, self.task_family, logs.get(parent_id, {})),
                )
                if before is not None and parent.score_mean is None:
                    parent.score_mean = before
                edges.append(WorkflowEdge(parent_id=parent_id, child_id=child_id, edit_text=edit_text))

        # Secondary source from logs if any structured edge fields exist.
        for round_id, feedback in logs.items():
            parent_id = self._to_str(feedback.get("parent_id") or feedback.get("father node") or feedback.get("source_round"))
            edit_text = self._to_str(feedback.get("edit") or feedback.get("modification") or "")
            if parent_id and round_id in nodes:
                nodes[round_id].parent_id = parent_id
                nodes[round_id].edit_text = edit_text or nodes[round_id].edit_text
                edges.append(WorkflowEdge(parent_id=parent_id, child_id=round_id, edit_text=nodes[round_id].edit_text))

        self._derive_edges(nodes, edges)
        return AFlowTreeBundle(nodes=nodes, edges=edges)

    def _derive_edges(self, nodes: Dict[str, WorkflowNode], edges: List[WorkflowEdge]) -> None:
        seen = {(e.parent_id, e.child_id) for e in edges}
        for node in nodes.values():
            if node.parent_id and (node.parent_id, node.node_id) not in seen:
                edges.append(WorkflowEdge(parent_id=node.parent_id, child_id=node.node_id, edit_text=node.edit_text))

    def _node_from_raw(self, raw: Dict[str, Any]) -> WorkflowNode:
        node_id = self._to_str(raw.get("id") or raw.get("node_id") or raw.get("round") or raw.get("child_id"))
        if not node_id:
            raise ValueError("Node missing id field")

        return WorkflowNode(
            node_id=node_id,
            parent_id=self._to_str(raw.get("parent_id") or raw.get("parent") or raw.get("father node")) or None,
            workflow=self._to_str(raw.get("workflow") or raw.get("child_workflow") or raw.get("graph") or ""),
            edit_text=self._to_str(raw.get("edit") or raw.get("modification") or raw.get("operation") or ""),
            score_mean=self._to_float(raw.get("score_mean") if raw.get("score_mean") is not None else raw.get("score")),
            score_std=self._to_float(raw.get("score_std") if raw.get("score_std") is not None else raw.get("std")),
            task_family=self._to_str(raw.get("task_family") or raw.get("dataset") or self.task_family),
            task_summary=self._to_str(raw.get("task_summary") or raw.get("task") or raw.get("task_description") or self.task_family),
            log_feedback=raw.get("execution_feedback") if isinstance(raw.get("execution_feedback"), dict) else {},
        )

    def _load_scores(self, workflows_dir: Path) -> Dict[str, float]:
        results_file = workflows_dir / "results.json"
        if not results_file.exists() or results_file.stat().st_size == 0:
            return {}
        payload = self._read_json(results_file)
        if not isinstance(payload, list):
            return {}

        grouped: Dict[str, List[float]] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            round_id = self._to_str(row.get("round"))
            score = self._to_float(row.get("score"))
            if round_id and score is not None:
                grouped.setdefault(round_id, []).append(score)

        return {k: sum(v) / len(v) for k, v in grouped.items() if v}

    def _load_workflows(self, workflows_dir: Path) -> Dict[str, str]:
        workflow_map: Dict[str, str] = {}
        for round_dir in workflows_dir.glob("round_*"):
            if not round_dir.is_dir():
                continue
            round_id = round_dir.name.split("_", 1)[-1]
            graph_file = round_dir / "graph.py"
            if graph_file.exists():
                workflow_map[round_id] = graph_file.read_text(encoding="utf-8")
        return workflow_map

    def _load_logs(self, workflows_dir: Path) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for round_dir in workflows_dir.glob("round_*"):
            round_id = round_dir.name.split("_", 1)[-1]
            log_file = round_dir / "log.json"
            if not log_file.exists() or log_file.stat().st_size == 0:
                continue
            payload = self._read_json(log_file)
            if isinstance(payload, list):
                fail_summary = self._summarize_failures(payload)
                trace_summary = self._summarize_trace(payload)
                out[round_id] = {
                    "failure_summary": fail_summary,
                    "trace_summary": trace_summary,
                    "num_failures": len(payload),
                }
            elif isinstance(payload, dict):
                out[round_id] = {
                    "failure_summary": self._to_str(payload.get("failure_summary") or payload.get("model_output") or ""),
                    "trace_summary": self._to_str(payload.get("trace_summary") or payload.get("question") or ""),
                    **payload,
                }
        return out

    def _summarize_failures(self, rows: Iterable[Dict[str, Any]]) -> str:
        snippets: List[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            question = self._to_str(row.get("question"))
            model_output = self._to_str(row.get("model_output"))
            extracted = self._to_str(row.get("extracted_output"))
            if question or model_output or extracted:
                snippets.append(f"Q:{question[:80]} | pred:{model_output[:80]} | extracted:{extracted[:80]}")
            if len(snippets) >= 4:
                break
        return " ; ".join(snippets)

    def _summarize_trace(self, rows: Iterable[Dict[str, Any]]) -> str:
        traces: List[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = self._to_str(row.get("extract_answer_code"))
            if code and code != "None":
                traces.append(code[:120])
            if len(traces) >= 3:
                break
        return " | ".join(traces)

    def _read_json(self, path: Path) -> Any:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def _to_str(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
