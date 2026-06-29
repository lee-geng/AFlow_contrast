import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TraceMetadata:
    trace_id: str
    task_name: str
    sample_id: str
    workflow_id: str
    success: bool
    final_score: Optional[float] = None
    latency: Optional[float] = None
    token_cost: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorState:
    operator_name: str
    operator_input: Any = None
    operator_output: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    verifier_score: Optional[float] = None
    confidence: Optional[float] = None
    timestamp: Optional[str] = None
    latency: Optional[float] = None
    token_cost: Optional[float] = None


@dataclass
class NodeState:
    node_id: str
    operator: OperatorState
    upstream: List[str] = field(default_factory=list)
    downstream: List[str] = field(default_factory=list)


@dataclass
class TraceIR:
    metadata: TraceMetadata
    input: Any = None
    final_output: Any = None
    nodes: List[NodeState] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceIR":
        metadata = TraceMetadata(**data["metadata"])
        nodes = []
        for node in data.get("nodes", []):
            operator = OperatorState(**node["operator"])
            nodes.append(
                NodeState(
                    node_id=node["node_id"],
                    operator=operator,
                    upstream=node.get("upstream", []),
                    downstream=node.get("downstream", []),
                )
            )
        return cls(metadata=metadata, input=data.get("input"), final_output=data.get("final_output"), nodes=nodes)

    def save_json(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(self.to_dict(), fout, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "TraceIR":
        with Path(path).open("r", encoding="utf-8") as fin:
            return cls.from_dict(json.load(fin))


def contrastive_record_to_trace_ir(record: Dict[str, Any]) -> TraceIR:
    metadata = TraceMetadata(
        trace_id=str(record.get("sample_id", "")),
        task_name=str(record.get("dataset", "")),
        sample_id=str(record.get("sample_id", "")),
        workflow_id=str(record.get("workflow_id", "")),
        success=record.get("final_result") == "success",
        final_score=record.get("sample_score"),
        token_cost=record.get("cost"),
    )
    nodes = []
    for op in record.get("operator_traces", []):
        operator = OperatorState(
            operator_name=op.get("operator_type") or op.get("operator_id") or "operator",
            operator_input=op.get("input_preview"),
            operator_output=op.get("output_preview"),
            metadata={
                "operator_id": op.get("operator_id"),
                "schema_type": op.get("schema_type"),
                "output_style": op.get("output_style"),
                "parse_status": op.get("parse_status"),
                "answer_type": op.get("answer_type"),
            },
            error_message=op.get("error_type"),
            timestamp=op.get("timestamp"),
        )
        nodes.append(NodeState(node_id=str(op.get("operator_id") or len(nodes)), operator=operator))
    return TraceIR(metadata=metadata, input=record.get("sample"), final_output=record.get("prediction"), nodes=nodes)


def aflow_log_record_to_trace_ir(record: Dict[str, Any], task_name: str = "", workflow_id: str = "") -> TraceIR:
    metadata = TraceMetadata(
        trace_id=str(record.get("question", ""))[:64],
        task_name=task_name,
        sample_id=str(record.get("question", ""))[:64],
        workflow_id=workflow_id,
        success=False,
        final_score=None,
    )
    operator = OperatorState(
        operator_name="workflow_output",
        operator_input=record.get("question"),
        operator_output=record.get("model_output"),
        metadata={"extracted_output": record.get("extracted_output")},
    )
    return TraceIR(
        metadata=metadata,
        input=record.get("question"),
        final_output=record.get("model_output"),
        nodes=[NodeState(node_id="workflow_output", operator=operator)],
    )

