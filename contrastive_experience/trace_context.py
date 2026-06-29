import contextvars
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from contrastive_experience.trace_abstraction import abstract_value
from contrastive_experience.trace_logger import TraceRecorder, preview_value


_CURRENT_SAMPLE = contextvars.ContextVar("contrastive_current_sample", default=None)
_CURRENT_CONFIG = contextvars.ContextVar("contrastive_current_config", default=None)


@dataclass
class TraceConfig:
    enabled: bool = False
    dataset: str = ""
    trace_dir: str = "traces"
    run_id: str = ""
    workflow_id: str = ""
    round_id: Optional[int] = None
    trace_full_io: bool = False
    trace_preview_chars: int = 512
    success_threshold: Optional[float] = None
    recorder: Optional[TraceRecorder] = None

    @classmethod
    def disabled(cls) -> "TraceConfig":
        return cls(enabled=False)

    @classmethod
    def create(
        cls,
        enabled: bool,
        dataset: str,
        trace_dir: str,
        run_id: str,
        round_id: Optional[int],
        workflow_id: Optional[str] = None,
        trace_full_io: bool = False,
        trace_preview_chars: int = 512,
        success_threshold: Optional[float] = None,
    ) -> "TraceConfig":
        if workflow_id:
            workflow = workflow_id
        elif round_id is not None:
            workflow = f"round_{round_id}"
        else:
            workflow = "workflow"
        path = Path(trace_dir) / dataset / run_id / f"workflow_{workflow}.jsonl"
        return cls(
            enabled=enabled,
            dataset=dataset,
            trace_dir=trace_dir,
            run_id=run_id,
            workflow_id=workflow,
            round_id=round_id,
            trace_full_io=trace_full_io,
            trace_preview_chars=trace_preview_chars,
            success_threshold=success_threshold,
            recorder=TraceRecorder(path) if enabled else None,
        )


@dataclass
class SampleTrace:
    config: TraceConfig
    sample_id: str
    raw_sample: Dict[str, Any] = field(default_factory=dict)
    operator_traces: list[Dict[str, Any]] = field(default_factory=list)

    def next_call_index(self) -> int:
        return len(self.operator_traces)

    def add_operator_trace(
        self,
        operator_type: str,
        input_value: Any,
        output_value: Any = None,
        mode: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> None:
        call_index = self.next_call_index()
        output_features = abstract_value(output_value, error_type=error_type)
        input_features = abstract_value(input_value)
        operator_id = f"{operator_type}#{call_index}"
        record = {
            "call_index": call_index,
            "operator_id": operator_id,
            "operator_type": operator_type,
            "input_preview": preview_value(
                input_value,
                max_chars=self.config.trace_preview_chars,
                full_io=self.config.trace_full_io,
            ),
            "output_preview": preview_value(
                output_value,
                max_chars=self.config.trace_preview_chars,
                full_io=self.config.trace_full_io,
            ),
            "input_schema_type": input_features["schema_type"],
            "input_output_style": input_features["output_style"],
            "schema_type": output_features["schema_type"],
            "output_style": output_features["output_style"],
            "answer_type": output_features["answer_type"],
            "output_length": output_features["output_length"],
            "parse_status": output_features["parse_status"],
            "error_type": error_type,
            "branch_id": None,
            "tool_id": None,
            "formatter_mode": mode,
            "timestamp": datetime.now().isoformat(),
        }
        self.operator_traces.append(record)

    def finalize(self, result: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "workflow_id": self.config.workflow_id,
            "round_id": self.config.round_id,
            "sample_id": self.sample_id,
            "dataset": self.config.dataset,
            "final_result": result.get("final_result"),
            "prediction": result.get("prediction"),
            "expected": result.get("expected"),
            "sample_score": result.get("sample_score"),
            "cost": result.get("cost"),
            "operator_traces": self.operator_traces,
        }
        if self.config.trace_full_io:
            record["sample"] = self.raw_sample
        return record


def make_sample_id(index: int, sample: Dict[str, Any]) -> str:
    for key in ("id", "question_id", "task_id", "problem_id"):
        if key in sample and sample[key] is not None:
            return str(sample[key])
    source = "|".join(str(sample.get(k, ""))[:80] for k in ("question", "problem", "prompt", "input"))
    digest = hashlib.sha1(source.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"sample_{index}_{digest}"


def set_trace_config(config: TraceConfig):
    return _CURRENT_CONFIG.set(config)


def reset_trace_config(token) -> None:
    _CURRENT_CONFIG.reset(token)


def get_trace_config() -> Optional[TraceConfig]:
    return _CURRENT_CONFIG.get()


def start_sample_trace(sample_id: str, raw_sample: Dict[str, Any]):
    config = get_trace_config()
    if not config or not config.enabled:
        return None
    trace = SampleTrace(config=config, sample_id=sample_id, raw_sample=raw_sample)
    return _CURRENT_SAMPLE.set(trace)


def finish_sample_trace(token, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if token is None:
        return None
    trace = _CURRENT_SAMPLE.get()
    _CURRENT_SAMPLE.reset(token)
    if trace and trace.config.recorder:
        record = trace.finalize(result)
        trace.config.recorder.append(record)
        return record
    return None


def record_operator_call(
    operator_type: str,
    input_value: Any,
    output_value: Any = None,
    mode: Optional[str] = None,
    error_type: Optional[str] = None,
) -> None:
    trace = _CURRENT_SAMPLE.get()
    if trace is None:
        return
    trace.add_operator_trace(operator_type, input_value, output_value, mode=mode, error_type=error_type)
