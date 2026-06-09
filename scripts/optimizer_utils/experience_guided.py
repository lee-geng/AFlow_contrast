import asyncio
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from analysis.blame_attributor import WorkflowBlockView
from scripts.logs import logger

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback when dependency is unavailable
    OpenAI = None


EXPERIENCE_TAXONOMY = [
    "planning_error",
    "reasoning_error",
    "verification_missing",
    "weak_revision",
    "tool_error",
    "premature_stop",
    "evidence_missing",
    "over_complexity",
    "regression",
    "unknown",
]


def _compact(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _mean(rows: Sequence[float]) -> float:
    if not rows:
        return 0.0
    return sum(rows) / max(len(rows), 1)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _token_set(text: str) -> set[str]:
    return {item for item in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(item) > 2}


class TaskEmbedder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        base_url: str = "http://localhost:8090/v1",
        api_key: str = "EMPTY",
        enabled: bool = True,
        dim: int = 64,
    ):
        self.model_name = model_name
        self.base_url = self._normalize_base_url(base_url)
        self.api_key = api_key
        self.enabled = enabled
        self.dim = max(16, int(dim))
        self._client = None
        self._warned_remote_failure = False
        self._cache: Dict[str, List[float]] = {}

        if self.enabled and OpenAI is not None:
            try:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception:
                self._client = None

    def embed_text(self, text: str) -> List[float]:
        normalized = _compact(text, limit=4000)
        if not normalized:
            return [0.0] * self.dim
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached

        if self.enabled and self._client is not None:
            try:
                response = self._client.embeddings.create(model=self.model_name, input=[normalized])
                vector = list(response.data[0].embedding)
                self._cache[normalized] = vector
                return vector
            except Exception as exc:
                if not self._warned_remote_failure:
                    logger.warning(f"Task embedding remote call failed, fallback to local hash embedding: {exc}")
                    self._warned_remote_failure = True

        vector = self._fallback_embedding(normalized)
        self._cache[normalized] = vector
        return vector

    def cluster_id(self, embedding: Sequence[float]) -> str:
        if not embedding:
            return "emb_empty"
        width = min(8, len(embedding))
        bits = "".join("1" if float(v) >= 0 else "0" for v in embedding[:width])
        return f"emb_{bits}"

    def similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        return _cosine_similarity(a, b)

    def _fallback_embedding(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return [0.0] * self.dim
        vec = [0.0] * self.dim
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(self.dim):
                vec[i] += 1.0 if digest[i % len(digest)] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _normalize_base_url(self, base_url: str) -> str:
        value = str(base_url or "").rstrip("/")
        if value.endswith("/embeddings"):
            value = value[: -len("/embeddings")]
        return value or "http://localhost:8090/v1"


@dataclass
class ExecutionTrace:
    task_id: str
    workflow_id: str
    workflow_version: str
    task_pattern: str
    node_execution_sequence: List[str]
    node_summaries: List[Dict[str, str]]
    final_output: str
    ground_truth: str
    score: float
    outcome: str
    token_cost: float
    runtime_seconds: float
    task_text: str = ""
    task_embedding: List[float] = field(default_factory=list)
    task_cluster: str = ""
    evaluator_feedback: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributionEvent:
    event_id: str
    task_id: str
    workflow_id: str
    workflow_version: str
    task_pattern: str
    outcome: str
    attributed_nodes: List[str]
    task_text: str = ""
    task_embedding: List[float] = field(default_factory=list)
    task_cluster: str = ""
    failed_node: str = ""
    contributing_nodes: List[str] = field(default_factory=list)
    failure_type: str = "unknown"
    root_cause: str = ""
    patch_target: str = ""
    patch_suggestion: str = ""
    successful_subworkflow: List[str] = field(default_factory=list)
    reuse_suggestion: str = ""
    confidence: float = 0.5
    token_cost: float = 0.0
    runtime_seconds: float = 0.0
    stage: str = "full"
    trace_digest: str = ""


@dataclass
class WorkflowMemory:
    memory_id: str
    task_pattern: str
    task_text_prototype: str = ""
    task_embedding: List[float] = field(default_factory=list)
    task_cluster: str = ""
    failure_pattern: str = ""
    success_pattern: str = ""
    failed_node: str = ""
    patch_target: str = ""
    successful_subworkflow: List[str] = field(default_factory=list)
    patch_rule: str = ""
    applicability: str = ""
    anti_condition: str = ""
    support: Dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0
    risk: str = "medium"
    evidence_task_ids: List[str] = field(default_factory=list)
    created_from_workflow_ids: List[str] = field(default_factory=list)
    status: str = "candidate"


@dataclass
class OptimizationStateQuery:
    parent_round: int = 0
    parent_workflow_id: str = ""
    score: float = 0.0
    failure_types: List[str] = field(default_factory=list)
    failed_nodes: List[str] = field(default_factory=list)
    patch_targets: List[str] = field(default_factory=list)
    mismatch_info: str = ""
    extract_result_summary: str = ""
    graph_summary: str = ""
    prompt_summary: str = ""
    recent_failed_modifications: List[str] = field(default_factory=list)
    recent_successful_modifications: List[str] = field(default_factory=list)
    modification_intent: str = ""
    task_query_text: str = ""


@dataclass
class MemoryGuidance:
    focus_nodes: List[str] = field(default_factory=list)
    preserve_nodes: List[str] = field(default_factory=list)
    avoid_nodes_or_patches: List[str] = field(default_factory=list)
    recommended_patch_rules: List[str] = field(default_factory=list)
    candidate_patch_rules: List[str] = field(default_factory=list)
    evidence_summary: List[str] = field(default_factory=list)
    confidence: float = 0.0
    validated_memories: List[WorkflowMemory] = field(default_factory=list)
    candidate_memories: List[WorkflowMemory] = field(default_factory=list)
    rejected_memories: List[WorkflowMemory] = field(default_factory=list)


@dataclass
class ExperienceGuidedConfig:
    enabled: bool = False
    store_path: str = ""
    min_support: int = 3
    min_confidence: float = 0.6
    compression_interval: int = 10
    attribution_enabled: bool = True
    attribution_mode: str = "heuristic"
    taxonomy: List[str] = field(default_factory=lambda: list(EXPERIENCE_TAXONOMY))
    adaptive_validation_enabled: bool = False
    use_staged_validation: bool = True
    targeted_failure_size: int = 8
    success_guard_size: int = 8
    anchor_size: int = 16
    full_validation_top_k: int = 3
    beta: float = 0.5
    gamma: float = 0.5
    lambda_cost: float = 0.1
    task_embedding_enabled: bool = True
    task_embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    task_embedding_base_url: str = "http://localhost:8090/v1"
    task_embedding_api_key: str = "EMPTY"
    task_embedding_dim: int = 64
    task_similarity_threshold: float = 0.55
    enable_prompt_budget_diagnostics: bool = True
    use_failure_cards: bool = True
    use_optimization_state_memory_query: bool = True
    use_memory_guided_parent_selection: bool = False
    use_memory_guided_patch_scope: bool = True
    use_memory_as_experience_replacement: bool = True
    use_graph_prompt_summarization: bool = True


@dataclass
class ValidationPlan:
    targeted_failure_set: List[int]
    success_guard_set: List[int]
    anchor_set: List[int]
    full_set: List[int]
    targeted_failure_tasks: List[str] = field(default_factory=list)
    success_guard_tasks: List[str] = field(default_factory=list)


@dataclass
class StageEvaluationResult:
    stage_name: str
    indices: List[int]
    task_ids: List[str]
    score: float
    avg_cost: float
    total_cost: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class CandidateEvaluationSummary:
    comparable_score: float
    selection_score: float
    score_source: str
    targeted_gain: float
    regression_loss: float
    anchor_score: float
    full_score: Optional[float]
    stages_run: List[str]
    stage_results: Dict[str, StageEvaluationResult]


class AttributionAnalyzer:
    """Heuristic-first attribution that can be swapped with an LLM implementation later."""

    def __init__(self, taxonomy: Optional[List[str]] = None):
        self.taxonomy = taxonomy or list(EXPERIENCE_TAXONOMY)
        self.block_view = WorkflowBlockView()

    def analyze(
        self,
        workflow_source: str,
        trace: ExecutionTrace,
        evaluator_feedback: Optional[Dict[str, Any]] = None,
        stage: str = "full",
    ) -> AttributionEvent:
        feedback = evaluator_feedback or trace.evaluator_feedback or {}
        nodes = trace.node_execution_sequence or self.block_view.extract(workflow_source)
        failure_type = self._infer_failure_type(trace, feedback)
        attributed_nodes = nodes[-2:] if len(nodes) >= 2 else nodes
        failed_node = self._infer_failed_node(nodes, trace, feedback, failure_type)
        contributing_nodes = nodes[-2:] if trace.outcome == "success" else []
        successful_subworkflow = contributing_nodes[:] if trace.outcome == "success" else []
        patch_target = failed_node if trace.outcome == "failure" else ""
        patch_suggestion = self._patch_suggestion(failure_type, patch_target)
        reuse_suggestion = (
            f"Preserve {' -> '.join(successful_subworkflow)} for {trace.task_pattern} tasks."
            if successful_subworkflow
            else ""
        )
        root_cause = self._root_cause(trace, failure_type, patch_target)
        confidence = self._confidence(trace, failure_type)

        return AttributionEvent(
            event_id=f"ae_{_hash_text(trace.workflow_id + trace.task_id + stage + trace.final_output[:80])}",
            task_id=trace.task_id,
            workflow_id=trace.workflow_id,
            workflow_version=trace.workflow_version,
            task_pattern=trace.task_pattern,
            task_text=trace.task_text,
            task_embedding=list(trace.task_embedding),
            task_cluster=trace.task_cluster,
            outcome=trace.outcome,
            attributed_nodes=attributed_nodes,
            failed_node=failed_node,
            contributing_nodes=contributing_nodes,
            failure_type=failure_type,
            root_cause=root_cause,
            patch_target=patch_target,
            patch_suggestion=patch_suggestion,
            successful_subworkflow=successful_subworkflow,
            reuse_suggestion=reuse_suggestion,
            confidence=confidence,
            token_cost=trace.token_cost,
            runtime_seconds=trace.runtime_seconds,
            stage=stage,
            trace_digest=_hash_text(json.dumps(asdict(trace), ensure_ascii=False)),
        )

    def _infer_failed_node(
        self,
        nodes: List[str],
        trace: ExecutionTrace,
        evaluator_feedback: Dict[str, Any],
        failure_type: str,
    ) -> str:
        if not nodes:
            return "b0"
        joined = " ".join(
            [
                trace.final_output.lower(),
                str(evaluator_feedback.get("expected_output") or "").lower(),
                str(evaluator_feedback.get("question") or "").lower(),
            ]
        )
        if failure_type in {"verification_missing", "premature_stop"}:
            return nodes[-1]
        if failure_type in {"tool_error", "evidence_missing"}:
            for node in nodes:
                if any(k in node.lower() for k in ["tool", "retrieve", "search", "program", "test"]):
                    return node
        if any(k in joined for k in ["review", "refine", "verify"]):
            for node in reversed(nodes):
                if any(k in node.lower() for k in ["review", "verify", "custom"]):
                    return node
        return nodes[min(len(nodes) - 1, 0)]

    def _infer_failure_type(self, trace: ExecutionTrace, evaluator_feedback: Dict[str, Any]) -> str:
        if trace.outcome == "success":
            return "unknown"

        final_output = (trace.final_output or "").lower()
        expected = str(evaluator_feedback.get("expected_output") or trace.ground_truth or "").lower()
        question = str(evaluator_feedback.get("question") or "").lower()
        merged = " ".join([final_output, expected, question])

        if any(k in merged for k in ["traceback", "exception", "runtime error", "api", "tool failed"]):
            return "tool_error"
        if any(k in merged for k in ["review", "refine", "revised version", "however", "improve readability"]):
            return "weak_revision"
        if any(k in merged for k in ["unsupported", "evidence", "justify", "prove", "verify"]) and "\\boxed" not in final_output:
            return "verification_missing"
        if any(k in merged for k in ["document", "context missing", "evidence", "supporting facts"]):
            return "evidence_missing"
        if any(k in merged for k in ["therefore", "thus", "boxed"]) and trace.score <= 0:
            return "reasoning_error"
        if final_output.count("step") >= 4 and trace.score <= 0:
            return "over_complexity"
        if len(final_output) < 16 or final_output.endswith("..."):
            return "premature_stop"
        if any(k in question for k in ["ways", "probability", "choose", "count"]) and trace.score <= 0:
            return "planning_error"
        return "unknown"

    def _patch_suggestion(self, failure_type: str, patch_target: str) -> str:
        target = patch_target or "current workflow"
        mapping = {
            "planning_error": f"Add a lightweight decomposition step before {target}.",
            "reasoning_error": f"Strengthen intermediate reasoning checks around {target}.",
            "verification_missing": f"Insert explicit answer verification after {target}.",
            "weak_revision": f"Reduce blind rewriting and make {target} revise against ground-truth format.",
            "tool_error": f"Add fallback handling or simpler operator usage near {target}.",
            "premature_stop": f"Ensure {target} produces a final boxed answer before returning.",
            "evidence_missing": f"Insert evidence retrieval or citation summarization before {target}.",
            "over_complexity": f"Simplify the subworkflow around {target} and reduce redundant calls.",
            "regression": f"Revert or isolate the recent patch around {target}.",
        }
        return mapping.get(failure_type, f"Apply a minimal local fix around {target}.")

    def _root_cause(self, trace: ExecutionTrace, failure_type: str, patch_target: str) -> str:
        if trace.outcome == "success":
            return f"Workflow succeeded on {trace.task_pattern} tasks using {' -> '.join(trace.node_execution_sequence[-2:])}."
        return f"{failure_type} concentrated near {patch_target or 'workflow tail'} for {trace.task_pattern} task {trace.task_id}."

    def _confidence(self, trace: ExecutionTrace, failure_type: str) -> float:
        base = 0.72 if failure_type != "unknown" else 0.48
        if trace.outcome == "success":
            base = 0.78
        if trace.runtime_seconds > 0 and trace.runtime_seconds < 2:
            base += 0.04
        if trace.token_cost > 0:
            base += min(trace.token_cost, 0.2)
        return max(0.35, min(0.98, base))


class ExperienceCompressor:
    def __init__(self, min_support: int = 3, min_confidence: float = 0.6):
        self.min_support = max(1, min_support)
        self.min_confidence = max(0.0, min(1.0, min_confidence))

    def compress(self, events: Sequence[AttributionEvent]) -> List[WorkflowMemory]:
        grouped: Dict[Tuple[str, str, str, str], List[AttributionEvent]] = defaultdict(list)
        for event in events:
            key = (
                event.task_cluster or event.task_pattern,
                event.failure_type if event.outcome == "failure" else "",
                event.failed_node if event.outcome == "failure" else "",
                "|".join(event.successful_subworkflow[:2]) if event.outcome == "success" else "",
            )
            grouped[key].append(event)

        memories: List[WorkflowMemory] = []
        for key, members in grouped.items():
            task_pattern, failure_pattern, failed_node, success_key = key
            success_events = [m for m in members if m.outcome == "success"]
            failure_events = [m for m in members if m.outcome == "failure"]
            support = {
                "failure_cases": len(failure_events),
                "success_cases": len(success_events),
                "patched_successes": len(
                    [m for m in success_events if m.patch_suggestion or m.reuse_suggestion]
                ),
            }
            support_count = support["failure_cases"] + support["success_cases"]
            if support_count < self.min_support:
                continue

            confidence = min(
                0.98,
                0.35
                + 0.08 * min(support["failure_cases"], 4)
                + 0.08 * min(support["success_cases"], 4)
                + 0.18 * _mean([m.confidence for m in members]),
            )
            if confidence < self.min_confidence:
                continue

            patch_rule = self._patch_rule(members)
            successful_subworkflow = success_key.split("|") if success_key else []
            status = self._status(failure_events, success_events, confidence)
            prototype_event = sorted(members, key=lambda x: x.confidence, reverse=True)[0]
            task_embedding = self._mean_embedding([m.task_embedding for m in members if m.task_embedding])
            memories.append(
                WorkflowMemory(
                    memory_id=f"wm_{_hash_text('|'.join(key) + patch_rule)}",
                    task_pattern=task_pattern,
                    task_text_prototype=prototype_event.task_text,
                    task_embedding=task_embedding,
                    task_cluster=prototype_event.task_cluster or task_pattern,
                    failure_pattern=failure_pattern,
                    success_pattern=success_key,
                    failed_node=failed_node,
                    patch_target=prototype_event.patch_target,
                    successful_subworkflow=successful_subworkflow,
                    patch_rule=patch_rule,
                    applicability=f"{task_pattern}:{failure_pattern or success_key or 'generic'}",
                    anti_condition=self._anti_condition(failure_events),
                    support=support,
                    confidence=confidence,
                    risk=self._risk(failure_events, success_events),
                    evidence_task_ids=sorted({m.task_id for m in members})[:20],
                    created_from_workflow_ids=sorted({m.workflow_id for m in members})[:20],
                    status=status,
                )
            )
        return self._dedupe(memories)

    def _patch_rule(self, members: Sequence[AttributionEvent]) -> str:
        if not members:
            return "No-op"
        suggestions = [m.patch_suggestion for m in members if m.patch_suggestion]
        if suggestions:
            counts = defaultdict(int)
            for item in suggestions:
                counts[item] += 1
            return sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)[0][0]
        reuse = [m.reuse_suggestion for m in members if m.reuse_suggestion]
        return reuse[0] if reuse else "Preserve the strongest subworkflow and patch the weakest node."

    def _anti_condition(self, failures: Sequence[AttributionEvent]) -> str:
        if not failures:
            return ""
        if any(f.failure_type == "over_complexity" for f in failures):
            return "Avoid adding more than one new expensive node."
        if any(f.failure_type == "weak_revision" for f in failures):
            return "Do not insert redundant revise-only loops."
        return ""

    def _risk(self, failures: Sequence[AttributionEvent], successes: Sequence[AttributionEvent]) -> str:
        if len(failures) > len(successes) * 2:
            return "high"
        if failures and successes:
            return "medium"
        return "low"

    def _status(
        self,
        failures: Sequence[AttributionEvent],
        successes: Sequence[AttributionEvent],
        confidence: float,
    ) -> str:
        if successes and confidence >= 0.7:
            return "validated"
        if failures and not successes and confidence >= 0.75:
            return "rejected"
        return "candidate"

    def _dedupe(self, memories: Sequence[WorkflowMemory]) -> List[WorkflowMemory]:
        keep: Dict[Tuple[str, str, str, str], WorkflowMemory] = {}
        for memory in memories:
            key = (
                memory.task_pattern,
                memory.failure_pattern,
                memory.failed_node,
                "|".join(memory.successful_subworkflow),
            )
            prev = keep.get(key)
            if prev is None or memory.confidence > prev.confidence:
                keep[key] = memory
        return list(keep.values())

    def _mean_embedding(self, rows: Sequence[Sequence[float]]) -> List[float]:
        rows = [list(r) for r in rows if r]
        if not rows:
            return []
        width = len(rows[0])
        vec = [0.0] * width
        for row in rows:
            if len(row) != width:
                continue
            for idx, value in enumerate(row):
                vec[idx] += float(value)
        denom = float(max(1, len(rows)))
        return [value / denom for value in vec]


class MemoryStore:
    def __init__(self, store_path: str):
        self.root = Path(store_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "attribution_events.jsonl"
        self.traces_path = self.root / "execution_traces.jsonl"
        self.memories_path = self.root / "workflow_memories.jsonl"
        self.metrics_path = self.root / "experience_metrics.jsonl"

    def add_trace(self, trace: ExecutionTrace) -> None:
        self._append_jsonl(self.traces_path, asdict(trace))

    def add_event(self, event: AttributionEvent) -> None:
        self._append_jsonl(self.events_path, asdict(event))

    def load_events(self) -> List[AttributionEvent]:
        return [AttributionEvent(**row) for row in self._read_jsonl(self.events_path)]

    def load_traces(self) -> List[ExecutionTrace]:
        return [ExecutionTrace(**row) for row in self._read_jsonl(self.traces_path)]

    def load_memories(self) -> List[WorkflowMemory]:
        return [WorkflowMemory(**row) for row in self._read_jsonl(self.memories_path)]

    def save_memories(self, memories: Sequence[WorkflowMemory]) -> None:
        payload = [asdict(m) for m in memories]
        self.memories_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in payload) + ("\n" if payload else ""),
            encoding="utf-8",
        )

    def compress(self, compressor: ExperienceCompressor) -> List[WorkflowMemory]:
        events = self.load_events()
        memories = compressor.compress(events)
        self.save_memories(memories)
        return memories

    def query(
        self,
        task_pattern: str = "",
        task_embedding: Optional[Sequence[float]] = None,
        failure_type: str = "",
        failed_node: str = "",
        optimization_state: Optional[OptimizationStateQuery] = None,
        statuses: Optional[Iterable[str]] = None,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[WorkflowMemory]:
        statuses = set(statuses or ["validated", "candidate", "rejected"])
        scored: List[Tuple[float, WorkflowMemory]] = []
        for memory in self.load_memories():
            if memory.status not in statuses:
                continue
            score = 0.0
            if task_pattern and memory.task_pattern == task_pattern:
                score += 1.0
            if task_embedding and memory.task_embedding:
                sim = _cosine_similarity(task_embedding, memory.task_embedding)
                if sim >= similarity_threshold:
                    score += sim
            if failure_type and memory.failure_pattern == failure_type:
                score += 1.0
            if failed_node and memory.failed_node == failed_node:
                score += 0.7
            if optimization_state is not None:
                score += self._score_state_match(memory, optimization_state)
            if not any([task_pattern, task_embedding, failure_type, failed_node]):
                score += 0.2
            score += memory.confidence
            if score > 0:
                scored.append((score, memory))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def _score_state_match(self, memory: WorkflowMemory, query: OptimizationStateQuery) -> float:
        score = 0.0
        if memory.failure_pattern and memory.failure_pattern in set(query.failure_types):
            score += 1.6
        if memory.failed_node and memory.failed_node in set(query.failed_nodes):
            score += 1.2
        if memory.patch_target and memory.patch_target in set(query.patch_targets):
            score += 1.0
        if query.parent_workflow_id and query.parent_workflow_id in set(memory.created_from_workflow_ids):
            score += 0.4
        if memory.successful_subworkflow and set(memory.successful_subworkflow) & _token_set(query.graph_summary):
            score += 0.6
        overlap = _token_set(memory.patch_rule) & (
            _token_set(query.mismatch_info)
            | _token_set(query.extract_result_summary)
            | _token_set(query.modification_intent)
            | _token_set(query.graph_summary)
            | _token_set(query.prompt_summary)
        )
        score += min(0.8, 0.12 * len(overlap))
        if memory.task_embedding and query.task_query_text:
            query_embedding = TaskEmbedder(enabled=False, dim=len(memory.task_embedding)).embed_text(query.task_query_text)
            score += 0.15 * _cosine_similarity(query_embedding, memory.task_embedding)
        return score

    def update_status(self, memory_id: str, status: str) -> None:
        memories = self.load_memories()
        for memory in memories:
            if memory.memory_id == memory_id:
                memory.status = status
        self.save_memories(memories)

    def log_metrics(self, payload: Dict[str, Any]) -> None:
        self._append_jsonl(self.metrics_path, payload)

    def recent_task_outcomes(self, workflow_id: str) -> Dict[str, AttributionEvent]:
        latest: Dict[str, AttributionEvent] = {}
        for event in self.load_events():
            if event.workflow_id == workflow_id:
                latest[event.task_id] = event
        return latest

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists() or path.stat().st_size == 0:
            return []
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSONL line in {path}")
        return rows


class ExperienceGuidedValidationSampler:
    def __init__(self, config: ExperienceGuidedConfig, embedder: Optional[TaskEmbedder] = None, seed: int = 17):
        self.config = config
        self.embedder = embedder or TaskEmbedder(
            model_name=config.task_embedding_model_name,
            base_url=config.task_embedding_base_url,
            api_key=config.task_embedding_api_key,
            enabled=config.task_embedding_enabled,
            dim=config.task_embedding_dim,
        )
        self.seed = seed

    def plan(
        self,
        validation_records: Sequence[Dict[str, Any]],
        memories: Sequence[WorkflowMemory],
        parent_outcomes: Dict[str, AttributionEvent],
    ) -> ValidationPlan:
        task_ids = [str(row.get("_task_id", idx)) for idx, row in enumerate(validation_records)]
        text_by_task = {tid: self._question(row) for tid, row in zip(task_ids, validation_records)}
        embedding_by_task = {tid: self.embedder.embed_text(text_by_task[tid]) for tid in task_ids}

        failed_tasks = [tid for tid, event in parent_outcomes.items() if event.outcome == "failure"]
        success_tasks = [tid for tid, event in parent_outcomes.items() if event.outcome == "success"]
        memory_vectors = [m.task_embedding for m in memories if m.status in {"validated", "candidate"} and m.task_embedding]

        targeted = self._prioritize(
            failed_tasks,
            embedding_by_task,
            memory_vectors,
            self.config.targeted_failure_size,
        )
        success_guard = self._prioritize(
            success_tasks,
            embedding_by_task,
            memory_vectors,
            self.config.success_guard_size,
        )
        anchor = self._anchor_indices(task_ids, self.config.anchor_size)

        id_to_index = {tid: idx for idx, tid in enumerate(task_ids)}
        targeted_indices = [id_to_index[tid] for tid in targeted if tid in id_to_index]
        success_indices = [id_to_index[tid] for tid in success_guard if tid in id_to_index]
        anchor_indices = [id_to_index[tid] for tid in anchor if tid in id_to_index]
        full_indices = list(range(len(validation_records)))
        return ValidationPlan(
            targeted_failure_set=targeted_indices,
            success_guard_set=success_indices,
            anchor_set=anchor_indices,
            full_set=full_indices,
            targeted_failure_tasks=targeted,
            success_guard_tasks=success_guard,
        )

    def _prioritize(
        self,
        task_ids: Sequence[str],
        embedding_by_task: Dict[str, Sequence[float]],
        memory_vectors: Sequence[Sequence[float]],
        limit: int,
    ) -> List[str]:
        ranked = sorted(
            set(task_ids),
            key=lambda tid: (
                self._best_similarity(embedding_by_task.get(tid, []), memory_vectors),
                tid,
            ),
            reverse=True,
        )
        if len(ranked) >= limit:
            return ranked[:limit]
        fallback = [tid for tid in embedding_by_task if tid not in ranked]
        return (ranked + fallback)[:limit]

    def _anchor_indices(self, task_ids: Sequence[str], size: int) -> List[str]:
        ranked = sorted(task_ids, key=lambda tid: hashlib.sha256(f"{self.seed}:{tid}".encode("utf-8")).hexdigest())
        return ranked[: min(size, len(ranked))]

    def _question(self, row: Dict[str, Any]) -> str:
        return str(row.get("problem") or row.get("question") or row.get("prompt") or "")

    def _best_similarity(self, query: Sequence[float], memory_vectors: Sequence[Sequence[float]]) -> float:
        if not query or not memory_vectors:
            return 0.0
        return max((_cosine_similarity(query, item) for item in memory_vectors if item), default=0.0)


class ExperienceGuidedController:
    def __init__(self, dataset: str, config: ExperienceGuidedConfig):
        self.dataset = dataset
        self.config = config
        self.analyzer = AttributionAnalyzer(config.taxonomy)
        self.compressor = ExperienceCompressor(config.min_support, config.min_confidence)
        self.store = MemoryStore(config.store_path)
        self.embedder = TaskEmbedder(
            model_name=config.task_embedding_model_name,
            base_url=config.task_embedding_base_url,
            api_key=config.task_embedding_api_key,
            enabled=config.task_embedding_enabled,
            dim=config.task_embedding_dim,
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def observe_evaluation(
        self,
        workflow_id: str,
        workflow_version: str,
        workflow_source: str,
        details: Dict[str, Any],
        stage: str,
    ) -> int:
        if not self.enabled or not self.config.attribution_enabled:
            return 0

        count = 0
        columns = details.get("columns", [])
        for row in details.get("results", []):
            result_map = self._row_to_map(columns, row.get("result", ()))
            task = row.get("problem", {})
            task_id = str(task.get("_task_id", row.get("task_index", "unknown")))
            outcome = "success" if _safe_float(result_map.get("score"), 0.0) > 0 else "failure"
            final_output = str(result_map.get("prediction") or result_map.get("output") or "")
            question = self._question(task)
            task_embedding = self.embedder.embed_text(question)
            task_cluster = self.embedder.cluster_id(task_embedding)
            ground_truth = str(result_map.get("expected_output") or task.get("solution") or task.get("answer") or "")
            nodes = self.analyzer.block_view.extract(workflow_source)
            trace = ExecutionTrace(
                task_id=task_id,
                workflow_id=workflow_id,
                workflow_version=workflow_version,
                task_pattern=task_cluster or "emb_generic",
                node_execution_sequence=nodes,
                node_summaries=self._node_summaries(nodes, question, final_output),
                final_output=final_output,
                ground_truth=ground_truth,
                score=_safe_float(result_map.get("score"), 0.0),
                outcome=outcome,
                token_cost=_safe_float(result_map.get("cost"), 0.0),
                runtime_seconds=_safe_float(row.get("runtime_seconds"), 0.0),
                task_text=question,
                task_embedding=task_embedding,
                task_cluster=task_cluster,
                evaluator_feedback={
                    "question": question,
                    "expected_output": ground_truth,
                    "score": result_map.get("score"),
                },
            )
            self.store.add_trace(trace)
            event = self.analyzer.analyze(
                workflow_source=workflow_source,
                trace=trace,
                evaluator_feedback=trace.evaluator_feedback,
                stage=stage,
            )
            self.store.add_event(event)
            count += 1

        if count and count % max(1, self.config.compression_interval) == 0:
            self.compress()
        return count

    def compress(self) -> List[WorkflowMemory]:
        memories = self.store.compress(self.compressor)
        return memories

    def query_memories(
        self,
        task_text: str = "",
        failure_type: str = "",
        failed_node: str = "",
        optimization_state: Optional[OptimizationStateQuery] = None,
        limit: int = 5,
    ) -> List[WorkflowMemory]:
        if not self.enabled:
            return []
        state = optimization_state
        if state is not None:
            return self.store.query(
                failure_type=state.failure_types[0] if state.failure_types else failure_type,
                failed_node=state.failed_nodes[0] if state.failed_nodes else failed_node,
                optimization_state=state,
                limit=limit,
                statuses=["validated", "candidate", "rejected"],
            )
        task_embedding = self.embedder.embed_text(task_text) if task_text else None
        task_cluster = self.embedder.cluster_id(task_embedding or []) if task_embedding else ""
        return self.store.query(
            task_pattern=task_cluster,
            task_embedding=task_embedding,
            failure_type=failure_type,
            failed_node=failed_node,
            limit=limit,
            similarity_threshold=self.config.task_similarity_threshold,
        )

    def build_optimization_state_query(
        self,
        parent_round: int,
        parent_workflow_id: str,
        score: float,
        failure_cards: Sequence[Dict[str, Any]],
        graph_summary: str,
        prompt_summary: str,
        recent_failed_modifications: Sequence[str],
        recent_successful_modifications: Sequence[str],
        modification_intent: str = "",
    ) -> OptimizationStateQuery:
        failure_types = [str(card.get("failure_type") or card.get("error_type") or "unknown") for card in failure_cards]
        failed_nodes = [str(card.get("node_name") or "") for card in failure_cards if card.get("node_name")]
        patch_targets = [str(card.get("patch_target") or card.get("node_name") or "") for card in failure_cards if card.get("patch_target") or card.get("node_name")]
        mismatch_info = " | ".join([str(card.get("mismatch_info") or "") for card in failure_cards if card.get("mismatch_info")])[:600]
        extract_result_summary = " | ".join([str(card.get("extract_result") or "") for card in failure_cards if card.get("extract_result")])[:400]
        task_query_text = " ".join(
            [
                mismatch_info,
                extract_result_summary,
                graph_summary[:200],
                prompt_summary[:160],
                " ".join(failure_types[:3]),
            ]
        ).strip()
        return OptimizationStateQuery(
            parent_round=parent_round,
            parent_workflow_id=parent_workflow_id,
            score=score,
            failure_types=failure_types,
            failed_nodes=failed_nodes,
            patch_targets=patch_targets,
            mismatch_info=mismatch_info,
            extract_result_summary=extract_result_summary,
            graph_summary=graph_summary[:700],
            prompt_summary=prompt_summary[:500],
            recent_failed_modifications=[_compact(x, 180) for x in recent_failed_modifications[:3]],
            recent_successful_modifications=[_compact(x, 180) for x in recent_successful_modifications[:2]],
            modification_intent=_compact(modification_intent, 240),
            task_query_text=task_query_text[:900],
        )

    def build_memory_guidance(
        self,
        optimization_state: OptimizationStateQuery,
        limit: int = 8,
    ) -> MemoryGuidance:
        memories = self.query_memories(optimization_state=optimization_state, limit=limit)
        validated = [m for m in memories if m.status == "validated"]
        candidates = [m for m in memories if m.status == "candidate"]
        rejected = [m for m in memories if m.status == "rejected"]
        focus_nodes = list(dict.fromkeys(
            [m.failed_node for m in validated if m.failed_node] +
            [m.patch_target for m in validated if m.patch_target] +
            [m.failed_node for m in candidates if m.failed_node]
        ))
        preserve_nodes = list(dict.fromkeys([node for m in validated for node in m.successful_subworkflow if node]))
        avoid_nodes_or_patches = list(dict.fromkeys(
            [m.patch_rule for m in rejected if m.patch_rule] + [m.failed_node for m in rejected if m.failed_node]
        ))
        recommended_patch_rules = list(dict.fromkeys([m.patch_rule for m in validated if m.patch_rule]))
        candidate_patch_rules = list(dict.fromkeys([m.patch_rule for m in candidates if m.patch_rule]))
        evidence_summary = [
            f"{m.status}:{m.failure_pattern or m.success_pattern or 'generic'}:{m.failed_node or m.patch_target or 'n/a'}"
            for m in memories[:6]
        ]
        confidence = _mean([m.confidence for m in validated]) if validated else _mean([m.confidence for m in memories])
        return MemoryGuidance(
            focus_nodes=focus_nodes[:5],
            preserve_nodes=preserve_nodes[:5],
            avoid_nodes_or_patches=avoid_nodes_or_patches[:5],
            recommended_patch_rules=recommended_patch_rules[:4],
            candidate_patch_rules=candidate_patch_rules[:4],
            evidence_summary=evidence_summary,
            confidence=confidence,
            validated_memories=validated,
            candidate_memories=candidates,
            rejected_memories=rejected,
        )

    def build_memory_guidance_text(self, guidance: MemoryGuidance) -> str:
        if not any(
            [
                guidance.validated_memories,
                guidance.candidate_memories,
                guidance.rejected_memories,
                guidance.focus_nodes,
                guidance.preserve_nodes,
            ]
        ):
            return ""
        rows = ["[Memory-Guided Optimization]"]
        rows.append("Validated memory:")
        if guidance.validated_memories:
            for memory in guidance.validated_memories[:3]:
                rows.append(
                    f"- fix={memory.patch_rule} node={memory.failed_node or memory.patch_target or 'n/a'} "
                    f"preserve={','.join(memory.successful_subworkflow[:3]) or 'n/a'} confidence={memory.confidence:.2f}"
                )
        else:
            rows.append("- none")
        rows.append("Candidate memory:")
        if guidance.candidate_memories:
            for memory in guidance.candidate_memories[:3]:
                rows.append(
                    f"- maybe={memory.patch_rule} node={memory.failed_node or memory.patch_target or 'n/a'} confidence={memory.confidence:.2f}"
                )
        else:
            rows.append("- none")
        rows.append("Rejected memory:")
        if guidance.rejected_memories:
            for memory in guidance.rejected_memories[:3]:
                rows.append(
                    f"- avoid={memory.patch_rule} node={memory.failed_node or memory.patch_target or 'n/a'} confidence={memory.confidence:.2f}"
                )
        else:
            rows.append("- none")
        rows.append("[Patch Scope]")
        rows.append(f"Focus nodes: {', '.join(guidance.focus_nodes) or 'none'}")
        rows.append(f"Preserve nodes: {', '.join(guidance.preserve_nodes) or 'none'}")
        rows.append(f"Avoid patches: {', '.join(guidance.avoid_nodes_or_patches) or 'none'}")
        rows.append(f"Recommended patch rules: {', '.join(guidance.recommended_patch_rules) or 'none'}")
        rows.append(f"Candidate patch rules: {', '.join(guidance.candidate_patch_rules) or 'none'}")
        rows.append(f"Evidence summary: {' | '.join(guidance.evidence_summary[:4]) or 'none'}")
        rows.append(f"Guidance confidence: {guidance.confidence:.2f}")
        return "\n".join(rows) + "\n"

    def build_prompt_hint(self, task_text: str, failure_type: str = "", failed_node: str = "") -> str:
        memories = self.query_memories(task_text=task_text, failure_type=failure_type, failed_node=failed_node, limit=3)
        if not memories:
            return ""
        guidance = MemoryGuidance(
            focus_nodes=[m.failed_node for m in memories if m.failed_node][:3],
            preserve_nodes=[node for m in memories if m.status == "validated" for node in m.successful_subworkflow][:3],
            avoid_nodes_or_patches=[m.patch_rule for m in memories if m.status == "rejected"][:3],
            recommended_patch_rules=[m.patch_rule for m in memories if m.status == "validated"][:3],
            candidate_patch_rules=[m.patch_rule for m in memories if m.status == "candidate"][:3],
            evidence_summary=[
                f"{m.status}:{m.failure_pattern or 'success'}:{m.failed_node or 'n/a'}" for m in memories[:3]
            ],
            confidence=_mean([m.confidence for m in memories]),
            validated_memories=[m for m in memories if m.status == "validated"],
            candidate_memories=[m for m in memories if m.status == "candidate"],
            rejected_memories=[m for m in memories if m.status == "rejected"],
        )
        return self.build_memory_guidance_text(guidance)

    def parent_outcomes(self, workflow_id: str) -> Dict[str, AttributionEvent]:
        return self.store.recent_task_outcomes(workflow_id)

    def log_metrics(self, payload: Dict[str, Any]) -> None:
        if self.enabled:
            self.store.log_metrics(payload)

    def _row_to_map(self, columns: Sequence[str], result: Sequence[Any]) -> Dict[str, Any]:
        mapped = dict(zip(columns, list(result)))
        return mapped

    def _node_summaries(self, nodes: Sequence[str], question: str, final_output: str) -> List[Dict[str, str]]:
        if not nodes:
            return []
        out: List[Dict[str, str]] = []
        prev_output = _compact(question)
        for idx, node in enumerate(nodes):
            is_last = idx == len(nodes) - 1
            out.append(
                {
                    "node": node,
                    "input_summary": prev_output,
                    "output_summary": _compact(final_output if is_last else f"intermediate_after_{node}"),
                }
            )
            prev_output = out[-1]["output_summary"]
        return out

    def _question(self, row: Dict[str, Any]) -> str:
        return str(row.get("problem") or row.get("question") or row.get("prompt") or "")


class ExperienceGuidedEvaluationProtocol:
    def __init__(self, config: ExperienceGuidedConfig):
        self.config = config

    async def evaluate_candidate(
        self,
        plan: ValidationPlan,
        subset_runner: Callable[[List[int], str, bool], Awaitable[Dict[str, Any]]],
        parent_outcomes: Dict[str, AttributionEvent],
        historical_top_scores: Sequence[float],
    ) -> CandidateEvaluationSummary:
        stages_run: List[str] = []
        stage_results: Dict[str, StageEvaluationResult] = {}

        targeted_ids = [str(x) for x in plan.targeted_failure_set]
        guard_ids = [str(x) for x in plan.success_guard_set]
        stage1_indices = sorted(set(plan.targeted_failure_set + plan.success_guard_set))
        targeted_result_payload = await subset_runner(stage1_indices, "targeted", True)
        stage1 = self._to_stage_result("targeted", stage1_indices, targeted_result_payload)
        stage_results["targeted"] = stage1
        stages_run.append("targeted")

        targeted_gain = self._targeted_gain(targeted_result_payload, set(targeted_ids), parent_outcomes)
        regression_loss = self._regression_loss(targeted_result_payload, set(guard_ids), parent_outcomes)
        stage1_pass = targeted_gain >= -0.05 and regression_loss <= 0.35

        if not stage1_pass:
            selection_score = self._selection_score(0.0, targeted_gain, regression_loss, stage1.avg_cost)
            return CandidateEvaluationSummary(
                comparable_score=0.0,
                selection_score=selection_score,
                score_source="screened_out",
                targeted_gain=targeted_gain,
                regression_loss=regression_loss,
                anchor_score=0.0,
                full_score=None,
                stages_run=stages_run,
                stage_results=stage_results,
            )

        anchor_payload = await subset_runner(plan.anchor_set, "anchor", True)
        anchor = self._to_stage_result("anchor", plan.anchor_set, anchor_payload)
        stage_results["anchor"] = anchor
        stages_run.append("anchor")

        anchor_score = anchor.score
        full_score: Optional[float] = None
        comparable_score = anchor_score
        score_source = "anchor"

        should_full_validate = self._should_full_validate(anchor_score, historical_top_scores)
        if should_full_validate:
            full_payload = await subset_runner(plan.full_set, "full", True)
            full = self._to_stage_result("full", plan.full_set, full_payload)
            stage_results["full"] = full
            stages_run.append("full")
            full_score = full.score
            comparable_score = full.score
            score_source = "full"

        selection_score = self._selection_score(anchor_score, targeted_gain, regression_loss, anchor.avg_cost)
        return CandidateEvaluationSummary(
            comparable_score=comparable_score,
            selection_score=selection_score,
            score_source=score_source,
            targeted_gain=targeted_gain,
            regression_loss=regression_loss,
            anchor_score=anchor_score,
            full_score=full_score,
            stages_run=stages_run,
            stage_results=stage_results,
        )

    def _to_stage_result(self, stage_name: str, indices: List[int], payload: Dict[str, Any]) -> StageEvaluationResult:
        task_ids = [str(x) for x in payload.get("task_ids", [str(x) for x in indices])]
        return StageEvaluationResult(
            stage_name=stage_name,
            indices=indices,
            task_ids=task_ids,
            score=_safe_float(payload.get("score"), 0.0),
            avg_cost=_safe_float(payload.get("avg_cost"), 0.0),
            total_cost=_safe_float(payload.get("total_cost"), 0.0),
            details=payload.get("details"),
        )

    def _targeted_gain(
        self,
        payload: Dict[str, Any],
        targeted_ids: set[str],
        parent_outcomes: Dict[str, AttributionEvent],
    ) -> float:
        if not targeted_ids:
            return 0.0
        rows = payload.get("details", {}).get("results", [])
        columns = payload.get("details", {}).get("columns", [])
        current_scores: List[float] = []
        baseline_scores: List[float] = []
        for row in rows:
            problem = row.get("problem", {})
            task_id = str(problem.get("_task_id", row.get("task_index", "")))
            if task_id not in targeted_ids:
                continue
            score = _safe_float(dict(zip(columns, row.get("result", ()))).get("score"), 0.0)
            current_scores.append(score)
            baseline_scores.append(1.0 if parent_outcomes.get(task_id, AttributionEvent("", "", "", "", "", "failure", [])).outcome == "success" else 0.0)
        return _mean(current_scores) - _mean(baseline_scores)

    def _regression_loss(
        self,
        payload: Dict[str, Any],
        success_guard_ids: set[str],
        parent_outcomes: Dict[str, AttributionEvent],
    ) -> float:
        if not success_guard_ids:
            return 0.0
        rows = payload.get("details", {}).get("results", [])
        columns = payload.get("details", {}).get("columns", [])
        current_scores: List[float] = []
        baseline_scores: List[float] = []
        for row in rows:
            problem = row.get("problem", {})
            task_id = str(problem.get("_task_id", row.get("task_index", "")))
            if task_id not in success_guard_ids:
                continue
            score = _safe_float(dict(zip(columns, row.get("result", ()))).get("score"), 0.0)
            current_scores.append(score)
            baseline_scores.append(1.0 if parent_outcomes.get(task_id, AttributionEvent("", "", "", "", "", "failure", [])).outcome == "success" else 0.0)
        return max(0.0, _mean(baseline_scores) - _mean(current_scores))

    def _selection_score(self, anchor_score: float, targeted_gain: float, regression_loss: float, avg_cost: float) -> float:
        return (
            anchor_score
            + self.config.beta * targeted_gain
            - self.config.gamma * regression_loss
            - self.config.lambda_cost * avg_cost
        )

    def _should_full_validate(self, anchor_score: float, historical_top_scores: Sequence[float]) -> bool:
        if not historical_top_scores:
            return True
        k = max(1, self.config.full_validation_top_k)
        threshold = sorted(historical_top_scores, reverse=True)[:k][-1]
        return anchor_score >= threshold or anchor_score >= max(historical_top_scores) - 0.05
