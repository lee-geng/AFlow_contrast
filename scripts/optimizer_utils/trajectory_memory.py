import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from analysis.blame_attributor import RuleBasedBlameAttributor, WorkflowBlockView
from data.aflow_adapters import AFlowTreeAdapter, AFlowTreeBundle, WorkflowNode


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _hash_workflow(workflow: str) -> str:
    return hashlib.sha256(_compact(workflow).encode("utf-8")).hexdigest()


def _round_id(node_id: str) -> Optional[int]:
    try:
        return int(str(node_id))
    except (TypeError, ValueError):
        return None


def _edit_primitive(edit_text: str) -> str:
    t = (edit_text or "").lower()
    if any(k in t for k in ["add", "insert", "append"]):
        return "insert_block"
    if any(k in t for k in ["remove", "delete"]):
        return "delete_block"
    if any(k in t for k in ["prompt", "instruction", "wording"]):
        return "prompt_change"
    if any(k in t for k in ["tool", "api"]):
        return "tool_change"
    if any(k in t for k in ["verify", "check", "review", "test"]):
        return "verification_patch"
    return "unknown"


def _minimality_estimate(parent_workflow: str, child_workflow: str, edit_text: str) -> float:
    pa = _compact(parent_workflow)
    ch = _compact(child_workflow)
    if not pa or not ch:
        return 0.5
    overlap = sum(1 for i, c in enumerate(ch[: len(pa)]) if pa[i] == c)
    ratio = overlap / max(len(pa), 1)
    edit_penalty = min(len((edit_text or "").split()) / 80.0, 1.0)
    return max(0.0, min(1.0, 0.7 * ratio + 0.3 * (1.0 - edit_penalty)))


@dataclass
class SuccessCase:
    case_id: str
    task_context: str
    workflow_graph: str
    graph_hash: str
    score: float
    cost: float
    node_trace_summary: str
    tool_trace_summary: str
    verifier_outputs: Dict[str, Any]
    critical_subgraph: List[str]
    parent_case_id: Optional[str]
    edit_from_parent: str
    search_depth: int
    confidence: float


@dataclass
class FailureCase:
    case_id: str
    task_context: str
    workflow_graph: str
    graph_hash: str
    score: float
    cost: float
    node_trace_summary: str
    tool_trace_summary: str
    verifier_outputs: Dict[str, Any]
    localized_failure_node: str
    failure_type: str
    propagation_path: List[str]
    parent_case_id: Optional[str]
    edit_from_parent: str
    search_depth: int
    confidence: float


@dataclass
class ContrastRecord:
    contrast_id: str
    base_case_id: str
    new_case_id: str
    edit_primitive: str
    changed_subgraph: List[str]
    delta_score: float
    delta_cost: float
    resolved_failure_type: str
    introduced_failure_type: str
    minimality_estimate: float
    edit_effect_label: str
    confidence: float


@dataclass
class FailureSignature:
    signature_id: str
    applicable_context: str
    trigger_condition: str
    failure_type: str
    root_cause_template: str
    first_bad_node_type: str
    propagation_template: str
    unsafe_edits: List[str]
    recommended_guardrails: List[str]
    support_cases: List[str]
    support_contrasts: List[str]
    confidence: float


@dataclass
class SuccessPattern:
    pattern_id: str
    applicable_context: str
    effective_subgraph: List[str]
    ordering_constraint: str
    required_verifiers: List[str]
    effective_edit_pattern: str
    expected_gain: float
    cost_profile: Dict[str, float]
    support_cases: List[str]
    support_contrasts: List[str]
    confidence: float


@dataclass
class EditRule:
    rule_id: str
    precondition: str
    target_failure_signature: str
    candidate_edit: str
    expected_delta_score: float
    expected_delta_cost: float
    risk_of_regression: float
    requires_guardrail: bool
    confidence: float


@dataclass
class MemoryArtifacts:
    success_cases: List[SuccessCase] = field(default_factory=list)
    failure_cases: List[FailureCase] = field(default_factory=list)
    contrast_records: List[ContrastRecord] = field(default_factory=list)
    failure_signatures: List[FailureSignature] = field(default_factory=list)
    success_patterns: List[SuccessPattern] = field(default_factory=list)
    edit_rules: List[EditRule] = field(default_factory=list)


class TrajectoryCompiler:
    """Compile AFlow trajectory artifacts into normalized success/failure/contrast memory."""

    def __init__(self):
        self.attributor = RuleBasedBlameAttributor()
        self.block_view = WorkflowBlockView()

    def compile(self, workflows_dir: str, task_context: Optional[str] = None) -> MemoryArtifacts:
        bundle = AFlowTreeAdapter(task_family=task_context or "unknown").load(workflows_dir)
        aggregates = self._load_round_aggregates(workflows_dir)
        case_depth = self._compute_depth(bundle.nodes)
        case_kind: Dict[str, str] = self._classify_cases(bundle.nodes)
        artifacts = MemoryArtifacts()

        failure_by_case: Dict[str, FailureCase] = {}
        all_cases: Dict[str, Dict[str, Any]] = {}
        for node in bundle.nodes.values():
            score = _safe_float(node.score_mean, 0.0)
            cost = self._lookup_cost(aggregates, node.node_id)
            trace_summary = str((node.log_feedback or {}).get("trace_summary") or "")[:400]
            fail_summary = str((node.log_feedback or {}).get("failure_summary") or "")[:500]
            verifier_outputs = self._extract_verifier_outputs(node.log_feedback)
            blocks = self.block_view.extract(node.workflow or "")
            critical_subgraph = blocks[:3]
            parent_case_id = f"case_{node.parent_id}" if node.parent_id else None
            confidence = self._estimate_case_confidence(node, bool(trace_summary or fail_summary))
            case_id = f"case_{node.node_id}"
            all_cases[case_id] = {
                "score": score,
                "cost": cost,
                "workflow": node.workflow or "",
                "task_context": node.task_family or task_context or "unknown",
            }

            if case_kind.get(node.node_id) == "success":
                artifacts.success_cases.append(
                    SuccessCase(
                        case_id=case_id,
                        task_context=node.task_family or task_context or "unknown",
                        workflow_graph=node.workflow or "",
                        graph_hash=_hash_workflow(node.workflow or ""),
                        score=score,
                        cost=cost,
                        node_trace_summary=trace_summary,
                        tool_trace_summary=self._extract_tool_trace(node.log_feedback),
                        verifier_outputs=verifier_outputs,
                        critical_subgraph=critical_subgraph,
                        parent_case_id=parent_case_id,
                        edit_from_parent=node.edit_text or "",
                        search_depth=case_depth.get(node.node_id, 0),
                        confidence=confidence,
                    )
                )
            else:
                blame = self.attributor.attribute(
                    parent_workflow=node.workflow or "",
                    execution_feedback={
                        "failure_summary": fail_summary,
                        "trace_summary": trace_summary,
                        "score_mean": score,
                        "score_std": node.score_std or 0.0,
                    },
                    edit_text=node.edit_text or "",
                    child_workflow="",
                )
                failure = FailureCase(
                    case_id=case_id,
                    task_context=node.task_family or task_context or "unknown",
                    workflow_graph=node.workflow or "",
                    graph_hash=_hash_workflow(node.workflow or ""),
                    score=score,
                    cost=cost,
                    node_trace_summary=trace_summary,
                    tool_trace_summary=self._extract_tool_trace(node.log_feedback),
                    verifier_outputs=verifier_outputs,
                    localized_failure_node=blame.target_block,
                    failure_type=blame.blame_type,
                    propagation_path=[blame.target_block, "final_answer"],
                    parent_case_id=parent_case_id,
                    edit_from_parent=node.edit_text or "",
                    search_depth=case_depth.get(node.node_id, 0),
                    confidence=max(0.4, min(0.98, 0.6 * blame.blame_confidence + 0.4 * confidence)),
                )
                artifacts.failure_cases.append(failure)
                failure_by_case[case_id] = failure

        artifacts.contrast_records = self._build_contrasts(bundle, all_cases, failure_by_case)
        return artifacts

    def _load_round_aggregates(self, workflows_dir: str) -> Dict[int, Dict[str, float]]:
        path = Path(workflows_dir)
        if (path / "workflows").exists():
            path = path / "workflows"
        results_file = path / "results.json"
        if not results_file.exists() or results_file.stat().st_size == 0:
            return {}
        payload = json.loads(results_file.read_text(encoding="utf-8-sig"))
        grouped: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"score": [], "cost": []})
        for row in payload if isinstance(payload, list) else []:
            rid = _round_id(row.get("round"))
            if rid is None:
                continue
            grouped[rid]["score"].append(_safe_float(row.get("score"), 0.0))
            grouped[rid]["cost"].append(_safe_float(row.get("avg_cost"), 0.0))
        out: Dict[int, Dict[str, float]] = {}
        for rid, data in grouped.items():
            out[rid] = {
                "score": sum(data["score"]) / max(len(data["score"]), 1),
                "cost": sum(data["cost"]) / max(len(data["cost"]), 1),
            }
        return out

    def _lookup_cost(self, aggregates: Dict[int, Dict[str, float]], node_id: str) -> float:
        rid = _round_id(node_id)
        if rid is None or rid not in aggregates:
            return 0.0
        return aggregates[rid].get("cost", 0.0)

    def _classify_cases(self, nodes: Dict[str, WorkflowNode]) -> Dict[str, str]:
        scores = [float(n.score_mean) for n in nodes.values() if n.score_mean is not None]
        baseline = sum(scores) / max(len(scores), 1) if scores else 0.0
        kind: Dict[str, str] = {}
        for node in nodes.values():
            if node.parent_id and node.parent_id in nodes:
                parent = nodes[node.parent_id]
                if parent.score_mean is not None and node.score_mean is not None:
                    kind[node.node_id] = "success" if float(node.score_mean) > float(parent.score_mean) else "failure"
                    continue
            node_score = _safe_float(node.score_mean, 0.0)
            kind[node.node_id] = "success" if node_score >= baseline else "failure"
        return kind

    def _compute_depth(self, nodes: Dict[str, WorkflowNode]) -> Dict[str, int]:
        depth: Dict[str, int] = {}
        visiting: set[str] = set()

        def _depth(node_id: str) -> int:
            if node_id in depth:
                return depth[node_id]
            if node_id in visiting:
                # Break parent-cycle in noisy trajectories (e.g., self-parent).
                depth[node_id] = 0
                return 0
            node = nodes.get(node_id)
            if node is None or not node.parent_id or node.parent_id not in nodes:
                depth[node_id] = 0
                return 0
            visiting.add(node_id)
            parent_id = node.parent_id
            if parent_id == node_id:
                depth[node_id] = 0
            else:
                depth[node_id] = _depth(parent_id) + 1
            visiting.discard(node_id)
            return depth[node_id]

        for node_id in nodes:
            _depth(node_id)
        return depth
    def _extract_verifier_outputs(self, log_feedback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        data = log_feedback or {}
        text = _compact(json.dumps(data, ensure_ascii=False)[:800])
        has_verify = any(k in text.lower() for k in ["verify", "check", "review", "test"])
        return {
            "contains_verifier_signal": has_verify,
            "summary": text[:220],
        }

    def _extract_tool_trace(self, log_feedback: Optional[Dict[str, Any]]) -> str:
        data = log_feedback or {}
        trace = str(data.get("trace_summary") or "")
        parts = [p.strip() for p in trace.split("|") if p.strip()]
        return " | ".join(parts[:3]) if parts else trace[:200]

    def _estimate_case_confidence(self, node: WorkflowNode, has_trace: bool) -> float:
        std = _safe_float(node.score_std, 0.05)
        score_signal = 1.0 - min(abs(std), 1.0)
        trace_signal = 1.0 if has_trace else 0.6
        return max(0.3, min(0.98, 0.5 * score_signal + 0.5 * trace_signal))

    def _build_contrasts(
        self,
        bundle: AFlowTreeBundle,
        all_cases: Dict[str, Dict[str, Any]],
        failures: Dict[str, FailureCase],
    ) -> List[ContrastRecord]:
        out: List[ContrastRecord] = []
        for edge in bundle.edges:
            base_case_id = f"case_{edge.parent_id}"
            new_case_id = f"case_{edge.child_id}"
            if base_case_id not in all_cases or new_case_id not in all_cases:
                continue
            base = all_cases[base_case_id]
            new = all_cases[new_case_id]
            delta_score = new["score"] - base["score"]
            delta_cost = new["cost"] - base["cost"]
            base_failure = failures.get(base_case_id)
            new_failure = failures.get(new_case_id)

            resolved = base_failure.failure_type if (base_failure and not new_failure) else ""
            introduced = new_failure.failure_type if (new_failure and not base_failure) else ""
            if base_failure and new_failure and base_failure.failure_type != new_failure.failure_type:
                introduced = new_failure.failure_type

            label = self._effect_label(base_failure is not None, new_failure is not None, delta_score)
            changed_subgraph = self._changed_subgraph(base.get("workflow", ""), new.get("workflow", ""))
            confidence = max(0.3, min(0.98, 0.4 + 0.4 * (1.0 - abs(delta_cost)) + 0.2 * min(abs(delta_score) * 2, 1.0)))

            out.append(
                ContrastRecord(
                    contrast_id=f"contrast_{edge.parent_id}_{edge.child_id}",
                    base_case_id=base_case_id,
                    new_case_id=new_case_id,
                    edit_primitive=_edit_primitive(edge.edit_text),
                    changed_subgraph=changed_subgraph,
                    delta_score=delta_score,
                    delta_cost=delta_cost,
                    resolved_failure_type=resolved,
                    introduced_failure_type=introduced,
                    minimality_estimate=_minimality_estimate(base.get("workflow", ""), new.get("workflow", ""), edge.edit_text),
                    edit_effect_label=label,
                    confidence=confidence,
                )
            )
        return out

    def _changed_subgraph(self, parent_workflow: str, child_workflow: str) -> List[str]:
        parent_blocks = set(self.block_view.extract(parent_workflow))
        child_blocks = set(self.block_view.extract(child_workflow))
        added = sorted(child_blocks - parent_blocks)
        removed = sorted(parent_blocks - child_blocks)
        merged = (added + removed)[:4]
        return merged if merged else sorted(list(child_blocks))[:2]

    def _effect_label(self, was_failure: bool, is_failure: bool, delta_score: float) -> str:
        if was_failure and not is_failure:
            return "failure_to_success"
        if was_failure and is_failure:
            return "failure_to_failure"
        if not was_failure and is_failure:
            return "success_to_failure"
        if delta_score > 0:
            return "success_to_success_gain"
        return "success_to_success_flat"


class PatternDistiller:
    """Distill failure signatures, success patterns, and minimal edit rules."""

    def distill(self, artifacts: MemoryArtifacts) -> MemoryArtifacts:
        artifacts.failure_signatures = self._mine_failure_signatures(artifacts.failure_cases, artifacts.contrast_records)
        artifacts.success_patterns = self._mine_success_patterns(artifacts.success_cases, artifacts.contrast_records)
        artifacts.edit_rules = self._induce_edit_rules(
            artifacts.failure_signatures,
            artifacts.success_patterns,
            artifacts.contrast_records,
        )
        return artifacts

    def _mine_failure_signatures(
        self,
        failures: List[FailureCase],
        contrasts: List[ContrastRecord],
    ) -> List[FailureSignature]:
        grouped: Dict[Tuple[str, str, str], List[FailureCase]] = defaultdict(list)
        for case in failures:
            key = (case.task_context, case.failure_type, case.localized_failure_node)
            grouped[key].append(case)

        contrasts_by_case: Dict[str, List[ContrastRecord]] = defaultdict(list)
        for contrast in contrasts:
            contrasts_by_case[contrast.base_case_id].append(contrast)
            contrasts_by_case[contrast.new_case_id].append(contrast)

        signatures: List[FailureSignature] = []
        for idx, ((context, failure_type, node), members) in enumerate(grouped.items(), start=1):
            support_cases = [m.case_id for m in members]
            support_contrasts = []
            unsafe_edits = []
            for cid in support_cases:
                for c in contrasts_by_case.get(cid, []):
                    support_contrasts.append(c.contrast_id)
                    if c.introduced_failure_type == failure_type or (c.delta_score < 0 and c.edit_effect_label.endswith("to_failure")):
                        unsafe_edits.append(c.edit_primitive)

            unique_unsafe = sorted(set(unsafe_edits))
            confidence = min(0.98, 0.45 + 0.1 * min(len(members), 5) + 0.05 * min(len(support_contrasts), 5))
            signatures.append(
                FailureSignature(
                    signature_id=f"fs_{idx}",
                    applicable_context=context,
                    trigger_condition=f"Workflow shows {failure_type} indicators near {node}.",
                    failure_type=failure_type,
                    root_cause_template=f"Repeated {failure_type} under context {context}.",
                    first_bad_node_type=node,
                    propagation_template=f"{node} -> final_answer",
                    unsafe_edits=unique_unsafe,
                    recommended_guardrails=self._guardrails_for_failure(failure_type),
                    support_cases=support_cases,
                    support_contrasts=sorted(set(support_contrasts)),
                    confidence=confidence,
                )
            )
        return signatures

    def _mine_success_patterns(
        self,
        successes: List[SuccessCase],
        contrasts: List[ContrastRecord],
    ) -> List[SuccessPattern]:
        contrasts_by_new: Dict[str, List[ContrastRecord]] = defaultdict(list)
        for c in contrasts:
            contrasts_by_new[c.new_case_id].append(c)

        groups: Dict[Tuple[str, str], List[SuccessCase]] = defaultdict(list)
        for case in successes:
            key = (case.task_context, "|".join(case.critical_subgraph[:2]))
            groups[key].append(case)

        patterns: List[SuccessPattern] = []
        for idx, ((context, key), members) in enumerate(groups.items(), start=1):
            support_cases = [m.case_id for m in members]
            related_contrasts = []
            gains = []
            costs = []
            edit_types = []
            verifier_keys = Counter()
            for m in members:
                for k, v in m.verifier_outputs.items():
                    if v:
                        verifier_keys[k] += 1
                for c in contrasts_by_new.get(m.case_id, []):
                    related_contrasts.append(c.contrast_id)
                    gains.append(c.delta_score)
                    costs.append(c.delta_cost)
                    edit_types.append(c.edit_primitive)

            effective_edit = Counter(edit_types).most_common(1)[0][0] if edit_types else "unknown"
            expected_gain = sum(gains) / max(len(gains), 1) if gains else 0.01
            cost_profile = {
                "mean_delta_cost": sum(costs) / max(len(costs), 1) if costs else 0.0,
                "n": float(len(costs)),
            }
            confidence = min(0.98, 0.4 + 0.12 * min(len(members), 4) + 0.06 * min(len(related_contrasts), 4))

            patterns.append(
                SuccessPattern(
                    pattern_id=f"sp_{idx}",
                    applicable_context=context,
                    effective_subgraph=key.split("|") if key else [],
                    ordering_constraint="preserve_topological_order_of_existing_blocks",
                    required_verifiers=[k for k, _ in verifier_keys.most_common(3)],
                    effective_edit_pattern=effective_edit,
                    expected_gain=expected_gain,
                    cost_profile=cost_profile,
                    support_cases=support_cases,
                    support_contrasts=sorted(set(related_contrasts)),
                    confidence=confidence,
                )
            )
        return patterns
    def _induce_edit_rules(
        self,
        signatures: List[FailureSignature],
        patterns: List[SuccessPattern],
        contrasts: List[ContrastRecord],
    ) -> List[EditRule]:
        by_failure_resolve: Dict[str, List[ContrastRecord]] = defaultdict(list)
        for c in contrasts:
            if c.resolved_failure_type:
                by_failure_resolve[c.resolved_failure_type].append(c)

        pattern_by_context: Dict[str, List[SuccessPattern]] = defaultdict(list)
        for p in patterns:
            pattern_by_context[p.applicable_context].append(p)

        rules: List[EditRule] = []
        for idx, sig in enumerate(signatures, start=1):
            evidence = by_failure_resolve.get(sig.failure_type, [])
            if evidence:
                best = sorted(evidence, key=lambda x: (x.delta_score, x.minimality_estimate), reverse=True)[0]
                candidate_edit = best.edit_primitive
                expected_delta_score = sum(e.delta_score for e in evidence) / len(evidence)
                expected_delta_cost = sum(e.delta_cost for e in evidence) / len(evidence)
                neg = len([e for e in evidence if e.delta_score <= 0])
                risk = neg / max(len(evidence), 1)
            else:
                context_patterns = pattern_by_context.get(sig.applicable_context, [])
                candidate_edit = context_patterns[0].effective_edit_pattern if context_patterns else "verification_patch"
                expected_delta_score = context_patterns[0].expected_gain if context_patterns else 0.01
                expected_delta_cost = context_patterns[0].cost_profile.get("mean_delta_cost", 0.0) if context_patterns else 0.0
                risk = 0.45

            conf = min(0.98, 0.35 + 0.3 * sig.confidence + 0.35 * max(0.0, 1.0 - risk))
            rules.append(
                EditRule(
                    rule_id=f"er_{idx}",
                    precondition=sig.trigger_condition,
                    target_failure_signature=sig.signature_id,
                    candidate_edit=candidate_edit,
                    expected_delta_score=expected_delta_score,
                    expected_delta_cost=expected_delta_cost,
                    risk_of_regression=risk,
                    requires_guardrail=(risk >= 0.35 or candidate_edit in sig.unsafe_edits),
                    confidence=conf,
                )
            )
        return rules

    def _guardrails_for_failure(self, failure_type: str) -> List[str]:
        if failure_type == "verification_missing":
            return ["require_explicit_verification_step", "reject_unverified_final_answer"]
        if failure_type == "retrieval_error":
            return ["require_retrieval_before_answer", "enforce_evidence_trace"]
        if failure_type == "tool_usage_error":
            return ["validate_tool_io_contracts", "fallback_on_tool_exception"]
        return ["enforce_single_local_edit", "run_lightweight_sanity_check"]


class MemoryGovernance:
    """Simple governance to prevent unbounded noisy growth."""

    def __init__(self, min_support: int = 2, max_patterns: int = 80):
        self.min_support = max(1, min_support)
        self.max_patterns = max_patterns

    def apply(self, artifacts: MemoryArtifacts) -> MemoryArtifacts:
        artifacts.failure_signatures = self._dedupe_failure_signatures(artifacts.failure_signatures)
        artifacts.success_patterns = self._dedupe_success_patterns(artifacts.success_patterns)
        artifacts.edit_rules = self._dedupe_edit_rules(artifacts.edit_rules)

        artifacts.failure_signatures = [
            s for s in artifacts.failure_signatures if len(s.support_cases) >= self.min_support or s.confidence >= 0.72
        ][: self.max_patterns]
        artifacts.success_patterns = [
            p for p in artifacts.success_patterns if len(p.support_cases) >= self.min_support or p.confidence >= 0.72
        ][: self.max_patterns]
        artifacts.edit_rules = sorted(artifacts.edit_rules, key=lambda r: r.confidence, reverse=True)[: self.max_patterns]
        return artifacts

    def _dedupe_failure_signatures(self, rows: List[FailureSignature]) -> List[FailureSignature]:
        keep: Dict[Tuple[str, str, str], FailureSignature] = {}
        for row in rows:
            key = (row.applicable_context, row.failure_type, row.first_bad_node_type)
            prev = keep.get(key)
            if prev is None or row.confidence > prev.confidence:
                keep[key] = row
        return list(keep.values())

    def _dedupe_success_patterns(self, rows: List[SuccessPattern]) -> List[SuccessPattern]:
        keep: Dict[Tuple[str, str], SuccessPattern] = {}
        for row in rows:
            key = (row.applicable_context, "|".join(row.effective_subgraph))
            prev = keep.get(key)
            if prev is None or row.confidence > prev.confidence:
                keep[key] = row
        return list(keep.values())

    def _dedupe_edit_rules(self, rows: List[EditRule]) -> List[EditRule]:
        keep: Dict[Tuple[str, str], EditRule] = {}
        for row in rows:
            key = (row.target_failure_signature, row.candidate_edit)
            prev = keep.get(key)
            if prev is None or row.confidence > prev.confidence:
                keep[key] = row
        return list(keep.values())


class FailureAwareEditor:
    """Retrieve distilled memory artifacts and propose guarded minimal edits."""

    def __init__(self, artifacts: MemoryArtifacts):
        self.artifacts = artifacts
        self.attributor = RuleBasedBlameAttributor()

    def propose_minimal_edit(
        self,
        task_context: str,
        workflow_graph: str,
        execution_feedback: Dict[str, Any],
        max_candidates: int = 3,
    ) -> Dict[str, Any]:
        localized = self.attributor.attribute(
            parent_workflow=workflow_graph,
            execution_feedback=execution_feedback,
            edit_text="",
            child_workflow="",
        )

        signatures = [
            s
            for s in self.artifacts.failure_signatures
            if (s.failure_type == localized.blame_type or localized.blame_type == "unknown")
            and (s.applicable_context == task_context or s.applicable_context == "unknown")
        ]
        if not signatures:
            signatures = sorted(self.artifacts.failure_signatures, key=lambda x: x.confidence, reverse=True)[:2]

        rules = self._collect_candidate_rules(signatures)
        patterns = [p for p in self.artifacts.success_patterns if p.applicable_context == task_context]
        favored_edits = {p.effective_edit_pattern for p in patterns}

        candidates = []
        for rule in rules:
            target_sig = next((s for s in signatures if s.signature_id == rule.target_failure_signature), None)
            if target_sig is None:
                continue
            veto_reason = self._veto_reason(rule, target_sig)
            score = rule.expected_delta_score - 0.35 * rule.risk_of_regression - 0.1 * abs(rule.expected_delta_cost)
            if rule.candidate_edit in favored_edits:
                score += 0.03
            candidates.append(
                {
                    "rule_id": rule.rule_id,
                    "candidate_edit": rule.candidate_edit,
                    "expected_delta_score": rule.expected_delta_score,
                    "expected_delta_cost": rule.expected_delta_cost,
                    "risk_of_regression": rule.risk_of_regression,
                    "requires_guardrail": rule.requires_guardrail,
                    "guardrails": target_sig.recommended_guardrails,
                    "target_failure_signature": target_sig.signature_id,
                    "localized_failure_type": localized.blame_type,
                    "localized_failure_node": localized.target_block,
                    "vetoed": bool(veto_reason),
                    "veto_reason": veto_reason,
                    "rank_score": score,
                }
            )

        non_vetoed = [c for c in candidates if not c["vetoed"]]
        ranked = sorted(non_vetoed or candidates, key=lambda x: x["rank_score"], reverse=True)
        if not ranked:
            return {
                "localized_failure_type": localized.blame_type,
                "localized_failure_node": localized.target_block,
                "selected_edit": {
                    "candidate_edit": "verification_patch",
                    "requires_guardrail": True,
                    "guardrails": ["enforce_single_local_edit", "run_lightweight_sanity_check"],
                },
                "alternatives": [],
            }

        return {
            "localized_failure_type": localized.blame_type,
            "localized_failure_node": localized.target_block,
            "selected_edit": ranked[0],
            "alternatives": ranked[1:max_candidates],
        }

    def _collect_candidate_rules(self, signatures: List[FailureSignature]) -> List[EditRule]:
        sig_ids = {s.signature_id for s in signatures}
        matched = [r for r in self.artifacts.edit_rules if r.target_failure_signature in sig_ids]
        if matched:
            return sorted(matched, key=lambda r: r.confidence, reverse=True)[:8]
        return sorted(self.artifacts.edit_rules, key=lambda r: r.confidence, reverse=True)[:5]

    def _veto_reason(self, rule: EditRule, signature: FailureSignature) -> str:
        if rule.candidate_edit in signature.unsafe_edits:
            return "edit_primitive_is_marked_unsafe_for_failure_signature"
        if rule.risk_of_regression > 0.65:
            return "risk_of_regression_above_threshold"
        return ""


class TrajectoryMemoryController:
    """High-level write/distill/use orchestration for optimizer integration."""

    def __init__(self, min_support: int = 2, max_patterns: int = 80):
        self.compiler = TrajectoryCompiler()
        self.distiller = PatternDistiller()
        self.governance = MemoryGovernance(min_support=min_support, max_patterns=max_patterns)
        self.artifacts = MemoryArtifacts()

    def refresh(self, workflows_dir: str, task_context: Optional[str] = None) -> MemoryArtifacts:
        artifacts = self.compiler.compile(workflows_dir=workflows_dir, task_context=task_context)
        artifacts = self.distiller.distill(artifacts)
        artifacts = self.governance.apply(artifacts)
        self.artifacts = artifacts
        self._persist(workflows_dir, artifacts)
        return artifacts

    def suggest_for_workflow(
        self,
        task_context: str,
        workflow_graph: str,
        execution_feedback: Dict[str, Any],
    ) -> Dict[str, Any]:
        editor = FailureAwareEditor(self.artifacts)
        return editor.propose_minimal_edit(
            task_context=task_context,
            workflow_graph=workflow_graph,
            execution_feedback=execution_feedback,
        )

    def _persist(self, workflows_dir: str, artifacts: MemoryArtifacts) -> None:
        path = Path(workflows_dir)
        if (path / "workflows").exists():
            path = path / "workflows"
        memory_dir = path / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(memory_dir / "success_cases.json", [asdict(x) for x in artifacts.success_cases])
        self._write_json(memory_dir / "failure_cases.json", [asdict(x) for x in artifacts.failure_cases])
        self._write_json(memory_dir / "contrast_records.json", [asdict(x) for x in artifacts.contrast_records])
        self._write_json(memory_dir / "failure_signatures.json", [asdict(x) for x in artifacts.failure_signatures])
        self._write_json(memory_dir / "success_patterns.json", [asdict(x) for x in artifacts.success_patterns])
        self._write_json(memory_dir / "edit_rules.json", [asdict(x) for x in artifacts.edit_rules])

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
