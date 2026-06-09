import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


BLAME_TYPES = {
    "planning_error",
    "retrieval_error",
    "tool_usage_error",
    "verification_missing",
    "patch_incomplete",
    "unknown",
}


@dataclass
class BlameSignal:
    target_block: str
    blame_type: str
    blame_confidence: float
    blame_rationale: str


class WorkflowBlockView:
    """A minimal block view over workflow code/operator sequence."""

    BLOCK_PATTERN = re.compile(r"await\s+self\.(\w+)\(")

    def extract(self, workflow_text: str) -> List[str]:
        blocks = self.BLOCK_PATTERN.findall(workflow_text)
        if not blocks:
            return ["b0"]
        return [f"b{i + 1}:{name}" for i, name in enumerate(blocks)]


class RuleBasedBlameAttributor:
    def __init__(self):
        self.block_view = WorkflowBlockView()

    def attribute(
        self,
        parent_workflow: str,
        execution_feedback: Dict[str, Any],
        edit_text: str,
        child_workflow: str = "",
    ) -> BlameSignal:
        failure_summary = str(execution_feedback.get("failure_summary") or "").lower()
        trace_summary = str(execution_feedback.get("trace_summary") or "").lower()
        merged = " ".join([failure_summary, trace_summary, edit_text.lower()])

        blocks = self.block_view.extract(parent_workflow)
        target_block = self._infer_target_block(blocks, merged, edit_text)

        blame_type, confidence, rationale = self._infer_blame_type(merged, parent_workflow, child_workflow)
        return BlameSignal(
            target_block=target_block,
            blame_type=blame_type,
            blame_confidence=confidence,
            blame_rationale=rationale,
        )

    def _infer_target_block(self, blocks: List[str], merged: str, edit_text: str) -> str:
        m = re.search(r"b(\d+)", edit_text.lower())
        if m:
            return f"b{m.group(1)}"

        keyword_map = {
            "retriev": "retrieve",
            "tool": "tool",
            "verify": "verify",
            "plan": "plan",
            "test": "test",
            "review": "review",
        }
        for key, block_hint in keyword_map.items():
            if key in merged:
                for block in blocks:
                    if block_hint in block.lower():
                        return block.split(":", 1)[0]

        return blocks[0].split(":", 1)[0] if blocks else "b0"

    def _infer_blame_type(self, merged: str, parent_workflow: str, child_workflow: str) -> tuple[str, float, str]:
        if any(k in merged for k in ["verify", "unsupported", "hallucin", "evidence"]):
            return (
                "verification_missing",
                0.78,
                "The feedback suggests missing evidence validation before final output.",
            )
        if any(k in merged for k in ["retriev", "context missing", "document", "knowledge"]):
            return (
                "retrieval_error",
                0.75,
                "The failure likely stems from weak retrieval quality or missing relevant context.",
            )
        if any(k in merged for k in ["tool", "api", "exception", "traceback", "runtime"]):
            return (
                "tool_usage_error",
                0.74,
                "Execution traces indicate incorrect or unstable tool invocation.",
            )
        if any(k in merged for k in ["plan", "decompose", "step", "reasoning"]):
            return (
                "planning_error",
                0.7,
                "The workflow appears to under-specify planning/decomposition steps.",
            )
        if child_workflow and parent_workflow and self._is_small_patch(parent_workflow, child_workflow):
            return (
                "patch_incomplete",
                0.69,
                "Observed edit scope is too narrow to address the reported failure mode.",
            )

        return (
            "unknown",
            0.45,
            "Insufficient failure evidence for a specific blame category.",
        )

    def _is_small_patch(self, parent: str, child: str) -> bool:
        parent_compact = re.sub(r"\s+", " ", parent)
        child_compact = re.sub(r"\s+", " ", child)
        if not parent_compact:
            return False
        overlap = sum(1 for i, ch in enumerate(child_compact[: len(parent_compact)]) if parent_compact[i] == ch)
        ratio = overlap / max(len(parent_compact), 1)
        return ratio > 0.97


class LLMBlameAttributor:
    """Optional interface for future LLM-based blame attribution."""

    def __init__(self, llm_callable):
        self.llm_callable = llm_callable

    async def attribute(self, prompt_payload: Dict[str, Any]) -> BlameSignal:
        raw = await self.llm_callable(prompt_payload)
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            bt = data.get("blame_type", "unknown")
            if bt not in BLAME_TYPES:
                bt = "unknown"
            return BlameSignal(
                target_block=str(data.get("target_block", "b0")),
                blame_type=bt,
                blame_confidence=float(data.get("blame_confidence", 0.5)),
                blame_rationale=str(data.get("blame_rationale", "")),
            )
        except Exception:
            return BlameSignal("b0", "unknown", 0.4, "LLM attributor parsing failed, fallback to unknown.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JudgeFlow-lite blame attribution")
    parser.add_argument("--parent_workflow", type=str, required=True)
    parser.add_argument("--feedback_json", type=str, default="{}")
    parser.add_argument("--edit_text", type=str, default="")
    parser.add_argument("--child_workflow", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feedback = json.loads(args.feedback_json)
    attributor = RuleBasedBlameAttributor()
    signal = attributor.attribute(
        parent_workflow=args.parent_workflow,
        execution_feedback=feedback,
        edit_text=args.edit_text,
        child_workflow=args.child_workflow,
    )
    print(json.dumps(asdict(signal), ensure_ascii=False))


if __name__ == "__main__":
    main()
