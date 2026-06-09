import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.blame_attributor import RuleBasedBlameAttributor
from models.workflow_revision_model import WorkflowRevisionModel


class RevisionPriorController:
    """Bridge revision model into AFlow search with minimal intrusion."""

    def __init__(self, model_dir: str, mode: str = "generate_then_search", rank_candidates: int = 3):
        self.model = WorkflowRevisionModel.load(model_dir)
        self.mode = mode
        self.rank_candidates = max(1, rank_candidates)
        self.blame_attributor = RuleBasedBlameAttributor()

    def build_context(
        self,
        dataset: str,
        parent_round: int,
        parent_workflow: str,
        parent_score: float,
        log_data: str,
        task_summary: str = "",
    ) -> Dict[str, Any]:
        execution_feedback = {
            "score_mean": parent_score,
            "score_std": 0.0,
            "failure_summary": log_data[:400],
            "trace_summary": log_data[:200],
        }
        blame = self.blame_attributor.attribute(
            parent_workflow=parent_workflow,
            execution_feedback=execution_feedback,
            edit_text="",
            child_workflow="",
        )
        return {
            "task_family": dataset,
            "task_summary": task_summary or dataset,
            "parent_id": str(parent_round),
            "parent_workflow": parent_workflow,
            "parent_score": parent_score,
            "execution_feedback": execution_feedback,
            "blame": {
                "target_block": blame.target_block,
                "blame_type": blame.blame_type,
                "blame_confidence": blame.blame_confidence,
                "blame_rationale": blame.blame_rationale,
            },
        }

    def generate_prior_edit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.model.generate_proposal(context)

    def score_edit(self, context: Dict[str, Any], edit_text: str) -> float:
        payload = {
            "edit_type": "unknown",
            "target_block": (context.get("blame") or {}).get("target_block", "b0"),
            "edit_text": edit_text,
        }
        return self.model.score_candidate(context, payload)

    def augment_prompt(self, prompt: str, prior_edit: Dict[str, Any]) -> str:
        block = (
            "\n\n[Amortized Edit Prior]\n"
            "Use the following prior as a strong but not mandatory hint for next local modification:\n"
            f"- edit_type: {prior_edit.get('edit_type', 'unknown')}\n"
            f"- target_block: {prior_edit.get('target_block', 'b0')}\n"
            f"- edit_text: {prior_edit.get('edit_text', '')}\n"
            f"- expected_benefit: {prior_edit.get('expected_benefit', '')}\n"
            "You may refine this suggestion if needed, but keep one local modification.\n"
        )
        return prompt + block


def append_prior_log(directory: str, payload: Dict[str, Any]) -> None:
    log_path = Path(directory) / "revision_prior_log.json"
    records: List[Dict[str, Any]] = []
    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            records = json.loads(log_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                records = []
        except Exception:
            records = []
    records.append(payload)
    log_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def load_prior_usage_logs(workflows_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in Path(workflows_dir).glob("round_*/revision_prior_log.json"):
        if path.stat().st_size == 0:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, list):
                records.extend(payload)
        except Exception:
            continue
    return records
