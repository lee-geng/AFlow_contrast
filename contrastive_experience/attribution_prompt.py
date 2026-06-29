import json
from typing import Any, Dict, List, Optional


def build_attribution_prompt(
    candidate: Dict[str, Any],
    prototypes: Dict[str, Any],
    experience: Optional[Dict[str, Any]] = None,
    forbidden_edits: Optional[List[str]] = None,
) -> str:
    payload = {
        "candidate_bottleneck": candidate,
        "success_prototypes": prototypes.get("success_prototypes", []),
        "failure_prototypes": prototypes.get("failure_prototypes", []),
        "matching_criteria": prototypes.get("matching_criteria", []),
        "relevant_experience": experience or {},
        "forbidden_edits": forbidden_edits or [],
        "required_success_guard": "Any edit must be checked against the selected success prototypes before acceptance.",
    }
    return (
        "You are diagnosing an AFlow workflow using success/failure contrastive operator traces.\n"
        "Use the candidate bottleneck language; do not claim root cause until edit verification succeeds.\n"
        "Recommend one scoped workflow edit that repairs the failure pattern while preserving the success pattern.\n\n"
        f"TRACE_DIAGNOSIS_CONTEXT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return strict JSON with this schema:\n"
        "{\n"
        '  "diagnosis": "...",\n'
        '  "recommended_edit_level": "prompt | operator | graph | control",\n'
        '  "recommended_edits": ["..."],\n'
        '  "forbidden_edits": ["..."],\n'
        '  "success_guard": "...",\n'
        '  "verification_plan": {\n'
        '    "target_failure_samples": ["..."],\n'
        '    "success_guard_samples": ["..."]\n'
        "  }\n"
        "}"
    )
