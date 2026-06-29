import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from contrastive_experience.attribution_prompt import build_attribution_prompt
from contrastive_experience.divergence import compute_operator_divergence
from contrastive_experience.grouping import group_traces, load_trace_records
from contrastive_experience.localization import localize_bottleneck
from contrastive_experience.prototypes import select_prototypes


def diagnose_trace_file(trace_file: Union[str, Path], success_threshold: Optional[float] = None) -> Dict[str, Any]:
    records = load_trace_records(trace_file)
    grouped = group_traces(records, success_threshold=success_threshold)
    divergence = compute_operator_divergence(grouped["success_traces"], grouped["failure_traces"])
    localization = localize_bottleneck(divergence)
    candidate_operator = localization.get("candidate_operator")
    prototypes = (
        select_prototypes(grouped["success_traces"], grouped["failure_traces"], candidate_operator)
        if candidate_operator
        else {}
    )
    prompt = build_attribution_prompt(localization.get("candidate", {}), prototypes)
    return {
        "stats": grouped["stats"],
        "operator_divergence": divergence,
        "localization": localization,
        "prototypes": prototypes,
        "attribution_prompt": prompt,
    }


def write_diagnosis_outputs(
    trace_file: Union[str, Path],
    report: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    output_path = Path(output_dir) if output_dir else Path(trace_file).parent
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "divergence_report.json").open("w", encoding="utf-8") as fout:
        json.dump(
            {key: value for key, value in report.items() if key != "attribution_prompt"},
            fout,
            ensure_ascii=False,
            indent=2,
        )
    with (output_path / "prototypes.json").open("w", encoding="utf-8") as fout:
        json.dump(report.get("prototypes", {}), fout, ensure_ascii=False, indent=2)
    with (output_path / "attribution_prompt.txt").open("w", encoding="utf-8") as fout:
        fout.write(report.get("attribution_prompt", ""))
