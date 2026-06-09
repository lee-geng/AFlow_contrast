import json
import re
import shutil
from pathlib import Path


def sanitize_artifact_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unknown_model"


def resolve_optimized_path(
    optimized_path: str,
    opt_model_name: str,
    exec_model_name: str,
    separate_model_artifacts: bool = False,
    artifact_tag: str = "",
) -> str:
    if not separate_model_artifacts:
        return optimized_path

    opt_part = sanitize_artifact_name(opt_model_name)
    exec_part = sanitize_artifact_name(exec_model_name)
    namespace = f"exec__{exec_part}__opt__{opt_part}"
    if artifact_tag:
        namespace = f"{namespace}__{sanitize_artifact_name(artifact_tag)}"
    return str(Path(optimized_path) / namespace)


def write_run_metadata(
    optimized_path: str,
    dataset: str,
    opt_model_name: str,
    exec_model_name: str,
    artifact_tag: str = "",
) -> None:
    dataset_root = Path(optimized_path) / dataset
    workflows_root = dataset_root / "workflows"
    workflows_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "optimized_path": str(Path(optimized_path)),
        "dataset_root": str(dataset_root),
        "opt_model_name": opt_model_name,
        "exec_model_name": exec_model_name,
        "artifact_tag": artifact_tag,
    }
    (dataset_root / "run_metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dataset_template_initialized(
    optimized_path: str,
    dataset: str,
    template_root: str = "workspace_template_ab",
) -> str:
    """
    Ensure a fresh experiment workspace contains the dataset seed workflow/template.

    The current AFlow implementation starts from an existing round_1 workflow rather
    than synthesizing the whole workspace from an empty directory. For new artifact
    namespaces, copy the dataset template into place automatically.
    """
    target_dataset_root = Path(optimized_path) / dataset
    target_workflows_root = target_dataset_root / "workflows"
    target_round_graph = target_workflows_root / "round_1" / "graph.py"
    if target_round_graph.exists():
        return str(target_dataset_root)

    source_dataset_root = Path(template_root) / dataset
    source_workflows_root = source_dataset_root / "workflows"
    source_round_graph = source_workflows_root / "round_1" / "graph.py"
    if not source_round_graph.exists():
        raise FileNotFoundError(
            f"Dataset template not found for {dataset}: expected seed workflow at {source_round_graph}"
        )

    target_dataset_root.mkdir(parents=True, exist_ok=True)
    if not target_round_graph.exists():
        shutil.copytree(source_dataset_root, target_dataset_root, dirs_exist_ok=True)
    _rewrite_dataset_graph_imports(target_dataset_root)
    return str(target_dataset_root)


def _rewrite_dataset_graph_imports(dataset_root: Path) -> None:
    workflows_root = dataset_root / "workflows"
    if not workflows_root.exists():
        return
    for graph_path in workflows_root.glob("round_*/graph.py"):
        text = graph_path.read_text(encoding="utf-8")
        updated = text
        updated = re.sub(
            r"import [A-Za-z0-9_\.]+\.workflows\.template\.operator as operator",
            "from ..template import operator",
            updated,
        )
        updated = re.sub(
            r"import [A-Za-z0-9_\.]+\.workflows\.round_\d+\.prompt as prompt_custom",
            "from . import prompt as prompt_custom",
            updated,
        )
        if updated != text:
            graph_path.write_text(updated, encoding="utf-8")

    template_operator = workflows_root / "template" / "operator.py"
    if template_operator.exists():
        text = template_operator.read_text(encoding="utf-8")
        updated = re.sub(
            r"from [A-Za-z0-9_\.]+\.workflows\.template\.operator_an import \*",
            "from .operator_an import *",
            text,
        )
        updated = re.sub(
            r"from [A-Za-z0-9_\.]+\.workflows\.template\.op_prompt import \*",
            "from .op_prompt import *",
            updated,
        )
        if updated != text:
            template_operator.write_text(updated, encoding="utf-8")
