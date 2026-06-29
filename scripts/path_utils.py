import re
import shutil
from pathlib import Path


def reset_file(file_path: Path, default_content: str = "") -> None:
    if not file_path.exists():
        return

    try:
        file_path.chmod(0o666)
    except OSError:
        pass

    try:
        file_path.unlink()
        return
    except OSError:
        pass

    try:
        file_path.write_text(default_content, encoding="utf-8")
    except OSError:
        pass


def sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", model_name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "model"
    if sanitized[0].isdigit():
        sanitized = f"model_{sanitized}"
    return sanitized


def get_model_run_name(opt_model_name: str, exec_model_name: str) -> str:
    opt_name = sanitize_model_name(opt_model_name)
    exec_name = sanitize_model_name(exec_model_name)
    return f"opt_{opt_name}__exec_{exec_name}"


def get_model_run_root(optimized_path: str, dataset: str, opt_model_name: str, exec_model_name: str) -> Path:
    run_name = get_model_run_name(opt_model_name, exec_model_name)
    return Path(optimized_path) / "model_runs" / dataset / run_name


def get_model_run_module(optimized_path: str, dataset: str, opt_model_name: str, exec_model_name: str) -> str:
    run_root = get_model_run_root(optimized_path, dataset, opt_model_name, exec_model_name)
    module_parts = list(run_root.parts)
    if run_root.is_absolute():
        raise ValueError("optimized_path must be a relative project path so the workflow package can be imported")
    return ".".join(module_parts).replace("-", "_")


def prepare_model_workspace(optimized_path: str, dataset: str, opt_model_name: str, exec_model_name: str) -> tuple[Path, str]:
    source_root = Path(optimized_path) / dataset
    if not source_root.exists():
        default_source_root = Path("workspace") / dataset
        if default_source_root.exists():
            source_root = default_source_root
        else:
            raise FileNotFoundError(
                f"Base workspace template not found: {source_root} or {default_source_root}. "
                "Please make sure the initial workspace has been downloaded or created."
            )

    run_root = get_model_run_root(optimized_path, dataset, opt_model_name, exec_model_name)
    module_root = get_model_run_module(optimized_path, dataset, opt_model_name, exec_model_name)
    workflows_root = run_root / "workflows"
    template_src = source_root / "workflows" / "template"
    round_src = source_root / "workflows" / "round_1"

    if not template_src.exists() or not round_src.exists():
        raise FileNotFoundError(
            f"Base workflow template is incomplete under {source_root}. "
            "Expected both workflows/template and workflows/round_1."
        )

    if not workflows_root.exists():
        Path(optimized_path).mkdir(parents=True, exist_ok=True)

        package_dirs = [
            Path(optimized_path),
            Path(optimized_path) / "model_runs",
            Path(optimized_path) / "model_runs" / dataset,
            run_root,
            workflows_root,
        ]
        for package_dir in package_dirs:
            package_dir.mkdir(parents=True, exist_ok=True)
            init_file = package_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text("", encoding="utf-8")

        shutil.copytree(template_src, workflows_root / "template", dirs_exist_ok=True)
        shutil.copytree(round_src, workflows_root / "round_1", dirs_exist_ok=True)

        for package_dir in [workflows_root / "template", workflows_root / "round_1"]:
            init_file = package_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text("", encoding="utf-8")

        reset_file(workflows_root / "round_1" / "log.json", "[]")
        reset_file(workflows_root / "results.json", "[]")
        reset_file(workflows_root / "processed_experience.json", "{}")

        old_prefix = f"workspace.{dataset}.workflows"
        new_prefix = f"{module_root}.workflows"
        for py_file in workflows_root.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            content = content.replace(old_prefix, new_prefix)
            py_file.write_text(content, encoding="utf-8")

    return run_root, module_root
