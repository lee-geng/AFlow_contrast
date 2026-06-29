import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEXT_RUN = PROJECT_ROOT / "commands" / "next_run.json"


def main() -> int:
    args = parse_args()
    workflow_dir = args.workflow_dir or latest_workflow_dir(args.dataset, args.history_dir)
    if not workflow_dir:
        raise SystemExit("No workflow_dir provided and no latest compiled workflow was found in history.")

    run_id = args.run_id or f"{args.dataset.lower()}_validation_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    request = build_request(args, workflow_dir, run_id)
    write_json(Path(args.write_to), request)
    print(f"Wrote {Path(args.write_to).resolve()}")
    print(f"run_id={run_id}")
    print(f"workflow_dir={workflow_dir}")
    return 0


def build_request(args: argparse.Namespace, workflow_dir: str, run_id: str) -> Dict[str, Any]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "-B",
        "-m",
        "patch_evolution.validation_loop",
        "--dataset",
        args.dataset,
        "--workflow_dir",
        workflow_dir,
        "--model_name",
        args.model_name,
        "--trace_dir",
        args.trace_dir,
        "--patch_registry_dir",
        args.patch_registry_dir,
        "--output_dir",
        args.output_dir,
        "--compiled_workflow_dir",
        args.compiled_workflow_dir,
        "--history_dir",
        args.history_dir,
        "--run_id",
        run_id,
        "--validation_iterations",
        str(args.validation_iterations),
        "--repair_rounds_per_iteration",
        str(args.repair_rounds_per_iteration),
        "--compile_max_patches",
        str(args.compile_max_patches),
        "--content_repair_batches",
        args.content_repair_batches,
        "--max_samples",
        str(args.max_samples),
        "--llm_generate_patches",
    ]
    if args.llm_patch_max_candidates is not None:
        command.extend(["--llm_patch_max_candidates", str(args.llm_patch_max_candidates)])
    if args.indices:
        command.extend(["--indices", args.indices])
    if args.start_index:
        command.extend(["--start_index", str(args.start_index)])
    if args.compiled_eval_max_samples is not None:
        command.extend(["--compiled_eval_max_samples", str(args.compiled_eval_max_samples)])
    if args.no_online_patch:
        command.append("--no_online_patch")
    if args.stop_on_no_improvement:
        command.append("--stop_on_no_improvement")
        command.extend(["--patience", str(args.patience)])
    if args.stop_when_no_adoption:
        command.append("--stop_when_no_adoption")
    if args.no_progress:
        command.append("--no_progress")

    request: Dict[str, Any] = {
        "run_id": run_id,
        "enabled": True,
        "dataset": args.dataset,
        "cwd": ".",
        "timeout_seconds": args.timeout_seconds,
        "command": command,
        "metrics_paths": [
            f"{args.output_dir}\\{args.dataset}\\{run_id}\\loop_summary.json",
            f"{args.history_dir}\\{args.dataset}\\latest.json",
        ],
    }
    if not args.no_codex_followup:
        request["codex_after_run"] = {
            "enabled": True,
            "require_followup": True,
            "command": ["codex", "exec", "--cd", "{repo_root}", "{prompt}"],
            "prompt": build_codex_prompt(args, run_id),
            "stdout_tail_chars": 8000,
            "stderr_tail_chars": 8000,
        }
    return request


def build_codex_prompt(args: argparse.Namespace, run_id: str) -> str:
    return (
        "This is an automated validation workflow optimization loop, not a one-off analysis.\n"
        f"Read commands/runs/{run_id}/run_status.json, metrics.json, stdout.txt, stderr.txt, "
        f"{args.output_dir}/{args.dataset}/{run_id}/loop_summary.json, and "
        f"{args.history_dir}/{args.dataset}/latest_context.md.\n"
        "Goal: improve the explicit workflow on the validation set step by step.\n"
        "If validation still has failures or a regression, inspect the failure traces and code, then modify the "
        "workflow patch/node/compiler logic as needed and write a new commands/next_run.json with a fresh run_id "
        "for another validation_loop run.\n"
        "If validation is repaired or no safe useful edit remains, write commands/loop_done.json with dataset, "
        "run_id, final_workflow_dir, score, failure_count, and stop_reason.\n"
        "Do not silently stop after analysis; either queue the next validation run or write loop_done.json."
    )


def latest_workflow_dir(dataset: str, history_dir: str) -> Optional[str]:
    latest_path = PROJECT_ROOT / history_dir / dataset / "latest.json"
    if not latest_path.exists():
        return None
    try:
        data = json.loads(latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    workflow = data.get("compiled_workflow") or {}
    compiled = workflow.get("compiled_workflow_dir")
    if compiled:
        return str(compiled)
    return data.get("workflow_dir")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp = resolved.with_suffix(resolved.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write commands/next_run.json for validation workflow evolution.")
    parser.add_argument("--dataset", default="DROP")
    parser.add_argument("--workflow_dir", default=None)
    parser.add_argument("--model_name", default="qwen3-32b-fp8")
    parser.add_argument("--conda_env", default="aflow")
    parser.add_argument("--trace_dir", default="traces_validation_loop_qwen3_fp8")
    parser.add_argument("--patch_registry_dir", default="results_validation_loop_qwen3_fp8")
    parser.add_argument("--output_dir", default="repair_runs_validation_loop_qwen3_fp8")
    parser.add_argument("--compiled_workflow_dir", default="compiled_workflows_validation_loop_qwen3_fp8")
    parser.add_argument("--history_dir", default="experiment_history")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--validation_iterations", type=int, default=3)
    parser.add_argument("--repair_rounds_per_iteration", type=int, default=1)
    parser.add_argument("--compile_max_patches", type=int, default=8)
    parser.add_argument("--content_repair_batches", default="all")
    parser.add_argument("--compiled_eval_max_samples", type=int, default=None)
    parser.add_argument("--llm_patch_max_candidates", type=int, default=1)
    parser.add_argument("--timeout_seconds", type=int, default=14400)
    parser.add_argument("--no_online_patch", action="store_true")
    parser.add_argument("--stop_on_no_improvement", action="store_true")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--stop_when_no_adoption", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--no_codex_followup", action="store_true")
    parser.add_argument("--write_to", default=str(DEFAULT_NEXT_RUN))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
