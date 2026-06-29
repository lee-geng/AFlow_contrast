import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMAND_FILE = PROJECT_ROOT / "commands" / "next_run.json"
DEFAULT_STATE_FILE = PROJECT_ROOT / "commands" / "runner_state.json"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "commands" / "runs"


def main() -> int:
    args = parse_args()
    runner = AutoRunner(
        command_file=Path(args.command_file),
        state_file=Path(args.state_file),
        runs_dir=Path(args.runs_dir),
        poll_seconds=args.poll_seconds,
        once=args.once,
        rerun_current=args.rerun_current,
        enable_codex=args.enable_codex,
        codex_command=parse_optional_json_list(args.codex_command),
    )
    return runner.run()


class AutoRunner:
    def __init__(
        self,
        command_file: Path,
        state_file: Path,
        runs_dir: Path,
        poll_seconds: float = 10.0,
        once: bool = False,
        rerun_current: bool = False,
        enable_codex: bool = False,
        codex_command: Optional[List[str]] = None,
    ):
        self.command_file = resolve_under_project(command_file)
        self.state_file = resolve_under_project(state_file)
        self.runs_dir = resolve_under_project(runs_dir)
        self.poll_seconds = max(1.0, poll_seconds)
        self.once = once
        self.enable_codex = enable_codex
        self.codex_command = codex_command
        self.last_fingerprint = None if rerun_current else self._load_last_fingerprint()

    def run(self) -> int:
        self.command_file.parent.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        print(f"[auto_runner] watching {self.command_file}")
        while True:
            try:
                handled = self._tick()
            except Exception as exc:
                self._write_runner_error(exc)
                print(f"[auto_runner] error: {exc}", file=sys.stderr)
                handled = False
            if self.once:
                return 0 if handled else 1
            time.sleep(self.poll_seconds)

    def _tick(self) -> bool:
        if not self.command_file.exists():
            return False
        raw = self.command_file.read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        if fingerprint == self.last_fingerprint:
            return False
        request = json.loads(raw.decode("utf-8"))
        if request.get("enabled") is False:
            self._save_state({"last_fingerprint": fingerprint, "last_status": "disabled"})
            self.last_fingerprint = fingerprint
            return True

        run_record = self._execute_request(request, fingerprint)
        self._save_state(
            {
                "last_fingerprint": fingerprint,
                "last_status": run_record.get("overall_status") or run_record.get("status"),
                "last_command_status": run_record.get("status"),
                "last_codex_status": (run_record.get("codex_after_run") or {}).get("status"),
                "last_run_id": run_record.get("run_id"),
                "last_run_dir": run_record.get("run_dir"),
                "updated_at": now_iso(),
            }
        )
        self.last_fingerprint = fingerprint
        return True

    def _execute_request(self, request: Dict[str, Any], fingerprint: str) -> Dict[str, Any]:
        run_id = safe_run_id(str(request.get("run_id") or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
        run_dir = resolve_under_project(self.runs_dir / run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        request_path = run_dir / "request.json"
        status_path = run_dir / "run_status.json"
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        metrics_path = run_dir / "metrics.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")

        command, shell = normalize_command(request)
        cwd = resolve_under_project(Path(request.get("cwd") or PROJECT_ROOT))
        timeout_seconds = request.get("timeout_seconds")
        timeout_seconds = int(timeout_seconds) if timeout_seconds is not None else None

        status = {
            "run_id": run_id,
            "status": "running",
            "request_fingerprint": fingerprint,
            "command": command,
            "shell": shell,
            "cwd": str(cwd),
            "run_dir": str(run_dir),
            "started_at": now_iso(),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "metrics_path": str(metrics_path),
        }
        write_json(status_path, status)

        started = time.time()
        returncode = None
        timed_out = False
        error = None
        try:
            returncode = run_subprocess(command, cwd, stdout_path, stderr_path, timeout_seconds, shell=shell)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -9
            error = f"timeout_after_{timeout_seconds}_seconds"
        except Exception as exc:
            returncode = -1
            error = repr(exc)

        finished = time.time()
        status.update(
            {
                "status": "timeout" if timed_out else ("success" if returncode == 0 else "failed"),
                "returncode": returncode,
                "duration_seconds": round(finished - started, 3),
                "finished_at": now_iso(),
                "error": error,
            }
        )
        write_json(status_path, status)

        metrics = collect_metrics(request, status, run_dir)
        write_json(metrics_path, metrics)
        status["metrics_path"] = str(metrics_path)
        write_json(status_path, status)

        codex_status = self._maybe_run_codex(request, run_dir, status_path, metrics_path, stdout_path, stderr_path)
        if codex_status:
            status["codex_after_run"] = codex_status
            status["overall_status"] = (
                status["status"]
                if codex_status.get("status") == "success"
                else f"{status['status']}_codex_{codex_status.get('status', 'unknown')}"
            )
        else:
            status["overall_status"] = status["status"]
        if codex_status:
            write_json(status_path, status)

        return status

    def _maybe_run_codex(
        self,
        request: Dict[str, Any],
        run_dir: Path,
        status_path: Path,
        metrics_path: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> Optional[Dict[str, Any]]:
        codex_cfg = request.get("codex_after_run") or {}
        if not codex_cfg.get("enabled"):
            return None
        if not self.enable_codex:
            return {"status": "skipped", "reason": "runner_started_without_enable_codex"}

        prompt_path = run_dir / "codex_prompt.txt"
        codex_stdout_path = run_dir / "codex_stdout.txt"
        codex_stderr_path = run_dir / "codex_stderr.txt"
        next_run_before = file_sha256(DEFAULT_COMMAND_FILE)
        loop_done_path = PROJECT_ROOT / "commands" / "loop_done.json"
        loop_done_before = file_sha256(loop_done_path)
        prompt = build_codex_prompt(codex_cfg, request, status_path, metrics_path, stdout_path, stderr_path)
        prompt_path.write_text(prompt, encoding="utf-8")

        command_template = codex_cfg.get("command") or self.codex_command or default_codex_command()
        command = render_command_template(command_template, prompt, prompt_path)
        started = time.time()
        try:
            returncode = run_subprocess(command, PROJECT_ROOT, codex_stdout_path, codex_stderr_path, None, shell=False)
            status = "success" if returncode == 0 else "failed"
            error = None
        except Exception as exc:
            returncode = -1
            status = "failed"
            error = repr(exc)
            codex_stderr_path.write_text(
                (
                    f"{error}\n"
                    f"command={json.dumps(command, ensure_ascii=False)}\n"
                    "hint=Codex CLI could not be launched by the runner. "
                    "Verify that `codex exec \"hello\"` works in a normal PowerShell session, "
                    "or set codex_after_run.command to a working Codex CLI executable.\n"
                ),
                encoding="utf-8",
            )
        next_run_after = file_sha256(DEFAULT_COMMAND_FILE)
        loop_done_after = file_sha256(loop_done_path)
        next_run_changed = bool(next_run_after and next_run_after != next_run_before)
        loop_done_changed = bool(loop_done_after and loop_done_after != loop_done_before)
        if (
            status == "success"
            and codex_cfg.get("require_followup")
            and not next_run_changed
            and not loop_done_changed
        ):
            status = "stalled"
            error = (
                "codex_after_run.require_followup was set, but Codex did not update "
                "commands/next_run.json or commands/loop_done.json"
            )
        return {
            "status": status,
            "returncode": returncode,
            "duration_seconds": round(time.time() - started, 3),
            "prompt_path": str(prompt_path),
            "stdout_path": str(codex_stdout_path),
            "stderr_path": str(codex_stderr_path),
            "command": command,
            "next_run_changed": next_run_changed,
            "loop_done_changed": loop_done_changed,
            "next_run_fingerprint_before": next_run_before,
            "next_run_fingerprint_after": next_run_after,
            "loop_done_path": str(loop_done_path),
            "error": error,
        }

    def _load_last_fingerprint(self) -> Optional[str]:
        if not self.state_file.exists():
            return None
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return state.get("last_fingerprint")

    def _save_state(self, state: Dict[str, Any]) -> None:
        write_json(self.state_file, state)

    def _write_runner_error(self, exc: Exception) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            self.state_file,
            {
                "last_status": "runner_error",
                "error": repr(exc),
                "updated_at": now_iso(),
            },
        )


def normalize_command(request: Dict[str, Any]) -> Tuple[Any, bool]:
    command = request.get("command")
    if not command:
        raise ValueError("next_run.json must contain a command")
    shell = bool(request.get("shell", False))
    if isinstance(command, list):
        return [str(part) for part in command], shell
    if isinstance(command, str):
        if shell:
            return command, True
        return shlex.split(command, posix=(os.name != "nt")), False
    raise TypeError("command must be a string or a list")


def run_subprocess(command: Any, cwd: Path, stdout_path: Path, stderr_path: Path, timeout_seconds: Optional[int], shell: bool) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file:
        with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            stdout_thread = threading.Thread(target=copy_stream, args=(process.stdout, stdout_file, sys.stdout), daemon=True)
            stderr_thread = threading.Thread(target=copy_stream, args=(process.stderr, stderr_file, sys.stderr), daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
                raise
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            return returncode


def copy_stream(stream, file_obj, console_obj) -> None:
    if stream is None:
        return
    for line in iter(stream.readline, ""):
        file_obj.write(line)
        file_obj.flush()
        console_obj.write(line)
        console_obj.flush()
    stream.close()


def collect_metrics(request: Dict[str, Any], status: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    requested_paths = [Path(path) for path in request.get("metrics_paths") or []]
    inferred_paths = infer_metrics_paths(request)
    summaries = []
    seen = set()
    for path in requested_paths + inferred_paths:
        resolved = resolve_under_project(path)
        if str(resolved) in seen or not resolved.exists():
            continue
        seen.add(str(resolved))
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summaries.append({"path": str(resolved), "metrics": compact_metrics(data)})

    latest_context_path = PROJECT_ROOT / "experiment_history" / str(request.get("dataset") or parse_arg(request, "--dataset") or "DROP") / "latest_context.md"
    latest_context = latest_context_path.read_text(encoding="utf-8") if latest_context_path.exists() else ""
    return {
        "run_id": status.get("run_id"),
        "status": status.get("status"),
        "returncode": status.get("returncode"),
        "duration_seconds": status.get("duration_seconds"),
        "finished_at": status.get("finished_at"),
        "summaries": summaries,
        "latest_context_path": str(latest_context_path) if latest_context_path.exists() else None,
        "latest_context_preview": latest_context[:4000],
        "run_dir": str(run_dir),
    }


def infer_metrics_paths(request: Dict[str, Any]) -> List[Path]:
    dataset = request.get("dataset") or parse_arg(request, "--dataset")
    run_id = request.get("run_id") or parse_arg(request, "--run_id")
    output_dir = request.get("output_dir") or parse_arg(request, "--output_dir")
    history_dir = request.get("history_dir") or parse_arg(request, "--history_dir") or "experiment_history"
    paths = []
    if output_dir and dataset and run_id:
        paths.append(Path(output_dir) / str(dataset) / str(run_id) / "summary.json")
        paths.append(Path(output_dir) / str(dataset) / str(run_id) / "compiled_workflow_eval_summary.json")
    if dataset:
        paths.append(Path(history_dir) / str(dataset) / "latest.json")
    return paths


def compact_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    fields = [
        "status",
        "stop_reason",
        "dataset",
        "run_id",
        "phase",
        "sample_count",
        "baseline_average_score",
        "average_score",
        "success_threshold",
        "success_count",
        "failure_count",
        "exact_count",
        "partial_count",
        "zero_count",
        "accepted_patch_count",
        "rejected_patch_count",
        "skipped_patch_count",
        "compiled_score",
        "latest_phase",
        "iterations_completed",
        "validation_iterations_requested",
        "best_score",
        "best_failure_count",
        "initial_workflow_dir",
        "final_workflow_dir",
        "best_workflow_dir",
    ]
    compact = {field: data.get(field) for field in fields if field in data}
    if "latest_metrics" in data:
        compact["latest_metrics"] = data.get("latest_metrics")
    if "online_repair_metrics" in data:
        compact["online_repair_metrics"] = data.get("online_repair_metrics")
    if "compiled_eval_metrics" in data:
        compact["compiled_eval_metrics"] = data.get("compiled_eval_metrics")
    if "compiled_workflow_evaluation" in data:
        compact["compiled_workflow_evaluation"] = compact_metrics(data["compiled_workflow_evaluation"])
    if "workflow_compilation" in data:
        compilation = data.get("workflow_compilation") or {}
        compact["workflow_compilation"] = {
            "compiled_workflow_dir": compilation.get("compiled_workflow_dir"),
            "compiled_patch_count": compilation.get("compiled_patch_count"),
            "content_repair_batches": compilation.get("content_repair_batches"),
            "inserted_nodes": [node.get("node_name") for node in compilation.get("inserted_nodes") or []],
        }
    if "iterations" in data:
        compact["iterations"] = [
            {
                "iteration": item.get("iteration"),
                "run_id": item.get("run_id"),
                "baseline_score": item.get("baseline_score"),
                "online_score": item.get("online_score"),
                "compiled_score": item.get("compiled_score"),
                "candidate_score": item.get("candidate_score"),
                "candidate_failure_count": item.get("candidate_failure_count"),
                "adopted": item.get("adopted"),
                "adoption_reason": item.get("adoption_reason"),
                "improved_best": item.get("improved_best"),
                "candidate_workflow_dir": item.get("candidate_workflow_dir"),
            }
            for item in (data.get("iterations") or [])[-10:]
        ]
    failures = data.get("failure_samples") or []
    compact["failure_samples"] = failures[:30]
    return compact


def build_codex_prompt(
    codex_cfg: Dict[str, Any],
    request: Dict[str, Any],
    status_path: Path,
    metrics_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> str:
    custom = str(codex_cfg.get("prompt") or "").strip()
    stdout_tail = tail_text(stdout_path, int(codex_cfg.get("stdout_tail_chars") or 6000))
    stderr_tail = tail_text(stderr_path, int(codex_cfg.get("stderr_tail_chars") or 6000))
    dataset = request.get("dataset") or parse_arg(request, "--dataset") or "DROP"
    latest_context_path = PROJECT_ROOT / "experiment_history" / str(dataset) / "latest_context.md"
    context_hint = latest_context_path.read_text(encoding="utf-8") if latest_context_path.exists() else ""
    parts = [
        custom
        or (
            "Read the auto-runner outputs, analyze the validation result, then either "
            "write the next commands/next_run.json for another validation iteration or "
            "write commands/loop_done.json with the stop reason."
        ),
        "",
        "Runner output files:",
        f"- status: {status_path}",
        f"- metrics: {metrics_path}",
        f"- stdout: {stdout_path}",
        f"- stderr: {stderr_path}",
        "",
        "Latest experiment context:",
        context_hint,
        "",
        "stdout tail:",
        stdout_tail,
        "",
        "stderr tail:",
        stderr_tail,
    ]
    return "\n".join(parts)


def default_codex_command() -> List[str]:
    return [find_codex_executable(), "exec", "--cd", str(PROJECT_ROOT), "{prompt}"]


def find_codex_executable() -> str:
    for name in ("codex.exe", "codex"):
        found = shutil.which(name)
        if found and executable_works(found):
            return found
    if os.name == "nt":
        windows_apps = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "WindowsApps"
        try:
            for candidate in sorted(windows_apps.glob("OpenAI.Codex_*/app/resources/codex.exe"), reverse=True):
                if candidate.exists() and executable_works(str(candidate)):
                    return str(candidate)
        except OSError:
            pass
    return "codex"


def executable_works(path: str) -> bool:
    try:
        subprocess.run(
            [path, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def render_command_template(command_template: List[str], prompt: str, prompt_path: Path) -> List[str]:
    replacements = {
        "{repo_root}": str(PROJECT_ROOT),
        "{prompt}": prompt,
        "{prompt_path}": str(prompt_path),
    }
    rendered = []
    for part in command_template:
        text = str(part)
        for key, value in replacements.items():
            text = text.replace(key, value)
        rendered.append(text)
    return rendered


def parse_arg(request: Dict[str, Any], name: str) -> Optional[str]:
    command = request.get("command")
    parts = command if isinstance(command, list) else shlex.split(str(command or ""), posix=(os.name != "nt"))
    for index, part in enumerate(parts):
        if part == name and index + 1 < len(parts):
            return str(parts[index + 1])
        prefix = f"{name}="
        if str(part).startswith(prefix):
            return str(part)[len(prefix) :]
    return None


def parse_optional_json_list(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("--codex_command must be a JSON list")
    return [str(item) for item in parsed]


def tail_text(path: Path, chars: int) -> str:
    if not path.exists() or chars <= 0:
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-chars:]


def file_sha256(path: Path) -> Optional[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def resolve_under_project(path: Path) -> Path:
    resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must stay under project root: {resolved}") from exc
    return resolved


def safe_run_id(raw: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in raw.strip())
    return cleaned or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_args():
    parser = argparse.ArgumentParser(description="Watch commands/next_run.json and execute queued experiments.")
    parser.add_argument("--command_file", default=str(DEFAULT_COMMAND_FILE))
    parser.add_argument("--state_file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--runs_dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--poll_seconds", type=float, default=10.0)
    parser.add_argument("--once", action="store_true", help="Process one changed command file and exit")
    parser.add_argument("--rerun_current", action="store_true", help="Ignore runner_state.json once and process the current command file")
    parser.add_argument("--enable_codex", action="store_true", help="Allow codex_after_run.enabled to invoke codex exec")
    parser.add_argument(
        "--codex_command",
        default=None,
        help='Optional JSON list command template, e.g. ["codex","exec","--cd","{repo_root}","{prompt}"]',
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
