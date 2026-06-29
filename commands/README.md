# Auto Runner Commands

`scripts/auto_runner.py` watches `commands/next_run.json`. When that file changes, it executes the command and writes outputs under `commands/runs/<run_id>/`.

Typical outputs:

- `request.json`: snapshot of the submitted command request
- `run_status.json`: start/end status, return code, duration, output paths
- `stdout.txt`: live stdout capture
- `stderr.txt`: live stderr capture
- `metrics.json`: compact metrics collected from run summaries and `experiment_history`
- `codex_prompt.txt`, `codex_stdout.txt`, `codex_stderr.txt`: present only when `codex_after_run.enabled` is true and the runner was started with `--enable_codex`

The command file should be JSON. Use `next_run.example.jsonc` as the human-readable template, then save the actual request as `next_run.json`.

## Validation Workflow Loop

For iterative workflow optimization on the validation set, generate `next_run.json` with:

```powershell
& "F:\ProgramData\anaconda3\envs\aflow\python.exe" -B scripts\queue_validation_loop.py `
  --dataset DROP `
  --workflow_dir compiled_workflows_generalized_v2_qwen3_fp8\DROP\drop_round3_generalized_v2_qwen3_fp8\round_3_compiled `
  --model_name qwen3-32b-fp8 `
  --run_id drop_validation_loop_qwen3_fp8 `
  --max_samples 200 `
  --validation_iterations 3
```

Then start the watcher:

```powershell
& "F:\ProgramData\anaconda3\envs\aflow\python.exe" -B scripts\auto_runner.py --poll_seconds 15 --enable_codex --rerun_current
```

The loop command runs `patch_evolution.validation_loop`: repair, compile explicit workflow edits, evaluate the compiled workflow on validation, adopt only non-regressing compiled workflows, and repeat. If `codex_after_run.require_followup` is enabled, Codex must either update `commands/next_run.json` for the next validation iteration or write `commands/loop_done.json`.
