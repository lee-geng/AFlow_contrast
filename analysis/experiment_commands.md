# Experiment Commands

This file records the main commands used for AFlow baseline and the proposed method.
The commands are written for PowerShell on this machine.

## Common Notes

- `--optimized_path workspace_smoke`: save all experiment artifacts under `workspace_smoke`.
- `--separate_model_artifacts`: isolate output by optimizer/executor model and `artifact_tag`.
- `--artifact_tag`: unique run name; change this to avoid overwriting old runs.
- `--sample 4`: sample 4 parent workflow candidates when selecting a parent.
- `--validation_rounds 3`: run validation 3 times for baseline-style repeated evaluation, unless the method aggregates full validation directly.
- `--eval_max_concurrent_tasks 5`: limit validation task concurrency to reduce server pressure.
- For unstable model service or 503 errors, lower `--eval_max_concurrent_tasks` to `2` or `3`.

## MBPP

### AFlow Baseline

This run is the MBPP AFlow baseline with Qwen3-32B-FP8 as both optimizer and executor.

```powershell
& "F:\ProgramData\anaconda3\envs\aflow\python.exe" run.py `
  --dataset MBPP `
  --sample 4 `
  --optimized_path workspace_smoke `
  --separate_model_artifacts `
  --artifact_tag baseline_qwen332bfp8_opt_exec_mbpp_run10 `
  --initial_round 1 `
  --max_rounds 30 `
  --check_convergence false `
  --validation_rounds 1 `
  --eval_max_concurrent_tasks 5 `
  --opt_model_name qwen3-32b-fp8 `
  --exec_model_name qwen3-32b-fp8
```

### Proposed Method

This run enables experience memory, failure cards, memory-guided prompt construction, full validation only, and disables external task embedding.

```powershell
& "F:\ProgramData\anaconda3\envs\aflow\python.exe" run.py `
  --dataset MBPP `
  --sample 4 `
  --optimized_path workspace_smoke `
  --separate_model_artifacts `
  --artifact_tag improved_qwen332bfp8_opt_exec_mbpp_run10_fullonly_codesafe `
  --initial_round 1 `
  --max_rounds 20 `
  --check_convergence false `
  --validation_rounds 1 `
  --eval_max_concurrent_tasks 5 `
  --opt_model_name qwen3-32b-fp8 `
  --exec_model_name qwen3-32b-fp8 `
  --workflow_guard_mode repair_then_continue `
  --experience_memory_enabled `
  --disable_staged_validation `
  --task_embedding_enabled false `
  --enable_prompt_budget_diagnostics `
  --use_failure_cards `
  --use_optimization_state_memory_query `
  --use_memory_guided_patch_scope `
  --use_memory_as_experience_replacement `
  --use_graph_prompt_summarization `
  --max_candidate_attempts_per_round 5
```

## HotpotQA

### AFlow Baseline

Baseline run for HotpotQA. It does not enable the proposed experience-store memory features.

```powershell
& "F:\ProgramData\anaconda3\envs\aflow\python.exe" run.py `
  --dataset HotpotQA `
  --sample 4 `
  --optimized_path workspace_smoke `
  --separate_model_artifacts `
  --artifact_tag qwen332bfp8_hotpotqa_aflow_baseline_sample4_run20_val3 `
  --initial_round 1 `
  --max_rounds 20 `
  --check_convergence true `
  --validation_rounds 3 `
  --eval_max_concurrent_tasks 5 `
  --opt_model_name qwen3-32b-fp8 `
  --exec_model_name qwen3-32b-fp8
```

### Proposed Method

HotpotQA run with experience memory enabled, full validation only, and external task embedding disabled.

```powershell
& "F:\ProgramData\anaconda3\envs\aflow\python.exe" run.py `
  --dataset HotpotQA `
  --sample 4 `
  --optimized_path workspace_smoke `
  --separate_model_artifacts `
  --artifact_tag qwen332bfp8_hotpotqa_sample4_run20_val3_exp_fullonly_noemb_conc5 `
  --initial_round 1 `
  --max_rounds 20 `
  --check_convergence true `
  --validation_rounds 3 `
  --eval_max_concurrent_tasks 5 `
  --opt_model_name qwen3-32b-fp8 `
  --exec_model_name qwen3-32b-fp8 `
  --workflow_guard_mode repair_then_continue `
  --experience_memory_enabled `
  --disable_staged_validation `
  --task_embedding_enabled false `
  --enable_prompt_budget_diagnostics `
  --use_failure_cards `
  --use_optimization_state_memory_query `
  --use_memory_guided_patch_scope `
  --use_memory_as_experience_replacement `
  --use_graph_prompt_summarization `
  --max_candidate_attempts_per_round 5
```

## DROP

### AFlow Baseline

Baseline run for DROP. The `workspace_template_ab/DROP` template must exist before running this.

```powershell
& "F:\ProgramData\anaconda3\envs\aflow\python.exe" run.py `
  --dataset DROP `
  --sample 4 `
  --optimized_path workspace_smoke `
  --separate_model_artifacts `
  --artifact_tag qwen332bfp8_drop_aflow_baseline_sample4_run20_val3 `
  --initial_round 1 `
  --max_rounds 20 `
  --check_convergence true `
  --validation_rounds 3 `
  --eval_max_concurrent_tasks 5 `
  --opt_model_name qwen3-32b-fp8 `
  --exec_model_name qwen3-32b-fp8
```

### Proposed Method

DROP run with the proposed experience-store memory path enabled.

```powershell
& "F:\ProgramData\anaconda3\envs\aflow\python.exe" run.py `
  --dataset DROP `
  --sample 4 `
  --optimized_path workspace_smoke `
  --separate_model_artifacts `
  --artifact_tag qwen332bfp8_drop_sample4_run20_val3_exp_fullonly_noemb_conc5 `
  --initial_round 1 `
  --max_rounds 20 `
  --check_convergence true `
  --validation_rounds 3 `
  --eval_max_concurrent_tasks 5 `
  --opt_model_name qwen3-32b-fp8 `
  --exec_model_name qwen3-32b-fp8 `
  --workflow_guard_mode repair_then_continue `
  --experience_memory_enabled `
  --disable_staged_validation `
  --task_embedding_enabled false `
  --enable_prompt_budget_diagnostics `
  --use_failure_cards `
  --use_optimization_state_memory_query `
  --use_memory_guided_patch_scope `
  --use_memory_as_experience_replacement `
  --use_graph_prompt_summarization `
  --max_candidate_attempts_per_round 5
```

## Resume A Run

Use this when a run stopped early and you want to continue from the last completed round.
Set `--initial_round` to the last completed round and adjust `--max_rounds` to the number of additional rounds you want.

```powershell
& "F:\ProgramData\anaconda3\envs\aflow\python.exe" run.py `
  --dataset DROP `
  --sample 4 `
  --optimized_path workspace_smoke `
  --separate_model_artifacts `
  --artifact_tag qwen332bfp8_drop_sample4_run20_val3_exp_fullonly_noemb_conc5 `
  --initial_round 13 `
  --max_rounds 7 `
  --check_convergence false `
  --validation_rounds 3 `
  --eval_max_concurrent_tasks 2 `
  --opt_model_name qwen3-32b-fp8 `
  --exec_model_name qwen3-32b-fp8 `
  --workflow_guard_mode repair_then_continue `
  --experience_memory_enabled `
  --disable_staged_validation `
  --task_embedding_enabled false `
  --enable_prompt_budget_diagnostics `
  --use_failure_cards `
  --use_optimization_state_memory_query `
  --use_memory_guided_patch_scope `
  --use_memory_as_experience_replacement `
  --use_graph_prompt_summarization `
  --max_candidate_attempts_per_round 5
```
