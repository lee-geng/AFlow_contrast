# Latest Workflow Repair Run

Use this file as compact context when opening a new Codex conversation.

- dataset: DROP
- run_id: drop_validation_loop_qwen3_fp8_v2_iter01
- workflow_id: round_3_compiled
- model: qwen3-32b-fp8
- latest_phase: compiled_eval
- latest_score: 0.9655827922077921
- latest_success/failure: 200 / 0
- online_score: 0.9655827922077921
- compiled_score: 0.9655827922077921
- repair_rounds_completed: 1
- accepted_patch_count: 0
- inserted_nodes: missing_answer_recovery, answer_canonicalizer, numeric_verifier, multi_span_extractor, answer_type_refiner, duration_normalizer, answer_generate_patch

## Key Paths
- summary: repair_runs_validation_loop_qwen3_fp8\DROP\drop_validation_loop_qwen3_fp8_v2_iter01\summary.json
- repair_events: repair_runs_validation_loop_qwen3_fp8\DROP\drop_validation_loop_qwen3_fp8_v2_iter01\repair_events.jsonl
- trace_root: traces_validation_loop_qwen3_fp8\DROP\drop_validation_loop_qwen3_fp8_v2_iter01
- patch_registry: results_validation_loop_qwen3_fp8
- compiled_workflow: F:\Code\Aflow_exp\AFlow_contrast\compiled_workflows_validation_loop_qwen3_fp8\DROP\drop_validation_loop_qwen3_fp8_v2_iter01\round_3_compiled_compiled
- compiled_eval_summary: repair_runs_validation_loop_qwen3_fp8\DROP\drop_validation_loop_qwen3_fp8_v2_iter01\compiled_workflow_eval_summary.json

## Remaining Failures
- none recorded

## Next Useful Action
- If compiled_eval_summary is empty but compiled_workflow exists, rerun with --eval_compiled_workflow.
