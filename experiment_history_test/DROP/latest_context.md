# Latest Workflow Repair Run

Use this file as compact context when opening a new Codex conversation.

- dataset: DROP
- run_id: drop_final_test_eval_qwen3_fp8
- workflow_id: round_3_compiled_compiled
- model: qwen3-32b-fp8
- latest_phase: online_repair
- latest_score: 0.8656827164064005
- latest_success/failure: 711 / 89
- online_score: 0.8656827164064005
- compiled_score: None
- repair_rounds_completed: 1
- accepted_patch_count: 0
- inserted_nodes: 

## Key Paths
- summary: repair_runs_final_test_qwen3_fp8\DROP\drop_final_test_eval_qwen3_fp8\summary.json
- repair_events: repair_runs_final_test_qwen3_fp8\DROP\drop_final_test_eval_qwen3_fp8\repair_events.jsonl
- trace_root: traces_final_test_qwen3_fp8\DROP\drop_final_test_eval_qwen3_fp8
- patch_registry: results_final_test_qwen3_fp8
- compiled_workflow: None
- compiled_eval_summary: None

## Remaining Failures
- sample 6389: pred=18 expected=27 score=0.0
- sample 6297: pred=The passage does not provide information on how many yards Simms ran expected=60 score=0.0
- sample 7936: pred=12 expected=7|14 score=0.0
- sample 1084: pred=Vietnamese expected=French score=0.0
- sample 1078: pred=30 expected=6|34 score=0.0
- sample 2602: pred=Not specified in the passage expected=1 score=0.0
- sample 7502: pred=300,000 expected=910000 score=0.0
- sample 9271: pred=1 expected=2|0 score=0.0
- sample 169: pred=Imports expected=import score=0.0
- sample 5244: pred=26 expected=26-yard and 33-yard|26-yard field goal and 33-yard field goal score=0.0
- sample 50: pred=14 expected=13 score=0.0
- sample 4438: pred=5 expected=4 score=0.0
- sample 2018: pred=Josh Lambo expected=Jaguars score=0.0
- sample 7129: pred=128 expected=86 score=0.0
- sample 683: pred=19 expected=3 score=0.0
- sample 7478: pred=6 expected=5|4 score=0.0
- sample 3689: pred=39 expected=9 score=0.0
- sample 586: pred=Yes expected=Lockette had a broken neck.|Lockette had a broken neck score=0.0
- sample 119: pred=37.5 expected=15.5 score=0.0
- sample 6778: pred=2 expected=3 score=0.0
- sample 2750: pred=17 expected=0.17|.3|10.6 score=0.0
- sample 7868: pred=53611 expected=51611|40,543 people and 11,068 families score=0.0
- sample 2471: pred=6 expected=5 score=0.0
- sample 8857: pred=1723103 expected=1719075|1700000 score=0.0
- sample 3184: pred=10 expected=2 score=0.0
- sample 668: pred=46 expected=41|7 score=0.0
- sample 8741: pred=30,626 expected=30826 score=0.0
- sample 3248: pred=39 expected=28|38|29 score=0.0
- sample 5950: pred=Panthers expected=Carolina score=0.0
- sample 9429: pred=12 expected=14 score=0.0

## Next Useful Action
- If compiled_eval_summary is empty but compiled_workflow exists, rerun with --eval_compiled_workflow.
