# Latest Workflow Repair Run

Use this file as compact context when opening a new Codex conversation.

- dataset: DROP
- run_id: drop_test_round3_baseline_qwen3_fp8
- workflow_id: round_3
- model: qwen3-32b-fp8
- latest_phase: online_repair
- latest_score: 0.827492149273012
- latest_success/failure: 681 / 119
- online_score: 0.827492149273012
- compiled_score: None
- repair_rounds_completed: 1
- accepted_patch_count: 0
- inserted_nodes: 

## Key Paths
- summary: repair_runs_test_round3_baseline_qwen3_fp8\DROP\drop_test_round3_baseline_qwen3_fp8\summary.json
- repair_events: repair_runs_test_round3_baseline_qwen3_fp8\DROP\drop_test_round3_baseline_qwen3_fp8\repair_events.jsonl
- trace_root: traces_test_round3_baseline_qwen3_fp8\DROP\drop_test_round3_baseline_qwen3_fp8
- patch_registry: results_test_round3_baseline_qwen3_fp8
- compiled_workflow: None
- compiled_eval_summary: None

## Remaining Failures
- sample 6297: pred=The passage does not provide information on how many yards Simms ran. expected=60 score=0.0
- sample 7936: pred=12 expected=7|14 score=0.0
- sample 8705: pred='answer' expected=22-yard and 36-yard and 35-yard|a 22-yard field goal and a 36-yard field goal and a 35-yard field goal score=0.0
- sample 1084: pred=Vietnamese expected=French score=0.0
- sample 1078: pred=30 expected=6|34 score=0.0
- sample 2602: pred=Not specified in the passage expected=1 score=0.0
- sample 8304: pred=75.40% expected=75.4 score=0.0
- sample 9378: pred='answer' expected=Norwegian and Polish|Norwegian people and Polish people score=0.0
- sample 7502: pred=300,000 expected=910000 score=0.0
- sample 2840: pred=The KDP had more supporters expelled from the other's region. expected=The KDP and estimated that 58,000|KDP|The KDP score=0.2222222222222222
- sample 265: pred=56.0% expected=96|56 score=0.0
- sample 9271: pred=1 expected=2|0 score=0.0
- sample 169: pred=Imports expected=import score=0.0
- sample 8825: pred=54.60% expected=54.6 score=0.0
- sample 50: pred=14 expected=13 score=0.0
- sample 6676: pred=Edward I's father and son died. expected=Prince Edward score=0.25
- sample 4438: pred=5 expected=4 score=0.0
- sample 2018: pred=Josh Lambo expected=Jaguars score=0.0
- sample 7129: pred=128 expected=86 score=0.0
- sample 7091: pred='answer' expected=Vietnamese and Chinese score=0.0
- sample 683: pred=19 expected=3 score=0.0
- sample 7478: pred=6 expected=5|4 score=0.0
- sample 586: pred=Yes expected=Lockette had a broken neck.|Lockette had a broken neck score=0.0
- sample 119: pred=37.5% expected=15.5 score=0.0
- sample 4085: pred=Fianna Fáil and Fine Gael are the two largest political parties in the Republic of Ireland, descended from the anti-treaty and pro-treaty forces of 1922, respectively. expected=political parties|two largest political parties score=0.29629629629629634
- sample 6778: pred=2 expected=3 score=0.0
- sample 567: pred=0.70% expected=0.7|.7 score=0.0
- sample 7663: pred='answer' expected=African American score=0.0
- sample 2471: pred=6 expected=5 score=0.0
- sample 8857: pred=Approximately 1,719,075 more people live in Bangkok in 2018 compared to 2010. expected=1719075|1700000 score=0.15384615384615385

## Next Useful Action
- If compiled_eval_summary is empty but compiled_workflow exists, rerun with --eval_compiled_workflow.
