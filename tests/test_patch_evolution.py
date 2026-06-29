import asyncio
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.async_llm import LLMsConfig
from scripts.auto_runner import compact_metrics, normalize_command, render_command_template, safe_run_id
from patch_evolution.consolidator import OfflineConsolidator
from patch_evolution.contracts import ContractChecker
from patch_evolution.fixer import FixerExecutor
from patch_evolution.guarded_runtime import GuardedRuntime
from patch_evolution.online_patch import (
    build_answer_style_patch,
    build_concise_answer_llm_patch,
    build_missing_answer_llm_patch,
    build_numeric_normalization_patch,
)
from patch_evolution.patch_registry import PatchRegistry
from patch_evolution.patch_spec import make_patch_spec, validate_patch_schema
from patch_evolution.patch_synthesizer import PatchSynthesizer
from patch_evolution.probe import ProbeBeforeEdit
from patch_evolution.repair_workflow import EvaluationOutcome, WorkflowRepairRunner, defined_prompt_symbols
from patch_evolution.run_history import build_history_record, write_run_history
from patch_evolution.shadow_validation import ShadowValidator
from patch_evolution.static_validator import StaticPatchValidator
from patch_evolution.trace_ir import NodeState, OperatorState, TraceIR, TraceMetadata
from patch_evolution.trigger_detector import TriggerDetector
from patch_evolution.workflow_compiler import WorkflowPatchCompiler
from patch_evolution.workflow_nodes import (
    AnswerCanonicalizerNode,
    CompiledPatchNode,
    ContentRepairNode,
    MissingAnswerRecoveryNode,
    MultiSpanExtractorNode,
    NumericVerifierNode,
)


def patch_spec(patch_id="patch_test", status="active_guarded"):
    return make_patch_spec(
        patch_id=patch_id,
        source_trace_id="trace_1",
        target_operator="Solver",
        diagnosed_bottleneck="output_schema_or_format",
        failure_symptom="empty output",
        trigger={
            "observable_signals": ["output_empty"],
            "decision_rule": "output_empty == true",
            "threshold": 0.5,
            "detector_type": "rule",
        },
        fixer={
            "type": "contract_repair",
            "name": "repair_empty",
            "definition": "fill empty outputs with fallback",
            "input_mapping": {},
            "output_mapping": {},
            "fallback_output": {"response": "fallback answer"},
        },
        scope={"task_type": "toy", "operator": "Solver", "conditions": ["empty output"]},
        contract={
            "input_requirements": [],
            "output_requirements": ["non_empty", "field:response"],
            "downstream_compatibility": ["dict with response field"],
        },
        status=status,
    )


class PatchEvolutionTests(unittest.TestCase):
    def test_patch_schema(self):
        patch = patch_spec()
        ok, errors = validate_patch_schema(patch)
        self.assertTrue(ok, errors)
        patch["trigger"]["observable_signals"] = ["gold_answer_correctness"]
        ok, errors = validate_patch_schema(patch)
        self.assertFalse(ok)

    def test_trigger_detector(self):
        detector = TriggerDetector()
        patch = patch_spec()
        should_trigger, risk, info = detector.should_trigger({"output_empty": True}, patch)
        self.assertTrue(should_trigger)
        self.assertEqual(risk, 1.0)
        should_trigger, risk, info = detector.should_trigger({"output_empty": False}, patch)
        self.assertFalse(should_trigger)
        should_trigger, _, _ = detector.should_trigger(
            {"output": {"answer": "The answer is Duke University."}},
            {
                **patch,
                "trigger": {
                    "observable_signals": ["output"],
                    "decision_rule": "output icontains answer is",
                    "threshold": 0.5,
                    "detector_type": "rule",
                },
            },
        )
        self.assertTrue(should_trigger)
        should_trigger, _, _ = detector.should_trigger(
            {"answer_word_count": 8},
            {
                **patch,
                "trigger": {
                    "detector": "rule",
                    "type": "int_gte",
                    "field": "answer_word_count",
                    "value": 6,
                },
            },
        )
        self.assertTrue(should_trigger)

    def test_contract_checker(self):
        checker = ContractChecker()
        contract = patch_spec()["contract"]
        self.assertTrue(checker.check({"response": "ok"}, contract).ok)
        self.assertFalse(checker.check({}, contract).ok)

    def test_registry_and_consolidator(self):
        root = Path("logs/patch_evolution_tests/registry")
        registry = PatchRegistry(str(root))
        patch = patch_spec("patch_registry")
        registry.add_patch(patch)
        registry.update_telemetry(
            "patch_registry",
            {"checked_count": 30, "activation_count": 25, "accepted_count": 24, "repair_success_count": 20},
        )
        loaded = registry.load_patch("patch_registry")
        loaded["telemetry"]["average_gain"] = 0.5
        registry.save_patch(loaded)
        active = registry.query_active_by_target("Solver")
        self.assertEqual(len(active), 1)
        decision = OfflineConsolidator().decide_patch(active[0])
        self.assertIn(decision["decision"], {"keep", "promote"})
        disabled = patch_spec("patch_disabled", status="disabled")
        disabled["telemetry"]["average_gain"] = 1.0
        disabled["telemetry"]["activation_count"] = 10
        disabled["telemetry"]["accepted_count"] = 10
        decision = OfflineConsolidator().decide_patch(disabled)
        self.assertEqual(decision["decision"], "prune")

    def test_run_history_writes_latest_context(self):
        summary = {
            "dataset": "DROP",
            "run_id": "history_run",
            "workflow_id": "round_3",
            "workflow_dir": "workspace/round_3",
            "repair_rounds_completed": 2,
            "sample_count": 2,
            "baseline_average_score": 0.5,
            "average_score": 0.75,
            "success_threshold": 0.3,
            "success_count": 1,
            "failure_count": 1,
            "accepted_patch_count": 1,
            "rejected_patch_count": 2,
            "skipped_patch_count": 0,
            "accepted_patch_ids": ["patch_a"],
            "trace_root": "traces/DROP/history_run",
            "patch_registry_dir": "results/history",
            "repair_events_path": "repair_runs/history/repair_events.jsonl",
            "failure_samples": [{"sample_id": "s1", "score": 0.0, "prediction": "8", "expected": "19"}],
            "workflow_compilation": {
                "compiled_workflow_dir": "compiled/DROP/history_run/round_3_compiled",
                "compiled_patch_count": 1,
                "inserted_nodes": [{"node_name": "numeric_verifier"}],
                "content_repair_batches": ["numeric"],
                "compiled_patch_ids": ["patch_a"],
            },
            "compiled_workflow_evaluation": {
                "sample_count": 2,
                "average_score": 1.0,
                "success_threshold": 0.3,
                "success_count": 2,
                "failure_count": 0,
                "failure_samples": [],
                "path": "repair_runs/history/compiled_workflow_eval_summary.json",
                "content_node_accept_counts": {"NumericVerifier": 1},
            },
        }
        record = build_history_record(summary, summary_path="repair_runs/history/summary.json", exec_model_name="qwen3")
        self.assertEqual(record["latest_phase"], "compiled_eval")
        self.assertEqual(record["latest_metrics"]["average_score"], 1.0)
        self.assertEqual(record["compiled_workflow"]["inserted_nodes"][0]["node_name"], "numeric_verifier")

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_run_history(
                summary,
                history_dir=str(Path(tmpdir) / "history"),
                summary_path="repair_runs/history/summary.json",
                exec_model_name="qwen3",
            )
            latest = json.loads(Path(paths["latest_json_path"]).read_text(encoding="utf-8"))
            context = Path(paths["latest_context_path"]).read_text(encoding="utf-8")
            self.assertEqual(latest["run_id"], "history_run")
            self.assertIn("latest_score: 1.0", context)
            self.assertTrue(Path(paths["runs_jsonl_path"]).exists())

    def test_auto_runner_helpers(self):
        command, shell = normalize_command({"command": ["conda", "run", "-n", "aflow", "python", "-V"]})
        self.assertFalse(shell)
        self.assertEqual(command[:4], ["conda", "run", "-n", "aflow"])

        command, shell = normalize_command({"command": "python -V"})
        self.assertFalse(shell)
        self.assertEqual(command[0], "python")
        self.assertEqual(safe_run_id("bad run/id"), "bad_run_id")

        metrics = compact_metrics(
            {
                "dataset": "DROP",
                "average_score": 0.9,
                "success_count": 9,
                "failure_count": 1,
                "workflow_compilation": {
                    "compiled_workflow_dir": "compiled",
                    "compiled_patch_count": 2,
                    "inserted_nodes": [{"node_name": "numeric_verifier"}],
                },
                "failure_samples": [{"sample_id": "s1"}],
            }
        )
        self.assertEqual(metrics["average_score"], 0.9)
        self.assertEqual(metrics["workflow_compilation"]["inserted_nodes"], ["numeric_verifier"])
        rendered = render_command_template(
            ["codex", "exec", "--cd", "{repo_root}", "{prompt_path}"],
            "analyze",
            Path("commands/runs/r1/codex_prompt.txt"),
        )
        self.assertIn("codex", rendered)
        self.assertTrue(rendered[-1].endswith("codex_prompt.txt"))

    def test_shadow_validator_and_probe_synthesizer(self):
        trace = TraceIR(
            metadata=TraceMetadata(
                trace_id="trace_empty",
                task_name="toy",
                sample_id="sample_1",
                workflow_id="workflow_1",
                success=False,
            ),
            input="question",
            final_output="",
            nodes=[
                NodeState(
                    node_id="Solver#0",
                    operator=OperatorState(
                        operator_name="Solver",
                        operator_input="question",
                        operator_output="",
                        metadata={"parse_status": "empty"},
                    ),
                )
            ],
        )
        diagnosis = ProbeBeforeEdit().diagnose(trace)
        self.assertEqual(diagnosis["target_operator"], "Solver")
        patch = asyncio.run(PatchSynthesizer().synthesize(trace, diagnosis))[0]
        ok, errors = validate_patch_schema(patch)
        self.assertTrue(ok, errors)
        result = ShadowValidator().validate(
            patch_spec(),
            failed_traces=[{"output_empty": True, "simulated_gain": 0.2}],
            success_traces=[{"output_empty": False, "simulated_regression": 0.0}],
        )
        self.assertTrue(result["passed"])

    def test_guarded_runtime_toy(self):
        root = Path("logs/patch_evolution_tests/runtime")
        registry = PatchRegistry(str(root))
        registry.add_patch(patch_spec("patch_runtime"))
        runtime = GuardedRuntime(registry)
        output = asyncio.run(runtime.apply("Solver", "question", {}, metadata={"parse_status": "empty"}))
        self.assertEqual(output["response"], "fallback answer")
        telemetry = registry.load_patch("patch_runtime")["telemetry"]
        self.assertEqual(telemetry["checked_count"], 1)
        self.assertEqual(telemetry["accepted_count"], 1)

    def test_answer_style_patch_and_fixer(self):
        record = {
            "sample_id": "s1",
            "dataset": "MATH",
            "prediction": "The answer is \\boxed{Two}.",
            "expected": "2",
            "operator_traces": [
                {
                    "operator_id": "ScEnsemble#0",
                    "operator_type": "ScEnsemble",
                    "output_preview": '{"solution_letter": "A"}',
                },
                {
                    "operator_id": "AnswerGenerate#1",
                    "operator_type": "AnswerGenerate",
                    "output_preview": '{"answer": "The answer is \\\\boxed{Two}.", "thought": "done"}',
                },
            ],
        }
        patch = build_answer_style_patch(record)
        self.assertIsNotNone(patch)
        self.assertEqual(patch["target_operator"], "AnswerGenerate")
        ok, errors = validate_patch_schema(patch)
        self.assertTrue(ok, errors)

        output = asyncio.run(
            FixerExecutor().execute(
                patch,
                operator_input="question",
                original_output={"answer": "The answer is \\boxed{Two}.", "thought": "done"},
                trace_state={},
            )
        )
        self.assertEqual(output["answer"], "2")
        self.assertEqual(output["thought"], "done")

    def test_numeric_and_llm_repair_patches(self):
        numeric_record = {
            "sample_id": "n1",
            "dataset": "DROP",
            "prediction": "1.00",
            "expected": "1",
            "operator_traces": [
                {
                    "operator_id": "AnswerGenerate#0",
                    "operator_type": "AnswerGenerate",
                    "output_preview": '{"answer": "1.00", "thought": "done"}',
                }
            ],
        }
        numeric_patch = build_numeric_normalization_patch(numeric_record)
        self.assertIsNotNone(numeric_patch)
        output = asyncio.run(
            FixerExecutor().execute(
                numeric_patch,
                operator_input="question",
                original_output={"answer": "1.00", "thought": "done"},
                trace_state={},
            )
        )
        self.assertEqual(output["answer"], "1")

        missing_record = {
            "sample_id": "m1",
            "dataset": "DROP",
            "prediction": "'answer'",
            "expected": "Albanians",
            "operator_traces": [
                {
                    "operator_id": "AnswerGenerate#0",
                    "operator_type": "AnswerGenerate",
                    "output_preview": '{"thought": "Therefore, the Albanian group is smaller."}',
                }
            ],
        }
        llm_patch = build_missing_answer_llm_patch(missing_record)
        self.assertIsNotNone(llm_patch)
        self.assertIsNone(build_concise_answer_llm_patch(missing_record))
        ok, errors = validate_patch_schema(llm_patch)
        self.assertTrue(ok, errors)
        should_trigger, _, _ = TriggerDetector().should_trigger(
            {"output_fields": ["thought"]},
            llm_patch,
        )
        self.assertTrue(should_trigger)

        repair_prompts = []

        async def fake_llm(prompt):
            repair_prompts.append(prompt)
            return '{"answer": "Albanians"}'

        output = asyncio.run(
            FixerExecutor(llm=fake_llm).execute(
                llm_patch,
                operator_input="question",
                original_output={"thought": "Therefore, the Albanian group is smaller."},
                trace_state={},
            )
        )
        self.assertEqual(output["answer"], "Albanians")
        self.assertIn("existing_operator_output", repair_prompts[0])
        self.assertIn("Albanian group is smaller", repair_prompts[0])

    def test_answer_normalize_fixer(self):
        patch = patch_spec("normalize_patch")
        patch["fixer"] = {
            "type": "answer_normalize",
            "name": "normalize_answer",
            "writes": ["answer"],
            "fields": ["answer"],
            "repair_goal": "Normalize existing answer only.",
        }
        output = asyncio.run(
            FixerExecutor().execute(
                patch,
                operator_input="question",
                original_output={"answer": "The answer is Duke University.", "thought": "done"},
                trace_state={},
            )
        )
        self.assertEqual(output["answer"], "Duke University")

    def test_static_patch_validator_accepts_and_rejects_runtime_grounded_specs(self):
        validator = StaticPatchValidator("DROP")
        valid_patch = patch_spec("runtime_valid")
        valid_patch["target_operator"] = "AnswerGenerate"
        valid_patch["trigger"] = {"detector": "rule", "type": "field_missing", "field": "answer"}
        valid_patch["fixer"] = {
            "type": "contract_repair",
            "writes": ["answer"],
            "repair_goal": "Recover answer field from existing output.",
        }
        result = validator.validate(valid_patch)
        self.assertTrue(result.ok, result.errors)
        self.assertIn(
            "DROP::AnswerGenerate::field_missing:answer::contract_repair:answer:",
            result.normalized_patch["patch_key"],
        )

        number_word_patch = deepcopy(valid_patch)
        number_word_patch["trigger"] = {"detector": "rule", "type": "icontains", "field": "answer_text", "value": "three"}
        number_word_patch["fixer"] = {
            "type": "contract_repair",
            "writes": ["answer"],
            "transforms": ["number_words_to_digits"],
            "repair_goal": "Convert answer number words to digits.",
        }
        million_patch = deepcopy(number_word_patch)
        million_patch["trigger"] = {"detector": "rule", "type": "icontains", "field": "answer_text", "value": "million"}
        million_patch["fixer"]["transforms"] = ["million_to_number"]
        number_word_result = validator.validate(number_word_patch)
        million_result = validator.validate(million_patch)
        self.assertTrue(number_word_result.ok, number_word_result.errors)
        self.assertTrue(million_result.ok, million_result.errors)
        self.assertNotEqual(
            number_word_result.normalized_patch["patch_key"],
            million_result.normalized_patch["patch_key"],
        )

        forbidden = dict(valid_patch)
        forbidden["trigger"] = {"detector": "rule", "type": "unsupported_raw", "field": "verifier_score", "decision_rule": "verifier_score < 1"}
        result = validator.validate(forbidden)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "rejected_invalid_trigger_signal")

        bad_scensemble = dict(valid_patch)
        bad_scensemble["target_operator"] = "ScEnsemble"
        bad_scensemble["trigger"] = {"detector": "rule", "type": "field_missing", "field": "solution_letter"}
        bad_scensemble["fixer"] = {"type": "llm_repair", "writes": ["answer"], "repair_goal": "Repair answer."}
        result = validator.validate(bad_scensemble)
        self.assertFalse(result.ok)
        self.assertIn(result.status, {"rejected_fixer_not_allowed", "rejected_target_field_mismatch"})

    def test_defined_prompt_symbols(self):
        symbols = defined_prompt_symbols('SOLVE_PROMPT = "x"\nOTHER: str = "y"\nclass Helper: pass\n')
        self.assertIn("SOLVE_PROMPT", symbols)
        self.assertIn("OTHER", symbols)
        self.assertIn("Helper", symbols)

    def test_llm_config_loads_when_cwd_is_not_project_root(self):
        original_cwd = os.getcwd()
        original_default = LLMsConfig._default_config
        original_env = os.environ.get("AFLOW_CONFIG_PATH")
        try:
            LLMsConfig._default_config = None
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    config_path = Path(tmpdir) / "config2.yaml"
                    config_path.write_text(
                        "models:\n"
                        "  toy-model:\n"
                        "    api_key: test-key\n"
                        "    base_url: http://localhost/v1\n",
                        encoding="utf-8",
                    )
                    os.environ["AFLOW_CONFIG_PATH"] = str(config_path)
                    os.chdir(tmpdir)
                    config = LLMsConfig.default().get("toy-model")
                    self.assertEqual(config.model, "toy-model")
                    self.assertEqual(config.key, "test-key")
                finally:
                    os.chdir(original_cwd)
        finally:
            os.chdir(original_cwd)
            LLMsConfig._default_config = original_default
            if original_env is None:
                os.environ.pop("AFLOW_CONFIG_PATH", None)
            else:
                os.environ["AFLOW_CONFIG_PATH"] = original_env

    def test_failure_family_patch_synthesis_omits_gold_and_normalizes(self):
        prompts = []

        async def fake_llm(prompt):
            prompts.append(prompt)
            return """
            {
          "patches": [
                {
                  "target_operator": "AnswerGenerate",
                  "diagnosed_bottleneck": "answer_extraction_family",
                  "failure_symptom": "verbose answer contains explanation instead of a short final answer",
                  "trigger": {
                    "detector": "rule",
                    "type": "int_gte",
                    "field": "answer_word_count",
                    "value": 6
                  },
                  "fixer": {
                    "type": "answer_normalize",
                    "name": "concise_answer_family",
                    "repair_goal": "For this failure family, keep only the final entity or number and remove explanatory prose.",
                    "writes": ["answer"]
                  },
                  "scope": {
                    "conditions": ["verbose final answers on extraction-style QA"]
                  },
                  "contract": {
                    "output_requirements": ["non_empty", "field:answer"],
                    "downstream_compatibility": ["preserve dict output"]
                  },
                  "generalization_note": "This patch targets answer-style failures rather than one specific sample."
                }
              ]
            }
            """

        record = {
            "sample_id": "ff1",
            "dataset": "DROP",
            "workflow_id": "round_3",
            "sample": {"question": "Which university did she attend?"},
            "prediction": "The answer is Duke University because she graduated from there.",
            "expected": "GOLD_ONLY_EXPECTED_VALUE",
            "operator_traces": [
                {
                    "operator_id": "AnswerGenerate#0",
                    "operator_type": "AnswerGenerate",
                    "input_preview": '{"question": "Which university did she attend?"}',
                    "output_preview": '{"answer": "The answer is Duke University because she graduated from there."}',
                    "parse_status": "parsed",
                }
            ],
        }
        diagnosis = {
            "target_operator": "AnswerGenerate",
            "suspected_bottleneck": "answer_style_or_extraction",
            "failure_symptom": "verbose answer contains explanation instead of a short final answer",
            "suggested_observable_signals": ["answer_word_count"],
        }

        patches = asyncio.run(PatchSynthesizer(llm=fake_llm).synthesize_failure_family(record, diagnosis))
        self.assertEqual(len(patches), 1)
        patch = patches[0]
        ok, errors = validate_patch_schema(patch)
        self.assertTrue(ok, errors)
        self.assertEqual(patch["source_trace_id"], "ff1")
        self.assertEqual(patch["status"], "candidate")
        self.assertEqual(patch["fixer"]["type"], "answer_normalize")
        self.assertEqual(patch["trigger"]["type"], "int_gte")
        self.assertEqual(patch["generation"]["mode"], "failure_family_llm")
        self.assertIn("Do not solve the current sample.", prompts[0])
        self.assertIn("Do not target ScEnsemble for final answer repair.", prompts[0])
        self.assertNotIn("verifier_score < 1", prompts[0])
        self.assertNotIn("GOLD_ONLY_EXPECTED_VALUE", prompts[0])

    def test_failure_family_coerces_weak_contract_repair_to_llm_repair(self):
        prompts = []

        async def fake_llm(prompt):
            prompts.append(prompt)
            return """
            {
              "patches": [
                {
                  "target_operator": "AnswerGenerate",
                  "diagnosed_bottleneck": "missing_answer_field",
                  "failure_symptom": "answer field missing although thought contains the answer",
                  "trigger": {
                    "detector": "rule",
                    "type": "enum_equals",
                    "field": "parse_status",
                    "value": "failed"
                  },
                  "fixer": {
                    "type": "contract_repair",
                    "name": "family_contract_repair",
                    "repair_goal": "Ensure answer is present.",
                    "writes": ["answer"]
                  },
                  "scope": {
                    "conditions": ["missing answer from generated thought"]
                  },
                  "contract": {
                    "output_requirements": ["non_empty", "field:answer"],
                    "downstream_compatibility": ["preserve dict output"]
                  }
                }
              ]
            }
            """

        record = {
            "sample_id": "missing1",
            "dataset": "DROP",
            "workflow_id": "round_3",
            "prediction": "'answer'",
            "operator_traces": [
                {
                    "operator_id": "AnswerGenerate#0",
                    "operator_type": "AnswerGenerate",
                    "output_preview": '{"thought": "Albanians are fewer than Macedonians, so the answer is Albanians."}',
                    "parse_status": "ok",
                }
            ],
        }
        diagnosis = {
            "target_operator": "AnswerGenerate",
            "suspected_bottleneck": "output_schema_or_format",
            "failure_symptom": "answer field missing",
            "suggested_observable_signals": ["schema_valid", "parse_status"],
        }

        patches = asyncio.run(PatchSynthesizer(llm=fake_llm).synthesize_failure_family(record, diagnosis))
        self.assertEqual(len(patches), 1)
        patch = patches[0]
        self.assertEqual(patch["trigger"], {"detector": "rule", "type": "field_missing", "field": "answer"})
        self.assertEqual(patch["fixer"]["type"], "llm_repair")
        self.assertEqual(patch["generation"]["coerced_fixer_from"], "contract_repair")
        self.assertIn("prefer fixer.type=llm_repair", prompts[0])

    def test_shadow_rejected_llm_patch_can_retry_on_new_sample(self):
        runner = object.__new__(WorkflowRepairRunner)
        failed = EvaluationOutcome(
            index=1,
            sample_id="new_sample",
            problem={},
            result=None,
            record={},
            baseline_score=0.0,
            score=0.0,
            cost=0.0,
        )
        patch = {
            "status": "shadow_rejected",
            "fixer": {"type": "llm_repair"},
            "telemetry": {"repair_failure_count": 1, "repair_success_count": 0},
            "shadow_failed_sample_ids": ["old_sample"],
        }
        self.assertTrue(runner._should_retry_shadow_rejected(patch, failed))

        patch["shadow_failed_sample_ids"].append("new_sample")
        self.assertFalse(runner._should_retry_shadow_rejected(patch, failed))

        exhausted = {
            "status": "shadow_rejected",
            "fixer": {"type": "llm_repair"},
            "telemetry": {"repair_failure_count": 3, "repair_success_count": 0},
            "shadow_failed_sample_ids": [],
        }
        self.assertFalse(runner._should_retry_shadow_rejected(exhausted, failed))

    def test_compiled_patch_node_executes_as_workflow_node(self):
        patch = make_patch_spec(
            patch_id="compiled_two_to_digit",
            source_trace_id="trace_two",
            target_operator="AnswerGenerate",
            diagnosed_bottleneck="answer_style",
            failure_symptom="number word answer",
            trigger={"detector": "rule", "type": "icontains", "field": "answer_text", "value": "two"},
            fixer={
                "type": "contract_repair",
                "name": "number_word_to_digit",
                "definition": "Convert number words to digits.",
                "fields": ["answer"],
                "writes": ["answer"],
                "transforms": ["number_words_to_digits", "trim_terminal_punctuation"],
            },
            scope={"task_type": "DROP", "operator": "AnswerGenerate", "conditions": ["number word answer"]},
            contract={
                "input_requirements": [],
                "output_requirements": ["non_empty", "field:answer"],
                "downstream_compatibility": ["preserve dict output"],
            },
            status="compiled",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            patch_path = Path(tmpdir) / "compiled_patches.json"
            patch_path.write_text(json.dumps({"patches": [patch]}, ensure_ascii=False), encoding="utf-8")
            node = CompiledPatchNode(llm=None, patch_path=str(patch_path), target_operator="AnswerGenerate")
            output = asyncio.run(node({"answer": "Two"}, operator_input="question"))
            self.assertEqual(output["answer"], "2")

    def test_content_repair_node_can_fix_numeric_candidate(self):
        prompts = []

        async def fake_llm(prompt):
            prompts.append(prompt)
            return json.dumps(
                {
                    "should_repair": True,
                    "family": "numeric_content",
                    "answer": "9",
                    "confidence": 0.9,
                    "reason": "observable arithmetic",
                }
            )

        node = ContentRepairNode(llm=fake_llm, families=["numeric_content"])
        problem = {
            "question": "How many more field goals did the Tigers make than the Bears?",
            "context": "The Tigers made 31 field goals. The Bears made 22 field goals.",
        }
        output = asyncio.run(
            node(
                {"answer": "31", "thought": "The Tigers made 31 and the Bears made 22, so the difference is 9."},
                operator_input=problem,
            )
        )
        self.assertEqual(output["answer"], "9")
        self.assertIn("numeric_content", prompts[0])

    def test_missing_answer_recovery_node_recovers_from_thought(self):
        async def fake_llm(prompt):
            return json.dumps(
                {
                    "should_repair": True,
                    "answer": "Albanians",
                    "confidence": 0.91,
                    "reason": "thought states the Albanian group is smaller",
                }
            )

        node = MissingAnswerRecoveryNode(llm=fake_llm)
        output = asyncio.run(
            node(
                {"thought": "Since 103,891 is less than 338,358, the Albanian group is smaller than the Macedonian group."},
                operator_input={"question": "Which group was smaller?"},
            )
        )
        self.assertEqual(output["answer"], "Albanians")

    def test_answer_canonicalizer_node_handles_low_risk_formats(self):
        node = AnswerCanonicalizerNode()
        number_problem = "Passage: France and Austria signed the treaty.\nQuestion: How many countries signed it?\nAnswer:"
        output = asyncio.run(node({"answer": "Two"}, operator_input=number_problem))
        self.assertEqual(output["answer"], "2")

        demonym_problem = (
            "Passage: The French employed up to 160,000 troops. Spanish forces numbered about 90,000.\n"
            "Question: Which nation had more troops?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "France"}, operator_input=demonym_problem))
        self.assertEqual(output["answer"], "French")

    def test_numeric_verifier_generic_operations(self):
        node = NumericVerifierNode(llm=None)

        listed_sum_problem = (
            "Passage: Russian ground equipment losses are estimated to be three tanks, at least 20 armoured "
            "and 32 non-armoured vehicles lost in combat.\n"
            "Question: How many tanks, armoured and non-armoured loses were estimated to have been lost?\nAnswer:"
        )
        output = asyncio.run(
            node(
                {"answer": "Three tanks, at least 20 armoured, and 32 non-armoured vehicles."},
                operator_input=listed_sum_problem,
            )
        )
        self.assertEqual(output["answer"], "55")

        not_percent_problem = (
            "Passage: A census gave 79% Chinese, 14% Malay, and 6% Indian.\n"
            "Question: How many percent of people were not Malay in 1891?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "85%"}, operator_input=not_percent_problem))
        self.assertEqual(output["answer"], "86")

        sports_problem = (
            "Passage: Maurice Jones-Drew scored on a 6-yard touchdown run. Taylor would then score on a "
            "13-yard touchdown run. Three plays later, Jones-Drew scored on a 4-yard touchdown run.\n"
            "Question: How many yards longer was the longest touchdown run than the shortest?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "31"}, operator_input=sports_problem))
        self.assertEqual(output["answer"], "9")

        second_touchdown_run_problem = (
            "Passage: Byron Leftwich completed a short pass to Fred Taylor, who turned it into a 32-yard gain. "
            "Two plays later, Maurice Jones-Drew scored on a 6-yard touchdown run. Taylor would then score on a "
            "13-yard touchdown run. Three plays later, Jones-Drew scored on a 4-yard touchdown run. Leftwich flipped "
            "a 1-yard touchdown pass to Wrighster. Scobee added a 40-yard field goal.\n"
            "Question: How many yards was the second longest touchdown run?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "40"}, operator_input=second_touchdown_run_problem))
        self.assertEqual(output["answer"], "6")

        touchdown_pass_problem = (
            "Passage: Dave Rayner nailed a 23-yard field goal. Rayner got a 54-yarder and a 46-yarder to end "
            "the half. David Akers got a 40-yard field goal, while McNabb and WR Greg Lewis connected on two "
            "touchdown passes of 45 and 30 yards. McNabb got a 15-yard TD run.\n"
            "Question: How many yards was the longest touchdown pass?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "54"}, operator_input=touchdown_pass_problem))
        self.assertEqual(output["answer"], "45")

        touchdown_pass_team_problem = (
            "Passage: The Broncos opened with a 59-yard TD pass from Jay Cutler to Eddie Royal. "
            "Later, Jay Cutler threw a 36-yard touchdown pass to Brandon Stokley for Denver.\n"
            "Question: Which team scored the longest touchdown pass of the game?\nAnswer:"
        )
        output = asyncio.run(
            node(
                {"answer": "The Broncos scored the longest touchdown pass of the game."},
                operator_input=touchdown_pass_team_problem,
            )
        )
        self.assertEqual(output["answer"], "The Broncos scored the longest touchdown pass of the game.")
        self.assertNotIn("_repair_lock", output)

        field_goal_problem = (
            "Passage: John Kasay got a 44-yard field goal. Kasay nailed a 33-yard and a 30-yard field goal. "
            "Nate Kaeding got a 27-yard field goal. Kasay nailed a 49-yard field goal.\n"
            "Question: How many yards was the shortest field goal of the game?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "30"}, operator_input=field_goal_problem))
        self.assertEqual(output["answer"], "27")
        self.assertEqual(output["_repair_lock"]["detail"], "numeric_operation:min")

        field_goal_diff_problem = (
            "Passage: Rian Lindell got a 22-yard field goal. Later Lindell kicked a 28-yard field goal.\n"
            "Question: How many yards longer was Rian Lindell's longest field goal than his shortest?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "22"}, operator_input=field_goal_diff_problem))
        self.assertEqual(output["answer"], "6")
        self.assertEqual(output["_repair_lock"]["detail"], "numeric_operation:max_minus_min")

        field_goal_subject_problem = (
            "Passage: The Cardinals would respond with two field goals from Neil Rackers from 44 and 22 yards. "
            "In the fourth quarter, Josh Brown made a 51-yard field goal.\n"
            "Question: How many yards was Neil Rackers's longest field goal?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "51"}, operator_input=field_goal_subject_problem))
        self.assertEqual(output["answer"], "44")

        field_goal_comparison_problem = (
            "Passage: Rob Bironas managed to get a 37-yard field goal. Bironas kicked a 37-yard field goal. "
            "John Carney got a 36-yard field goal. Bironas nailed a 40-yard and a 25-yard field goal.\n"
            "Question: How many yards longer was Rob Bironas' longest field goal compared to John Carney's only field goal?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "40"}, operator_input=field_goal_comparison_problem))
        self.assertEqual(output["answer"], "4")

        complex_percent_problem = (
            "Passage: 7.9% were between 40 and 44 years old, and 6.3% were between 45 and 49 years old.\n"
            "Question: How many percent of people were not between 40 and 44 years?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "93.7"}, operator_input=complex_percent_problem))
        self.assertEqual(output["answer"], "92.1")

        race_percent_problem = (
            "Passage: 0.73% Native American, 0.9% Asian, and 3.72% from two or more races.\n"
            "Question: How many percent of people were not from 2 or more races?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "97.82"}, operator_input=race_percent_problem))
        self.assertEqual(output["answer"], "96.28")

        exact_malay_percent_problem = (
            "Passage: A census in 1891 gave 79% of whom were Chinese, 14% Malay, and 6% Indian.\n"
            "Question: How many percent of people were not Malay in 1891?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "85%"}, operator_input=exact_malay_percent_problem))
        self.assertEqual(output["answer"], "86")

        full_malay_percent_problem = (
            "Passage: A census in 1891 of uncertain accuracy gave 79% of whom were Chinese, 14% Malay, and 6% Indian. "
            "In 1980 the population had 52% Chinese, 33% Malay, and 15% Indian. By the 2010 census, the percentage "
            "of Malay had reached 44.7%.\n"
            "Question: How many percent of people were not Malay in 1891?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "85%"}, operator_input=full_malay_percent_problem))
        self.assertEqual(output["answer"], "86")

        did_not_vote_problem = (
            "Passage: In the federal election, the voter turnout was 55.4%.\n"
            "Question: How many percent of the population did not vote?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "44.6%"}, operator_input=did_not_vote_problem))
        self.assertEqual(output["answer"], "44.6%")

        ancestry_difference_problem = (
            "Passage: 33.7% were of germans, 13.9% swedish people, 10.1% irish people, "
            "8.8% united states, 7.0% english people and 5.4% Danish people ancestry according to Census 2000.\n"
            "Question: How many more people were German than from the US or Danish?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "1517"}, operator_input=ancestry_difference_problem))
        self.assertEqual(output["answer"], "19.5")

        ancestry_with_census_metadata_problem = (
            "Passage: The racial makeup was 97.63% Race (United States Census), 0.18% Race (United States Census), "
            "and 0.69% from two or more races. 33.7% were of germans, 8.8% united states, and 5.4% Danish people "
            "ancestry according to Census 2000.\n"
            "Question: How many more people were German than from the US or Danish?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "1517"}, operator_input=ancestry_with_census_metadata_problem))
        self.assertEqual(output["answer"], "19.5")

        month_span_problem = (
            "Passage: The committee ran its review between May 1981 and December 1982 before publishing the report.\n"
            "Question: What was the time span in months between the start and end of the review?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "8"}, operator_input=month_span_problem))
        self.assertEqual(output["answer"], "19")
        self.assertEqual(output["_repair_lock"]["detail"], "numeric_operation:month_span")

        age_at_birth_problem = (
            "Passage: On September 6, 2006, Barron William Trump, the son of Donald Trump, was born. "
            "Donald Trump was the only male child born in the family since 1965.\n"
            "Question: How many years old was Donald Trump when his son Barron was born?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "40"}, operator_input=age_at_birth_problem))
        self.assertEqual(output["answer"], "41")
        self.assertEqual(output["_repair_lock"]["detail"], "numeric_operation:age_at_birth_event")

        repeated_field_goal_problem = (
            "Passage: Mason Crosby made field goals from 29, 40, 35, 35, and 26 yards in the game.\n"
            "Question: From what distance did Mason Crosby make two field goals?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "35 yards"}, operator_input=repeated_field_goal_problem))
        self.assertEqual(output["answer"], "35-yard")
        self.assertEqual(output["_repair_lock"]["detail"], "numeric_operation:repeated_value")

        percent_less_problem = (
            "Passage: In 2000, 6.7% worked at home, while 7.9% were without a car. "
            "A later table listed 5.9% without a car in 2010.\n"
            "Question: How many percent fewer people worked at home than were without a car?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "8"}, operator_input=percent_less_problem))
        self.assertEqual(output["answer"], "1.2")
        self.assertEqual(output["_repair_lock"]["detail"], "numeric_operation:percent_less_difference")

        tied_time_problem = (
            "Passage: With 1:24 remaining, the game was tied after a late touchdown. "
            "The next drive ended after less than four minutes.\n"
            "Question: At what time was the game tied?\nAnswer:"
        )
        output = asyncio.run(node({"answer": "Less than four minutes"}, operator_input=tied_time_problem))
        self.assertEqual(output["answer"], "1:24")
        self.assertEqual(output["_repair_lock"]["detail"], "numeric_operation:tied_game_time_remaining")

    def test_repair_lock_prevents_downstream_content_overwrite(self):
        numeric = NumericVerifierNode(llm=None)
        problem = (
            "Passage: Russian ground equipment losses are estimated to be three tanks, at least 20 armoured "
            "and 32 non-armoured vehicles lost in combat.\n"
            "Question: How many tanks, armoured and non-armoured loses were estimated to have been lost?\nAnswer:"
        )
        repaired = asyncio.run(
            numeric(
                {"answer": "Three tanks, at least 20 armoured, and 32 non-armoured vehicles."},
                operator_input=problem,
            )
        )
        self.assertEqual(repaired["answer"], "55")

        async def fake_llm(prompt):
            return json.dumps(
                {
                    "should_repair": True,
                    "family": "multi_span",
                    "answer": "3 tanks, 20 armoured, 32 non-armoured",
                    "confidence": 0.99,
                    "reason": "would overwrite without lock",
                }
            )

        downstream = MultiSpanExtractorNode(llm=fake_llm)
        after_downstream = asyncio.run(downstream(repaired, operator_input=problem))
        self.assertEqual(after_downstream["answer"], "55")
        self.assertEqual(after_downstream["_repair_lock"]["detail"], "numeric_operation:sum_listed_quantities")

    def test_content_repair_batch_all_includes_canonicalizer(self):
        compiler = WorkflowPatchCompiler(content_repair_batches=["all"])
        self.assertEqual(
            compiler.content_repair_batches,
            ["schema", "canonical", "numeric", "multispan", "entity", "duration"],
        )

    def test_workflow_compiler_inserts_explicit_patch_node(self):
        patch = make_patch_spec(
            patch_id="compile_graph_patch",
            source_trace_id="trace_compile",
            target_operator="AnswerGenerate",
            diagnosed_bottleneck="answer_style",
            failure_symptom="number word answer",
            trigger={"detector": "rule", "type": "icontains", "field": "answer_text", "value": "two"},
            fixer={
                "type": "contract_repair",
                "name": "number_word_to_digit",
                "definition": "Convert number words to digits.",
                "fields": ["answer"],
                "writes": ["answer"],
                "transforms": ["number_words_to_digits", "trim_terminal_punctuation"],
            },
            scope={"task_type": "DROP", "operator": "AnswerGenerate", "conditions": ["number word answer"]},
            contract={
                "input_requirements": [],
                "output_requirements": ["non_empty", "field:answer"],
                "downstream_compatibility": ["preserve dict output"],
            },
            status="active_guarded",
        )
        patch["telemetry"]["checked_count"] = 10
        patch["telemetry"]["activation_count"] = 3
        patch["telemetry"]["accepted_count"] = 3
        patch["telemetry"]["average_gain"] = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow_dir = root / "round_3"
            workflow_dir.mkdir()
            (workflow_dir / "prompt.py").write_text("", encoding="utf-8")
            (workflow_dir / "graph.py").write_text(
                "import workspace.foo.workflows.template.operator as operator\n"
                "from scripts.async_llm import create_llm_instance\n\n"
                "class Workflow:\n"
                "    def __init__(self, name, llm_config, dataset):\n"
                "        self.llm = create_llm_instance(llm_config)\n"
                "        self.answer_gen = operator.AnswerGenerate(self.llm)\n"
                "        self.ensemble = operator.ScEnsemble(self.llm)\n\n"
                "    async def __call__(self, problem: str):\n"
                "        solutions = [\n"
                "            await self.answer_gen(input=problem),\n"
                "            await self.answer_gen(input=problem)\n"
                "        ]\n"
                "        solution_list = [sol['answer'] for sol in solutions]\n"
                "        final_answer = await self.ensemble(solutions=solution_list)\n"
                "        return final_answer['response'], self.llm.get_usage_summary()[\"total_cost\"]\n",
                encoding="utf-8",
            )

            result = WorkflowPatchCompiler().compile(
                workflow_dir=str(workflow_dir),
                patches=[patch],
                output_dir=str(root / "compiled"),
                dataset="DROP",
                run_id="test_run",
            )

            graph_text = Path(result.compiled_workflow_dir, "graph.py").read_text(encoding="utf-8")
            self.assertIn("CompiledPatchNode", graph_text)
            self.assertIn("self.answer_generate_patch", graph_text)
            self.assertIn("await self.answer_generate_patch(solution, operator_input=problem)", graph_text)
            compiled_patches = json.loads(Path(result.compiled_patches_path).read_text(encoding="utf-8"))
            self.assertEqual(compiled_patches["patches"][0]["status"], "compiled")
            self.assertEqual(result.compiled_patch_count, 1)

    def test_workflow_compiler_inserts_content_repair_node(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow_dir = root / "round_3"
            workflow_dir.mkdir()
            (workflow_dir / "prompt.py").write_text("", encoding="utf-8")
            (workflow_dir / "graph.py").write_text(
                "import workspace.foo.workflows.template.operator as operator\n"
                "from scripts.async_llm import create_llm_instance\n\n"
                "class Workflow:\n"
                "    def __init__(self, name, llm_config, dataset):\n"
                "        self.llm = create_llm_instance(llm_config)\n"
                "        self.answer_gen = operator.AnswerGenerate(self.llm)\n"
                "        self.ensemble = operator.ScEnsemble(self.llm)\n\n"
                "    async def __call__(self, problem: str):\n"
                "        solutions = [\n"
                "            await self.answer_gen(input=problem),\n"
                "            await self.answer_gen(input=problem)\n"
                "        ]\n"
                "        solution_list = [sol['answer'] for sol in solutions]\n"
                "        final_answer = await self.ensemble(solutions=solution_list)\n"
                "        return final_answer['response'], self.llm.get_usage_summary()[\"total_cost\"]\n",
                encoding="utf-8",
            )

            result = WorkflowPatchCompiler(
                compile_content_nodes=True,
                content_repair_families=["numeric_content", "multi_span"],
            ).compile(
                workflow_dir=str(workflow_dir),
                patches=[],
                output_dir=str(root / "compiled"),
                dataset="DROP",
                run_id="test_run",
            )

            graph_text = Path(result.compiled_workflow_dir, "graph.py").read_text(encoding="utf-8")
            self.assertIn("ContentRepairNode", graph_text)
            self.assertIn("self.content_repair", graph_text)
            self.assertIn("await self.content_repair(solution, operator_input=problem)", graph_text)
            self.assertEqual(result.content_repair_families, ["numeric_content", "multi_span"])
            self.assertEqual(result.inserted_nodes[0]["node_name"], "content_repair")

    def test_workflow_compiler_inserts_selectable_content_repair_batches(self):
        patch = make_patch_spec(
            patch_id="compile_order_patch",
            source_trace_id="trace_compile_order",
            target_operator="AnswerGenerate",
            diagnosed_bottleneck="answer_style",
            failure_symptom="number word answer",
            trigger={"detector": "rule", "type": "icontains", "field": "answer_text", "value": "two"},
            fixer={
                "type": "contract_repair",
                "name": "number_word_to_digit",
                "definition": "Convert number words to digits.",
                "fields": ["answer"],
                "writes": ["answer"],
                "transforms": ["number_words_to_digits"],
            },
            scope={"task_type": "DROP", "operator": "AnswerGenerate", "conditions": ["number word answer"]},
            contract={
                "input_requirements": [],
                "output_requirements": ["non_empty", "field:answer"],
                "downstream_compatibility": ["preserve dict output"],
            },
            status="active_guarded",
        )
        patch["telemetry"]["checked_count"] = 10
        patch["telemetry"]["activation_count"] = 3
        patch["telemetry"]["accepted_count"] = 3
        patch["telemetry"]["average_gain"] = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow_dir = root / "round_3"
            workflow_dir.mkdir()
            (workflow_dir / "prompt.py").write_text("", encoding="utf-8")
            (workflow_dir / "graph.py").write_text(
                "import workspace.foo.workflows.template.operator as operator\n"
                "from scripts.async_llm import create_llm_instance\n\n"
                "class Workflow:\n"
                "    def __init__(self, name, llm_config, dataset):\n"
                "        self.llm = create_llm_instance(llm_config)\n"
                "        self.answer_gen = operator.AnswerGenerate(self.llm)\n"
                "        self.ensemble = operator.ScEnsemble(self.llm)\n\n"
                "    async def __call__(self, problem: str):\n"
                "        solutions = [\n"
                "            await self.answer_gen(input=problem),\n"
                "            await self.answer_gen(input=problem)\n"
                "        ]\n"
                "        solution_list = [sol['answer'] for sol in solutions]\n"
                "        final_answer = await self.ensemble(solutions=solution_list)\n"
                "        return final_answer['response'], self.llm.get_usage_summary()[\"total_cost\"]\n",
                encoding="utf-8",
            )

            result = WorkflowPatchCompiler(
                content_repair_batches=["numeric", "multi_span", "entity", "duration"],
            ).compile(
                workflow_dir=str(workflow_dir),
                patches=[patch],
                output_dir=str(root / "compiled"),
                dataset="DROP",
                run_id="test_run",
            )

            graph_text = Path(result.compiled_workflow_dir, "graph.py").read_text(encoding="utf-8")
            self.assertIn("NumericVerifierNode", graph_text)
            self.assertIn("MultiSpanExtractorNode", graph_text)
            self.assertIn("AnswerTypeRefinerNode", graph_text)
            self.assertIn("DurationNormalizerNode", graph_text)
            self.assertIn("self.numeric_verifier", graph_text)
            self.assertIn("self.multi_span_extractor", graph_text)
            self.assertIn("self.answer_type_refiner", graph_text)
            self.assertIn("self.duration_normalizer", graph_text)
            self.assertIn("self.answer_generate_patch", graph_text)
            self.assertIn("await self.numeric_verifier(solution, operator_input=problem)", graph_text)
            self.assertIn("await self.multi_span_extractor(solution, operator_input=problem)", graph_text)
            self.assertIn("await self.answer_type_refiner(solution, operator_input=problem)", graph_text)
            self.assertIn("await self.duration_normalizer(solution, operator_input=problem)", graph_text)
            self.assertLess(
                graph_text.index("await self.duration_normalizer(solution, operator_input=problem)"),
                graph_text.index("await self.answer_generate_patch(solution, operator_input=problem)"),
            )
            self.assertEqual(result.content_repair_batches, ["numeric", "multispan", "entity", "duration"])
            self.assertEqual(
                [node["node_name"] for node in result.inserted_nodes],
                ["numeric_verifier", "multi_span_extractor", "answer_type_refiner", "duration_normalizer", "answer_generate_patch"],
            )


if __name__ == "__main__":
    unittest.main()
