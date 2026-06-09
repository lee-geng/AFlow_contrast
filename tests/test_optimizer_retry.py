import unittest
import uuid
from pathlib import Path
import sys
import types

sys.modules.setdefault("aiofiles", types.ModuleType("aiofiles"))

from scripts.prompts.optimize_prompt import WORKFLOW_OPTIMIZE_PROMPT
from scripts.optimizer import Optimizer
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.experience_guided import ExperienceGuidedConfig, MemoryGuidance
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.optimizer_utils.retry_utils import is_retryable_runtime_error
from scripts.optimizer_utils.run_paths import (
    ensure_dataset_template_initialized,
    resolve_optimized_path,
    write_run_metadata,
)
from scripts.utils.common import write_json_file


class OptimizerRetryTests(unittest.TestCase):
    def test_optimize_prompt_is_python_first(self):
        self.assertIn("pure valid Python code for graph.py", WORKFLOW_OPTIMIZE_PROMPT)
        self.assertIn("Do not output `<Workflow>`, `<node>`, `interface`", WORKFLOW_OPTIMIZE_PROMPT)
        self.assertIn("Start with `class Workflow:`", WORKFLOW_OPTIMIZE_PROMPT)
        self.assertIn("Use only operator names that appear verbatim in the provided operator description", WORKFLOW_OPTIMIZE_PROMPT)
        self.assertIn("Do not create any new operator class names", WORKFLOW_OPTIMIZE_PROMPT)
        self.assertIn("Do not reference, instantiate, or call any attribute from `template/operator.py`", WORKFLOW_OPTIMIZE_PROMPT)

    def test_generation_failure_summary_extracts_root_causes(self):
        summary = Optimizer._summarize_recent_generation_failures(
            [
                "Attempt 1: invalid graph rejected by guard because missing_Workflow_class.",
                "Attempt 2: invalid graph rejected by guard because syntax_error_after_repair: invalid syntax (<unknown>, line 1).",
                "Attempt 3: modification rejected because it repeats known history: add a new review step",
            ]
        )
        self.assertIn("missing_Workflow_class", summary)
        self.assertIn("syntax_error_after_repair", summary)
        self.assertIn("repeated modification:", summary)

    def test_retryable_error_detection(self):
        self.assertTrue(is_retryable_runtime_error(RuntimeError("Connection error.")))
        self.assertTrue(is_retryable_runtime_error(RuntimeError("Request timed out")))
        self.assertTrue(is_retryable_runtime_error(RuntimeError("Rate limit exceeded")))

    def test_non_retryable_error_detection(self):
        self.assertFalse(
            is_retryable_runtime_error(
                RuntimeError("ScEnsemble.__init__() got an unexpected keyword argument 'solutions'")
            )
        )
        self.assertFalse(is_retryable_runtime_error(RuntimeError("name 're' is not defined")))

    def test_resolve_optimized_path_with_model_namespace(self):
        resolved = resolve_optimized_path(
            optimized_path="workspace",
            opt_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            exec_model_name="gpt-4o-mini",
            separate_model_artifacts=True,
            artifact_tag="smoke",
        )
        self.assertEqual(
            Path(resolved),
            Path("workspace") / "exec__gpt_4o_mini__opt__meta_llama_Meta_Llama_3_8B_Instruct__smoke",
        )

    def test_write_run_metadata(self):
        root = Path(__file__).resolve().parents[1] / ".tmp_tests" / str(uuid.uuid4())
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        write_run_metadata(
            optimized_path=str(root),
            dataset="MATH",
            opt_model_name="opt-model",
            exec_model_name="exec-model",
            artifact_tag="trial",
        )
        metadata_path = root / "MATH" / "run_metadata.json"
        self.assertTrue(metadata_path.exists())
        content = metadata_path.read_text(encoding="utf-8")
        self.assertIn('"opt_model_name": "opt-model"', content)
        self.assertIn('"exec_model_name": "exec-model"', content)

    def test_auto_initialize_dataset_template(self):
        repo_root = Path(__file__).resolve().parents[1]
        root = repo_root / ".tmp_tests" / str(uuid.uuid4())
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        dataset_root = ensure_dataset_template_initialized(
            optimized_path=str(root),
            dataset="MATH",
            template_root=str(repo_root / "workspace_template_ab"),
        )
        round_graph = Path(dataset_root) / "workflows" / "round_1" / "graph.py"
        template_operator = Path(dataset_root) / "workflows" / "template" / "operator.py"
        self.assertTrue(round_graph.exists())
        self.assertTrue(template_operator.exists())
        round_graph_text = round_graph.read_text(encoding="utf-8")
        self.assertIn("from ..template import operator", round_graph_text)
        self.assertIn("from . import prompt as prompt_custom", round_graph_text)
        template_operator_text = template_operator.read_text(encoding="utf-8")
        self.assertIn("from .operator_an import *", template_operator_text)
        self.assertIn("from .op_prompt import *", template_operator_text)

    def test_failure_summary_uses_compact_fields(self):
        root = Path(__file__).resolve().parents[1] / ".tmp_tests" / str(uuid.uuid4())
        log_dir = root / "workflows" / "round_1"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        write_json_file(
            log_dir / "log.json",
            [
                {
                    "question": "Q" * 500,
                    "right_answer": "A" * 400,
                    "model_output": "M" * 1000,
                    "extracted_output": "E" * 300,
                    "extract_answer_code": "def ignored(): pass",
                }
            ],
            encoding="utf-8",
            indent=2,
        )
        summary = DataUtils(str(root)).summarize_failures(1)
        self.assertIn("input_summary=", summary)
        self.assertNotIn("Question:", summary)
        self.assertNotIn("Model Output Preview:", summary)
        self.assertNotIn("extract_answer_code", summary)
        self.assertLess(len(summary), 1600)

    def test_workflow_summary_is_compact(self):
        summary, prompt_summary = GraphUtils("workspace/MATH").summarize_workflow(
            "class Workflow:\n    async def __call__(self, problem):\n        draft = await self.custom(problem)\n        answer = await self.ensemble(draft)\n        return answer\n",
            "XXX_PROMPT = '''Solve carefully.'''\nUNUSED_PROMPT = '''ignore'''\n",
            graph_char_limit=120,
            prompt_char_limit=80,
        )
        self.assertIn("Workflow nodes: custom, ensemble", summary)
        self.assertIn("Prompt vars: UNUSED_PROMPT, XXX_PROMPT", prompt_summary)
        self.assertLess(len(summary), 220)

    def test_experience_replacement_mode_prefers_memory(self):
        optimizer = Optimizer.__new__(Optimizer)
        optimizer.type = "math"
        optimizer.experience_guided_enabled = True
        optimizer.experience_guided_config = ExperienceGuidedConfig(
            enabled=True,
            store_path="unused",
            use_memory_as_experience_replacement=True,
        )
        context, mode = optimizer._build_experience_context(
            sample_round=1,
            old_experience="old experience " * 80,
            memory_guidance_text="[Memory-Guided Optimization]\nvalidated fix",
            memory_guidance=MemoryGuidance(),
        )
        self.assertEqual(mode, "memory_replacement")
        self.assertNotIn("old experience old", context)

    def test_code_tasks_keep_old_experience_in_hybrid_mode(self):
        optimizer = Optimizer.__new__(Optimizer)
        optimizer.type = "code"
        optimizer.experience_guided_enabled = True
        optimizer.experience_guided_config = ExperienceGuidedConfig(
            enabled=True,
            store_path="unused",
            use_memory_as_experience_replacement=True,
        )
        context, mode = optimizer._build_experience_context(
            sample_round=1,
            old_experience="old experience " * 80,
            memory_guidance_text="[Memory-Guided Optimization]\nvalidated fix",
            memory_guidance=MemoryGuidance(),
        )
        self.assertEqual(mode, "hybrid")
        self.assertIn("[Supporting Old Experience]", context)
        self.assertIn("old experience", context)

    def test_prompt_budget_and_memory_guided_prompt_structure(self):
        optimizer = Optimizer.__new__(Optimizer)
        optimizer.type = "math"
        optimizer.graph_utils = GraphUtils("workspace/MATH")
        optimizer.experience_guided_config = ExperienceGuidedConfig(
            enabled=True,
            store_path="unused",
            enable_prompt_budget_diagnostics=True,
        )
        budget_rows = []
        import scripts.optimizer as optimizer_module

        original_logger = optimizer_module.logger.info
        optimizer_module.logger.info = lambda msg: budget_rows.append(str(msg))
        try:
            prompt = optimizer._build_optimize_prompt(
                sample={"round": 1, "score": 0.2},
                workflow_summary="Workflow nodes: custom, review",
                prompt_summary="Prompt vars: XXX_PROMPT",
                operator_description="1. Custom: desc",
                failure_summaries='[FailureCard 1]\nextract_result=41\nmismatch_info=expected=42 vs extracted=41',
                trajectory_memory_text="[Trajectory Memory]\n- localized_failure_type: verification_missing",
                memory_guidance_text="[Memory-Guided Optimization]\nValidated memory:\n- fix=Insert explicit answer verification after b2:review.\nRejected memory:\n- avoid=Broad prompt expansion.",
                experience_context="[Memory-Guided Optimization]\nValidated memory:\n- fix=something",
                memory_guidance=MemoryGuidance(
                    focus_nodes=["b2:review"],
                    preserve_nodes=["b3:ensemble"],
                    avoid_nodes_or_patches=["Broad prompt expansion"],
                    recommended_patch_rules=["Insert explicit answer verification after b2:review."],
                ),
                guard_feedback_text="- Attempt 1: invalid graph rejected by guard because syntax_error_after_repair: invalid syntax.",
            )
        finally:
            optimizer_module.logger.info = original_logger
        self.assertTrue(any("[PromptBudget]" in row for row in budget_rows))
        self.assertIn("[Failure Evidence]", prompt)
        self.assertIn("[Memory-Guided Optimization]", prompt)
        self.assertNotIn('"model_output"', prompt)
        self.assertIn("[Recent Candidate Generation Failures]", prompt)

    def test_max_candidate_attempts_has_default(self):
        optimizer = Optimizer.__new__(Optimizer)
        optimizer.max_candidate_attempts_per_round = max(1, int(3))
        self.assertEqual(optimizer.max_candidate_attempts_per_round, 3)

    def test_code_tasks_skip_memory_guided_parent_selection(self):
        optimizer = Optimizer.__new__(Optimizer)
        optimizer.type = "code"
        rounds = [{"round": 1, "score": 0.5}, {"round": 2, "score": 0.6}]
        selected = optimizer._apply_memory_guided_parent_selection(rounds, "unused", {})
        self.assertEqual(selected, rounds)

    def test_code_tasks_diversify_parent_candidates_after_duplicate(self):
        optimizer = Optimizer.__new__(Optimizer)
        optimizer.type = "code"
        optimizer.sample = 1
        optimizer.experience_guided_enabled = False

        class StubDataUtils:
            def get_top_rounds(self, sample):
                self.last_sample = sample
                return [
                    {"round": 5, "score": 0.88},
                    {"round": 3, "score": 0.87},
                    {"round": 1, "score": 0.79},
                ][:sample]

        optimizer.data_utils = StubDataUtils()
        selected = optimizer._select_parent_round_candidates(
            graph_path="unused",
            processed_experience={},
            recent_generation_failures=[
                "Attempt 1: modification rejected because it repeats known history: same patch"
            ],
            blocked_parent_rounds=[5],
        )
        self.assertEqual(optimizer.data_utils.last_sample, 3)
        self.assertEqual([item["round"] for item in selected], [3, 1])

    def test_candidate_snapshot_is_persisted_and_updatable(self):
        repo_root = Path(__file__).resolve().parents[1]
        root = repo_root / ".tmp_tests" / str(uuid.uuid4()) / "artifact_space" / "MATH" / "workflows" / "round_9"
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(repo_root / ".tmp_tests", ignore_errors=True))
        optimizer = Optimizer.__new__(Optimizer)
        optimizer.round = 8
        saved_dir = optimizer._persist_candidate_snapshot(
            directory=str(root),
            candidate_attempt=2,
            sample_round=1,
            response={
                "modification": "add review node",
                "graph": "class Workflow:\n    pass\n",
                "prompt": 'XXX_PROMPT = "Solve"\n',
            },
        )
        optimizer._update_candidate_snapshot_status(
            saved_dir,
            status="rejected",
            rejection_kind="invalid_graph",
            rejection_reasons=["missing_async___call__"],
            guard_result=None,
        )
        saved_path = Path(saved_dir)
        self.assertTrue((saved_path / "graph.py").exists())
        self.assertTrue((saved_path / "prompt.py").exists())
        self.assertTrue((saved_path / "modification.txt").exists())
        metadata = (saved_path / "metadata.json").read_text(encoding="utf-8")
        self.assertIn('"status": "rejected"', metadata)
        self.assertIn('"rejection_kind": "invalid_graph"', metadata)
        self.assertIn('"candidate_attempt": 2', metadata)

    def test_extract_fields_sanitizes_graph_text(self):
        optimizer = Optimizer.__new__(Optimizer)
        from scripts.optimizer_utils.workflow_guard import WorkflowGuard

        optimizer.workflow_guard = WorkflowGuard(mode="repair_then_continue")
        raw = """
<modification>add verification step</modification>
<graph>
Here is the updated graph:
```python
class Workflow:
    async def __call__(self, problem: str):
        return "ok", self.llm.get_usage_summary()["total_cost"]
```
</graph>
<prompt>XXX_PROMPT = "Solve carefully."</prompt>
"""
        payload = optimizer._extract_fields_from_response(raw)
        self.assertIsNotNone(payload)
        self.assertTrue(payload["graph"].startswith("class Workflow:"))
        self.assertNotIn("```", payload["graph"])
        self.assertNotIn("Here is the updated graph", payload["graph"])

    def test_detect_degenerate_candidate_flags_single_custom_and_generic_prompt(self):
        reasons = Optimizer._detect_degenerate_candidate(
            response={
                "graph": """class Workflow:\n    async def __call__(self, problem: str):\n        response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)\n        return response['response'], self.llm.get_usage_summary()['total_cost']\n""",
                "prompt": 'XXX_PROMPT = """Solve it."""',
            },
            parent_graph="""class Workflow:\n    async def __call__(self, problem: str):\n        plan = await self.plan(problem=problem)\n        draft = await self.answer_generate(input=problem)\n        review = await self.review(problem=problem, solution='x')\n        return 'x', 0\n""",
            parent_prompt="Detailed prompt with answer formatting constraints.",
        )
        self.assertIn("collapsed_workflow_to_single_custom_call", reasons)
        self.assertIn("prompt_replaced_with_generic_solve_it", reasons)

    def test_write_graph_files_rewrites_managed_imports(self):
        repo_root = Path(__file__).resolve().parents[1]
        rel_root = Path(".tmp_tests") / str(uuid.uuid4()) / "artifact_space"
        abs_root = repo_root / rel_root
        dataset_root = rel_root / "MATH"
        workflows_root = abs_root / "MATH" / "workflows"
        workflows_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(repo_root / ".tmp_tests", ignore_errors=True))
        graph_utils = GraphUtils(str(dataset_root))
        out_dir = graph_utils.create_round_directory(str(workflows_root), 2)
        graph_utils.write_graph_files(
            out_dir,
            {
                "graph": """from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    async def __call__(self, problem: str):
        return "ok", self.llm.get_usage_summary()["total_cost"]
""",
                "prompt": 'XXX_PROMPT = "Solve carefully."',
            },
            round_number=2,
            dataset="MATH",
        )
        graph_text = (Path(out_dir) / "graph.py").read_text(encoding="utf-8")
        self.assertIn("from ..template import operator", graph_text)
        self.assertIn("from . import prompt as prompt_custom", graph_text)
        self.assertNotIn("workspace.MATH.workflows.round_1.prompt as prompt_custom", graph_text)
        prompt_text = (Path(out_dir) / "prompt.py").read_text(encoding="utf-8")
        self.assertIn('XXX_PROMPT = "Solve carefully."', prompt_text)


if __name__ == "__main__":
    unittest.main()
