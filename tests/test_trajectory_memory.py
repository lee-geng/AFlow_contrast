import json
import shutil
import unittest
import uuid
from pathlib import Path

from scripts.optimizer_utils.trajectory_memory import TrajectoryMemoryController


class TrajectoryMemoryTests(unittest.TestCase):
    def _mk_tmp(self) -> Path:
        root = Path(__file__).resolve().parents[1] / ".tmp_tests" / str(uuid.uuid4())
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _write_round(self, workflows: Path, rid: int, graph: str, exp: dict, log_payload: dict) -> None:
        rd = workflows / f"round_{rid}"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "graph.py").write_text(graph, encoding="utf-8")
        (rd / "__init__.py").write_text("", encoding="utf-8")
        if exp:
            (rd / "experience.json").write_text(json.dumps(exp, ensure_ascii=False, indent=2), encoding="utf-8")
        (rd / "log.json").write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_compile_distill_and_use(self):
        tmp = self._mk_tmp()
        workflows = tmp / "workflows"
        workflows.mkdir(parents=True, exist_ok=True)

        round1_graph = (
            "class Workflow:\n"
            "    async def __call__(self, problem):\n"
            "        x = await self.retrieve(problem)\n"
            "        y = await self.generate(x)\n"
            "        return y\n"
        )
        round2_graph = (
            "class Workflow:\n"
            "    async def __call__(self, problem):\n"
            "        x = await self.retrieve(problem)\n"
            "        v = await self.review(x)\n"
            "        y = await self.generate(v)\n"
            "        return y\n"
        )
        round3_graph = (
            "class Workflow:\n"
            "    async def __call__(self, problem):\n"
            "        y = await self.generate(problem)\n"
            "        return y\n"
        )

        self._write_round(
            workflows,
            1,
            round1_graph,
            exp={},
            log_payload={"failure_summary": "unsupported answer without verification", "trace_summary": "retrieve|generate"},
        )
        self._write_round(
            workflows,
            2,
            round2_graph,
            exp={"father node": 1, "modification": "insert review verification block", "before": 0.50, "after": 0.62, "succeed": True},
            log_payload={"failure_summary": "", "trace_summary": "retrieve|review|generate"},
        )
        self._write_round(
            workflows,
            3,
            round3_graph,
            exp={"father node": 1, "modification": "remove retrieval and simplify prompt", "before": 0.50, "after": 0.43, "succeed": False},
            log_payload={"failure_summary": "context missing and unsupported", "trace_summary": "generate"},
        )

        results = [
            {"round": 1, "score": 0.50, "avg_cost": 0.12, "total_cost": 0.12},
            {"round": 2, "score": 0.62, "avg_cost": 0.14, "total_cost": 0.14},
            {"round": 3, "score": 0.43, "avg_cost": 0.08, "total_cost": 0.08},
        ]
        (workflows / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        controller = TrajectoryMemoryController(min_support=1, max_patterns=20)
        artifacts = controller.refresh(workflows_dir=str(workflows), task_context="MATH")

        self.assertGreaterEqual(len(artifacts.success_cases), 1)
        self.assertGreaterEqual(len(artifacts.failure_cases), 1)
        self.assertGreaterEqual(len(artifacts.contrast_records), 2)
        self.assertGreaterEqual(len(artifacts.failure_signatures), 1)
        self.assertGreaterEqual(len(artifacts.success_patterns), 1)
        self.assertGreaterEqual(len(artifacts.edit_rules), 1)

        memory_dir = workflows / "memory"
        self.assertTrue((memory_dir / "success_cases.json").exists())
        self.assertTrue((memory_dir / "failure_cases.json").exists())
        self.assertTrue((memory_dir / "contrast_records.json").exists())
        self.assertTrue((memory_dir / "failure_signatures.json").exists())
        self.assertTrue((memory_dir / "success_patterns.json").exists())
        self.assertTrue((memory_dir / "edit_rules.json").exists())

        suggestion = controller.suggest_for_workflow(
            task_context="MATH",
            workflow_graph=round1_graph,
            execution_feedback={
                "failure_summary": "answer unsupported without verification",
                "trace_summary": "retrieve then generate",
            },
        )
        self.assertIn("selected_edit", suggestion)
        self.assertIn("candidate_edit", suggestion["selected_edit"])


if __name__ == "__main__":
    unittest.main()
