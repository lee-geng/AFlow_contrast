import unittest

from scripts.optimizer_utils.workflow_guard import WorkflowGuard


class WorkflowGuardTests(unittest.TestCase):
    def test_auto_repair_programmer_and_return_shape(self):
        graph = """
class Workflow:
    async def __call__(self, problem: str):
        self.programmer = operator.Programmer()
        solution = await self.custom(input=problem, instruction="")
        code_solution = await self.programmer(problem=problem, analysis=solution['response'])
        return solution['response'], code_solution['output'], self.llm.get_usage_summary()["total_cost"]
"""
        guard = WorkflowGuard(mode="repair_then_continue")
        out = guard.ensure(graph)
        self.assertTrue(out.valid)
        self.assertTrue(out.repaired)
        self.assertIn("operator.Programmer(self.llm)", out.graph)
        self.assertIn('return solution[\'response\'], self.llm.get_usage_summary()["total_cost"]', out.graph)

    def test_fail_when_missing_workflow(self):
        graph = "class NotWorkflow:\n    pass\n"
        guard = WorkflowGuard(mode="repair_then_continue")
        out = guard.ensure(graph)
        self.assertFalse(out.valid)
        self.assertIn("missing_Workflow_class", out.issues)

    def test_auto_repair_ensemble_constructor_runtime_kwargs(self):
        graph = """
class Workflow:
    def __init__(self):
        self.ensemble = operator.ScEnsemble(self.llm, solutions=solutions, problem=problem)
    async def __call__(self, problem: str):
        return "ok", self.llm.get_usage_summary()["total_cost"]
"""
        guard = WorkflowGuard(mode="repair_then_continue")
        out = guard.ensure(graph)
        self.assertTrue(out.valid)
        self.assertTrue(out.repaired)
        self.assertIn("operator.ScEnsemble(self.llm)", out.graph)

    def test_sanitize_markdown_fence_and_preface_text(self):
        graph = """
Here is the updated graph:
```python
from typing import Literal

class Workflow:
    async def __call__(self, problem: str):
        return "ok", self.llm.get_usage_summary()["total_cost"]
```
"""
        guard = WorkflowGuard(mode="repair_then_continue")
        out = guard.ensure(graph)
        self.assertTrue(out.valid)
        self.assertTrue(out.repaired)
        self.assertTrue(out.graph.lstrip().startswith("from typing import Literal"))
        self.assertNotIn("```", out.graph)
        self.assertNotIn("Here is the updated graph", out.graph)

    def test_sanitize_trailing_non_code_text_after_workflow(self):
        graph = """
class Workflow:
    async def __call__(self, problem: str):
        return "ok", self.llm.get_usage_summary()["total_cost"]

This patch focuses on verification.
"""
        guard = WorkflowGuard(mode="repair_then_continue")
        out = guard.ensure(graph)
        self.assertTrue(out.valid)
        self.assertTrue(out.repaired)
        self.assertNotIn("This patch focuses on verification.", out.graph)


if __name__ == "__main__":
    unittest.main()
