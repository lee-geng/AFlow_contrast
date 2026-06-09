from typing import Literal
import .tmp_tests.0357b8b3-2da5-4a99-84d8-504aff628f4d.artifact_space.MATH.workflows.template.operator as operator
import .tmp_tests.0357b8b3-2da5-4a99-84d8-504aff628f4d.artifact_space.MATH.workflows.round_2.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    async def __call__(self, problem: str):
        return "ok", self.llm.get_usage_summary()["total_cost"]
