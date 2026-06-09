from typing import Literal
import F:.Code.AFlow_contrast..tmp_tests.777cc383-e5ca-4028-b8a3-dd4b0e8200ad.MATH.workflows.template.operator as operator
import F:.Code.AFlow_contrast..tmp_tests.777cc383-e5ca-4028-b8a3-dd4b0e8200ad.MATH.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance

from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.custom(input=problem, instruction="")
        return solution['response'], self.llm.get_usage_summary()["total_cost"]