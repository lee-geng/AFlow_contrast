from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_7.prompt as prompt_custom
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
        self.programmer = operator.Programmer()

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate a solution using the Programmer operator
        code_response = await self.programmer(problem=problem)
        solution = await self.custom(input=code_response['output'], instruction="")
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
