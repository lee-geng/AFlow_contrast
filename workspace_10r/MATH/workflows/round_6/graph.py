from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_6.prompt as prompt_custom
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
        # Generate the solution using the custom method
        solution = await self.custom(input=problem, instruction="")
        
        # Use the programmer to calculate the probability
        code = f"""
def calculate_probability():
    from math import comb
    total_ways = comb(52, 3)
    same_color_ways = 2 * comb(26, 3)
    return 1 - (same_color_ways / total_ways)

result = calculate_probability()
"""
        execution_result = await self.programmer(problem=code)
        return solution['response'], execution_result['output'], self.llm.get_usage_summary()["total_cost"]
