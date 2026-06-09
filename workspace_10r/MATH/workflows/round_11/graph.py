from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_11.prompt as prompt_custom
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
        # Generate initial solution using custom method
        solution = await self.custom(input=problem, instruction="")
        
        # Verify calculations using the programmer operator
        code = f"""
# Python code to verify the probability calculation
from math import comb

total_ways = comb(52, 5)
one_suit_ways = 4 * comb(13, 5)
two_suit_ways = 6 * (comb(13, 4) * 13 + comb(13, 3) * comb(13, 2) * 2)
fewer_than_three_suits = one_suit_ways + two_suit_ways
at_least_three_suits = total_ways - fewer_than_three_suits

# Return the simplified fraction
from fractions import Fraction
result = Fraction(at_least_three_suits, total_ways).limit_denominator()
result
"""
        verification_result = await self.programmer(problem=code)
        
        return solution['response'], verification_result['output'], self.llm.get_usage_summary()["total_cost"]
