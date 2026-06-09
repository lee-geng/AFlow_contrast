from typing import Literal
from ..template import operator
from . import prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

from typing import Literal, List

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.review = operator.Review(self.llm)

    async def __call__(self, problem: str):
        """ Implementation of the workflow """
        # Generate a solution using the custom operator
        response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        solution = response['response']
        
        # Review the generated solution
        review_result = await self.review(problem=problem, solution=solution)
        
        # Check if the review indicates the solution is correct
        if review_result['review_result'] == 'correct':
            return solution, self.llm.get_usage_summary()["total_cost"]
        else:
            # If not correct, provide feedback and revise if necessary
            feedback = review_result['feedback']
            revised_solution = await self.custom(input=problem, instruction=feedback)
            return revised_solution['response'], self.llm.get_usage_summary()["total_cost"]
