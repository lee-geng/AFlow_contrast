from typing import Literal
from ..template import operator
from . import prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

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
        solution = await self.custom(input=problem, instruction="")
        
        # Review the generated solution for correctness
        review_result = await self.review(problem=problem, solution=solution['response'])
        
        # If the review indicates the solution is not correct, handle accordingly
        if review_result['review_result'] != 'correct':
            # Optionally, you could revise the solution based on feedback
            feedback = review_result['feedback']
            # Here we could implement a revise step if needed
            # For now, we will just return the review feedback
            return solution['response'], self.llm.get_usage_summary()["total_cost"]
        
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
