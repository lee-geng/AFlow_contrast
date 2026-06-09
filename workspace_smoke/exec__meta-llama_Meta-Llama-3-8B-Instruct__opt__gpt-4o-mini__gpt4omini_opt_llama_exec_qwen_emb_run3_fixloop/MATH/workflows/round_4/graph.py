from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_4.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.review = operator.Review(self.llm)  # Added Review operator

    async def __call__(self, problem: str):
        """ Implementation of the workflow """
        # Generate a solution using the custom operator
        solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        
        # Review the generated solution
        review_result = await self.review(problem=problem, solution=solution['response'])
        
        # Check if the review indicates the solution is correct
        if review_result['review_result'] == 'correct':
            return solution['response'], self.llm.get_usage_summary()["total_cost"]
        else:
            # If the solution is not correct, provide feedback
            feedback = review_result['feedback']
            return f"Solution not validated: {feedback}", self.llm.get_usage_summary()["total_cost"]
