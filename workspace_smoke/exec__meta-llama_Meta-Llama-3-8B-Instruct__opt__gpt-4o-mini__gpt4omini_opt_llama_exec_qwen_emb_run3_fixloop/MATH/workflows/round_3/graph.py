from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_3.prompt as prompt_custom
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
        self.review = operator.Review(self.llm)  # Added review operator

    async def __call__(self, problem: str):
        """ Implementation of the workflow """
        # Generate solution using custom operator
        solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        
        # Review the generated solution
        review_result = await self.review(problem=problem, solution=solution['response'])
        
        # Check review result and provide feedback if necessary
        if review_result['review_result'] == 'incorrect':
            return review_result['feedback'], self.llm.get_usage_summary()["total_cost"]
        
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
