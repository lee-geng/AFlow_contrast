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
        self.review = operator.Review(self.llm)  # Adding review operator

    async def __call__(self, problem: str):
        """ Implementation of the workflow """
        solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        review_result = await self.review(problem=problem, solution=solution['response'])  # Review the solution
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
