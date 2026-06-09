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

    async def __call__(self, problem: str):
        response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        # Add a check to ensure the response is not empty before returning it
        if response["response"]:
            return response["response"], self.llm.get_usage_summary()["total_cost"]
        else:
            # If the response is empty, fall back to the original workflow
            return await self.original_workflow(problem)

    async def original_workflow(self, problem: str):
        # Original workflow implementation
        # ...
        return final_answer, self.llm.get_usage_summary()["total_cost"]
