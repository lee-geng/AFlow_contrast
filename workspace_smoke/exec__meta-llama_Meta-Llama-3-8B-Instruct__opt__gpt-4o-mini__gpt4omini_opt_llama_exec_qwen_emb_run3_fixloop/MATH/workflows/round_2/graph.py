from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_2.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """ Implementation of the workflow """
        # Enhanced input handling for better alignment with ground truth
        instruction = prompt_custom.XXX_PROMPT
        response = await self.custom(input=problem, instruction=instruction)
        return response['response'], self.llm.get_usage_summary()["total_cost"]
