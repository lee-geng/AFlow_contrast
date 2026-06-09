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
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        solution = await self.custom_code_generate(
            problem=problem,
            entry_point=entry_point,
            instruction=prompt_custom.CODE_GEN_PROMPT
        )
        test_result = await self.test(
            problem=problem,
            solution=solution["response"],
            entry_point=entry_point
        )
        if test_result["result"]:
            return test_result["solution"], self.llm.get_usage_summary()["total_cost"]
        else:
            return test_result["solution"], self.llm.get_usage_summary()["total_cost"]
