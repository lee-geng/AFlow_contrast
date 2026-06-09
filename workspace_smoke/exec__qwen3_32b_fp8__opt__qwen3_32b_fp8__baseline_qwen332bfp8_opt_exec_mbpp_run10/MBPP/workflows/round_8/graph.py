from typing import Literal
from ..template import operator
from . import prompt as prompt_custom
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
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        # Generate 3 candidate solutions
        sol1 = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
        sol2 = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
        sol3 = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")

        # Use ScEnsemble to select the most consistent solution
        solutions = [sol1["response"], sol2["response"], sol3["response"]]
        ensemble_result = await self.sc_ensemble(solutions=solutions, problem=problem)

        # Test the selected solution
        tested = await self.test(problem=problem, solution=ensemble_result["response"], entry_point=entry_point)
        return tested["solution"], self.llm.get_usage_summary()["total_cost"]
