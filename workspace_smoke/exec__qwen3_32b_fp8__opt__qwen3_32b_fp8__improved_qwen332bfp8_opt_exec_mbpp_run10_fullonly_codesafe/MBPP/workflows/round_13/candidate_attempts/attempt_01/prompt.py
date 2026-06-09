from typing import Literal
from ..template import operator
from . import prompt as prompt_custom
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self, name: str, llm_config, dataset: DatasetType
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.test = operator.Test(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        solution = await self.custom_code_generate(
            problem=problem,
            entry_point=entry_point,
            instruction=prompt_custom.XXX_PROMPT
        )
        # Apply ScEnsemble to refine the solution before testing
        refined_solution = await self.sc_ensemble(
            solutions=[solution["response"]],
            problem=problem
        )
        test_result = await self.test(
            problem=problem,
            solution=refined_solution["response"],
            entry_point=entry_point
        )
        return test_result["solution"], self.llm.get_usage_summary()["total_cost"]


# Prompt modification to ensure complete and testable code is generated
XXX_PROMPT = """Write a complete and correct Python function to solve the problem. Include the full function definition and any necessary test cases to verify correctness."""