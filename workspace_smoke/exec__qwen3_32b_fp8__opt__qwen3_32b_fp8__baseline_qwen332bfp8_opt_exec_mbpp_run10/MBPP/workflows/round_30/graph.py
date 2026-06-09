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
        self.test = operator.Test(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        # Generate multiple candidate solutions
        solutions = []
        for _ in range(3):  # Generate 3 solutions for ensemble
            solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(solution["response"])

        # Use ScEnsemble to select the most consistent solution
        ensemble_solution = await self.sc_ensemble(solutions=solutions, problem=problem)

        # Post-processing refinement to ensure syntactic completeness and correct test function
        refined_solution = await self.custom(
            input=ensemble_solution["response"],
            instruction="Ensure the code is syntactically complete and includes a correct test function."
        )

        # Test the refined solution
        tested = await self.test(problem=problem, solution=refined_solution["response"], entry_point=entry_point)
        return tested["solution"], self.llm.get_usage_summary()["total_cost"]
