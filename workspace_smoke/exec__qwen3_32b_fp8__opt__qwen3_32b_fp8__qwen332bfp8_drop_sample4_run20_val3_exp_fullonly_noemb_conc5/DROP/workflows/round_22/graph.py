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
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        solutions = []
        for _ in range(3):
            raw_solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
            solutions.append(raw_solution["response"])
        ensemble_solution = await self.sc_ensemble(solutions=solutions)
        structured_solution = await self.answer_generate(input=ensemble_solution["response"])
        return structured_solution["answer"], self.llm.get_usage_summary()["total_cost"]
