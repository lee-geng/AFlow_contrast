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
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        solution = await self.answer_generate(input=problem)
        normalized_answer = await self.custom(
            input=solution["answer"],
            instruction=prompt_custom.NORMALIZE_ANSWER_PROMPT
        )
        refined_solution = await self.sc_ensemble(solutions=[normalized_answer["response"]])
        return refined_solution["response"], self.llm.get_usage_summary()["total_cost"]
