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
        # Refine the problem input to clarify ambiguity or missing context
        refined_problem = await self.custom(input=problem, instruction=prompt_custom.REFINE_PROBLEM_PROMPT)
        # Generate solution based on the refined problem
        solution = await self.answer_generate(input=refined_problem["response"])
        # Use self-consistency to refine the final answer
        refined_answer = await self.sc_ensemble(solutions=[solution["answer"]])
        return refined_answer["response"], self.llm.get_usage_summary()["total_cost"]
