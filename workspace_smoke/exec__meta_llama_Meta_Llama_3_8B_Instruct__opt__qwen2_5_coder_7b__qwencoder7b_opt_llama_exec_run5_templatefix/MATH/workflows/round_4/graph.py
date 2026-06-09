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
        self.plan = operator.Plan(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.review = operator.Review(self.llm)
        self.revise = operator.Revise(self.llm)
        self.verify = operator.Verify(self.llm)
        self.extract_answer = operator.ExtractAnswer(self.llm)

    async def __call__(self, problem: str):
        plan_result = await self.plan(problem=problem)
        plan_text = str(plan_result.get("plan", "")).strip()
        draft_input = problem if plan_text else f"Problem:\n{problem}"
        draft = await self.answer_generate(input=draft_input)
        draft_solution = str(draft.get("thought", "")).strip()
        draft_answer = str(draft.get("answer", "")).strip()
        combined_solution = draft_solution if draft_answer else f"{draft_solution}\n\nFinal answer: {draft_answer}".strip()
        review_result = await self.review(problem=problem, solution=combined_solution)
        if review_result["review_result"] == "incorrect":
            revised_solution = await self.revise(problem=problem, solution=combined_solution, feedback=review_result["feedback"])
            combined_solution = revised_solution
        verified_result = await self.verify(problem=problem, solution=combined_solution)
        final_answer = await self.extract_answer(problem=problem, solution=verified_result["corrected_answer"])
        return final_answer["answer"], self.llm.get_usage_summary()["total_cost"]
