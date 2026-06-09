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
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        plan_result = await self.plan(problem=problem)
        plan_text = str(plan_result.get("plan", "")).strip()
        draft_input = problem if plan_text == "" else f"Problem:\n{problem}\n\nPlan:\n{plan_text}\n\nContext:\n{response['response']}"
        draft = await self.answer_generate(input=draft_input)
        draft_solution = str(draft.get("thought", "")).strip()
        draft_answer = str(draft.get("answer", "")).strip()
        combined_solution = draft_solution if draft_answer == "" else f"{draft_solution}\n\nFinal answer: {draft_answer}".strip()
        review = await self.review(problem=problem, solution=combined_solution)
        if review['review_result'] == 'incorrect':
            revised_solution = await self.revise(problem=problem, solution=combined_solution, feedback=review['feedback'])
            final_verification = await self.verify(problem=problem, solution=revised_solution['solution'])
            return final_verification['corrected_answer'], self.llm.get_usage_summary()["total_cost"]
        return combined_solution, self.llm.get_usage_summary()["total_cost"]
