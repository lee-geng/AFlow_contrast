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
        custom_response = await self.custom(input=problem, instruction=XXX_PROMPT)
        custom_input = custom_response['response']

        plan_result = await self.plan(problem=custom_input)
        plan_text = str(plan_result.get("plan", "")).strip()

        draft_input = custom_input
        if plan_text:
            draft_input = f"Problem:\n{custom_input}\n\nPlan:\n{plan_text}"
        draft = await self.answer_generate(input=draft_input)
        draft_solution = str(draft.get("thought", "")).strip()
        draft_answer = str(draft.get("answer", "")).strip()
        combined_solution = draft_solution
        if draft_answer:
            combined_solution = f"{draft_solution}\n\nFinal answer: {draft_answer}".strip()

        review = await self.review(problem=custom_input, solution=combined_solution)
        if not review.get("review_result", True):
            revised = await self.revise(
                problem=custom_input,
                solution=combined_solution,
                feedback=str(review.get("feedback", "")).strip(),
            )
            combined_solution = str(revised.get("solution", combined_solution)).strip()

        verification = await self.verify(problem=custom_input, solution=combined_solution)
        final_solution = combined_solution
        corrected_answer = str(verification.get("corrected_answer", "")).strip()
        if verification.get("verdict", False) and corrected_answer:
            final_solution = f"{combined_solution}\n\nValidated answer: {corrected_answer}".strip()

        extracted = await self.extract_answer(problem=custom_input, solution=final_solution)
        final_answer = str(extracted.get("answer", "")).strip() or corrected_answer or final_solution
        return final_answer, self.llm.get_usage_summary()["total_cost"]