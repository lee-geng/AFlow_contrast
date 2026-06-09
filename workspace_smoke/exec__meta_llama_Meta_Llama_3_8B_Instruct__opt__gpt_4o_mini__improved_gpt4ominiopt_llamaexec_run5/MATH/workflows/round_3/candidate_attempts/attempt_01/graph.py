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
        draft_input = problem
        if plan_text:
            draft_input = f"Problem:\n{problem}\n\nPlan:\n{plan_text}"
        draft = await self.answer_generate(input=draft_input)
        draft_solution = str(draft.get("thought", "")).strip()
        draft_answer = str(draft.get("answer", "")).strip()
        combined_solution = draft_solution
        if draft_answer:
            combined_solution = f"{draft_solution}\n\nFinal answer: {draft_answer}".strip()
        
        # New verification step added here
        verification = await self.verify(problem=problem, solution=combined_solution)
        if verification['verdict'] == 'incorrect':
            combined_solution = verification['corrected_answer']
        
        review = await self.review(problem=problem, solution=combined_solution)
        feedback = review.get("feedback", "")
        if feedback:
            revised_solution = await self.revise(problem=problem, solution=combined_solution, feedback=feedback)
            combined_solution = revised_solution['solution']
        
        final_answer = await self.extract_answer(problem=problem, solution=combined_solution)
        return final_answer['answer'], self.llm.get_usage_summary()["total_cost"]