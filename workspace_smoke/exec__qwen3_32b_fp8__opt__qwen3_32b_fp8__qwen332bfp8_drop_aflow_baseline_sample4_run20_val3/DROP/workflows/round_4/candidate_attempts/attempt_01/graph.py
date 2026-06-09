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

    async def __call__(self, problem: str):
        # Generate multiple solutions
        solution1 = await self.answer_generate(input=problem)
        solution2 = await self.answer_generate(input=problem)
        solution3 = await self.answer_generate(input=problem)

        # Use ScEnsemble to select the most consistent answer
        solutions = [solution1["answer"], solution2["answer"], solution3["answer"]]
        ensemble_result = await self.sc_ensemble(solutions=solutions)

        return ensemble_result["response"], self.llm.get_usage_summary()["total_cost"]