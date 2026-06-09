class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        # First, use the custom operator to generate a raw solution
        raw_solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        # Then, use the answer_generate operator to structure the solution into a thought and final answer
        structured_solution = await self.answer_generate(input=raw_solution["response"])
        return structured_solution["answer"], self.llm.get_usage_summary()["total_cost"]