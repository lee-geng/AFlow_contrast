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
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        solution = await self.answer_generate(input=problem)
        extracted_answer = await self.custom(input=solution["answer"], instruction=prompt_custom.EXTRACT_ANSWER_PROMPT)
        aligned_answer = await self.custom(input=extracted_answer["response"], instruction=prompt_custom.ANSWER_ALIGNMENT_PROMPT)
        return aligned_answer["response"], self.llm.get_usage_summary()["total_cost"]