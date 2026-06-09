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
        # Extract structured or numerical information from the input
        structured_input = await self.custom(input=problem, instruction=prompt_custom.STRUCTURE_EXTRACTION_PROMPT)
        # Generate solution based on the structured input
        solution = await self.answer_generate(input=structured_input["response"])
        # Refine the answer using self-consistency
        refined_answer = await self.sc_ensemble(solutions=[solution["answer"]])
        return refined_answer["response"], self.llm.get_usage_summary()["total_cost"]