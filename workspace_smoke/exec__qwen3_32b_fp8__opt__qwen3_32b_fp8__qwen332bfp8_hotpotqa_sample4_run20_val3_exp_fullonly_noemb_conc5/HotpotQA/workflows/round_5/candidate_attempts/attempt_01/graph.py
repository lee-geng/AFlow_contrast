class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        response = await self.custom(input=problem, instruction=prompt_custom.ANSWER_PROMPT_BOXED)
        answer = response["response"].strip().lower()
        if "yes" in answer:
            answer = "\\boxed{Yes}"
        elif "no" in answer:
            answer = "\\boxed{No}"
        else:
            answer = response["response"]  # fallback to raw if ambiguous
        return answer, self.llm.get_usage_summary()["total_cost"]