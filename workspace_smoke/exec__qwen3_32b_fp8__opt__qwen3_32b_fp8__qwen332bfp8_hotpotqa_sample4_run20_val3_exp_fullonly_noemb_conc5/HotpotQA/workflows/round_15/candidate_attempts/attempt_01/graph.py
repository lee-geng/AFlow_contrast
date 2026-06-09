class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        response1 = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        response2 = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT_REFINED)
        solutions = [response1["response"], response2["response"]]
        final = await self.sc_ensemble(solutions=solutions)
        return final["response"], self.llm.get_usage_summary()["total_cost"]