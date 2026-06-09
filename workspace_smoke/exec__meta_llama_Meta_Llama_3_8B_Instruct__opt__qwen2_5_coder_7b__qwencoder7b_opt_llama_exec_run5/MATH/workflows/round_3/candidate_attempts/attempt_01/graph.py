class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        if "expected" in response["response"] and "extracted" in response["response"]:
            expected = int(response["response"].split("expected=")[1].split()[0])
            extracted = int(response["response"].split("extracted=")[1].split()[0])
            if expected != extracted:
                revised_response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT + " Revise the answer to match the expected result.")
                response["response"] = revised_response["response"]
        return response["response"], self.llm.get_usage_summary()["total_cost"]