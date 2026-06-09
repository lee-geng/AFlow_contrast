class Workflow:
    def __init__( self, name: str, llm_config, dataset: DatasetType, ) -> None: 
        self.name = name 
        self.dataset = dataset 
        self.llm = create_llm_instance(llm_config) 
        self.custom = operator.Custom(self.llm) 
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.test = operator.Test(self.llm)

    async def __call__(self, problem: str, entry_point: str): 
        """ Implementation of the workflow Custom operator to generate anything you want. But when you want to get standard code, you should use custom_code_generate operator. """ 
        # Generate code
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.CODE_GEN_PROMPT) 

        # Test the generated code
        test_result = await self.test(problem=problem, solution=solution["response"], entry_point=entry_point)

        # Return the final solution and cost
        return test_result["solution"], self.llm.get_usage_summary()["total_cost"]