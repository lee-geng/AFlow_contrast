from typing import Literal
import workspace_smoke.exec__meta_llama_Meta_Llama_3_8B_Instruct__opt__gpt_4o_mini__gpt4omini_opt_llama_exec_qwen_emb_run3_fixloop_v3.MATH.workflows.template.operator as operator
import workspace_smoke.exec__meta_llama_Meta_Llama_3_8B_Instruct__opt__gpt_4o_mini__gpt4omini_opt_llama_exec_qwen_emb_run3_fixloop_v3.MATH.workflows.round_4.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.review = operator.Review(self.llm)

    async def __call__(self, problem: str):
        """ Implementation of the workflow """
        # Generate a solution using the Custom operator
        solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        
        # Review the generated solution
        review_result = await self.review(problem=problem, solution=solution['response'])
        
        # If the review indicates issues, we can revise the solution
        if review_result['review_result'] != 'correct':
            feedback = review_result['feedback']
            revised_solution = await self.custom(input=problem, instruction=feedback)
            return revised_solution['response'], self.llm.get_usage_summary()["total_cost"]
        
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
