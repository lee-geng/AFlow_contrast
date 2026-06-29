from typing import Literal
import workspace_patch_qwen3_fp8.model_runs.DROP.opt_qwen3_32b_fp8__exec_qwen3_32b_fp8.workflows.template.operator as operator
import workspace_patch_qwen3_fp8.model_runs.DROP.opt_qwen3_32b_fp8__exec_qwen3_32b_fp8.workflows.round_3.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

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
        self.answer_gen = operator.AnswerGenerate(self.llm)
        self.ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple potential answers using AnswerGenerate
        solutions = [
            await self.answer_gen(input=problem),
            await self.answer_gen(input=problem),
            await self.answer_gen(input=problem)
        ]
        # Extract the 'answer' field from each solution
        solution_list = [sol['answer'] for sol in solutions]
        # Use ScEnsemble to select the most consistent answer
        final_answer = await self.ensemble(solutions=solution_list)
        return final_answer['response'], self.llm.get_usage_summary()["total_cost"]
