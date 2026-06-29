from typing import Literal
import workspace_patch_qwen3_fp8.model_runs.DROP.opt_qwen3_32b_fp8__exec_qwen3_32b_fp8.workflows.template.operator as operator
import workspace_patch_qwen3_fp8.model_runs.DROP.opt_qwen3_32b_fp8__exec_qwen3_32b_fp8.workflows.round_2.prompt as prompt_custom
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
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate 3 different solutions using the custom method
        solutions = []
        for _ in range(3):
            solution = await self.custom(input=problem, instruction=prompt_custom.MULTI_ANSWER_PROMPT)
            solutions.append(solution['response'])

        # Use ScEnsemble to select the most consistent solution
        final_solution = await self.sc_ensemble(solutions=solutions)
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]
