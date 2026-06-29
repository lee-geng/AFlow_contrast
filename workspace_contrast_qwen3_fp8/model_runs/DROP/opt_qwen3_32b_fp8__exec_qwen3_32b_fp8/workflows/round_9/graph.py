from typing import Literal
import workspace_contrast_qwen3_fp8.model_runs.DROP.opt_qwen3_32b_fp8__exec_qwen3_32b_fp8.workflows.template.operator as operator
import workspace_contrast_qwen3_fp8.model_runs.DROP.opt_qwen3_32b_fp8__exec_qwen3_32b_fp8.workflows.round_9.prompt as prompt_custom
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
        self.disambiguate_and_clarify = operator.Custom(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.answer_format = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Disambiguate and clarify the question to extract key entities
        clarified_problem = await self.disambiguate_and_clarify(input=problem, instruction=prompt_custom.DISAMBIGUATE_AND_CLARIFY_PROMPT)
        # Generate multiple solutions using answer_generate
        solutions = [
            await self.answer_generate(input=clarified_problem["response"]),
            await self.answer_generate(input=clarified_problem["response"]),
            await self.answer_generate(input=clarified_problem["response"])
        ]
        # Extract answers from the generated solutions
        solution_list = [sol["answer"] for sol in solutions]
        # Use self-consistency to select the most consistent answer
        final_answer = await self.sc_ensemble(solutions=solution_list)
        # Format the final answer to match the expected format
        formatted_answer = await self.answer_format(input=final_answer["response"], instruction=prompt_custom.FORMAT_ANSWER_PROMPT)
        return formatted_answer["response"], self.llm.get_usage_summary()["total_cost"]
