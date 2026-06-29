from typing import Literal
import workspace_contrast_qwen3_fp8.model_runs.DROP.opt_qwen3_32b_fp8__exec_qwen3_32b_fp8.workflows.template.operator as operator
import workspace_contrast_qwen3_fp8.model_runs.DROP.opt_qwen3_32b_fp8__exec_qwen3_32b_fp8.workflows.round_4.prompt as prompt_custom
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
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.answer_format = operator.Custom(self.llm)
        self.answer_review = operator.Custom(self.llm)  # New operator for review

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple solutions using answer_generate
        solutions = [
            await self.answer_generate(input=problem),
            await self.answer_generate(input=problem),
            await self.answer_generate(input=problem)
        ]
        # Extract answers from the generated solutions
        solution_list = [sol["answer"] for sol in solutions]
        # Use self-consistency to select the most consistent answer
        final_answer = await self.sc_ensemble(solutions=solution_list)
        # Review the final answer for correctness and alignment with the question
        reviewed_answer = await self.answer_review(
            input=problem + f"\nProposed Answer: {final_answer['response']}",
            instruction=prompt_custom.REVIEW_ANSWER_PROMPT
        )
        # Format the final answer to match the expected format
        formatted_answer = await self.answer_format(
            input=reviewed_answer["response"],
            instruction=prompt_custom.FORMAT_ANSWER_PROMPT
        )
        return formatted_answer["response"], self.llm.get_usage_summary()["total_cost"]
