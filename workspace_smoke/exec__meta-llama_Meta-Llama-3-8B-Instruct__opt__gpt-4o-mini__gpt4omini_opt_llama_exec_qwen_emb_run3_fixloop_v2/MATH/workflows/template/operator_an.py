# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 19:46 PM
# @Author  : didi
# @Desc    : action nodes for operator

from pydantic import BaseModel, Field


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")


class PlanOp(BaseModel):
    plan: str = Field(default="", description="A concise step-by-step plan for solving this problem")


class AnswerGenerateOp(BaseModel):
    thought: str = Field(default="", description="The step-by-step reasoning process")
    answer: str = Field(default="", description="The final answer to the question")


class CodeGenerateOp(BaseModel):
    code: str = Field(default="", description="Your complete code solution for this problem")


class ScEnsembleOp(BaseModel):
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")


class ReviewOp(BaseModel):
    review_result: bool = Field(
        default=False,
        description="Whether the solution is likely correct. Return true unless you find a concrete issue.",
    )
    feedback: str = Field(
        default="",
        description="Concrete feedback about any issue found in the solution, or a short confirmation if it looks sound.",
    )


class ReviseOp(BaseModel):
    solution: str = Field(default="", description="A revised solution that addresses the feedback")


class VerifyOp(BaseModel):
    verdict: bool = Field(
        default=False,
        description="Whether the final answer is supported by the reasoning and constraints of the problem.",
    )
    reasoning: str = Field(
        default="",
        description="A concise explanation of the verification result.",
    )
    corrected_answer: str = Field(
        default="",
        description="The corrected final answer if the previous answer is unsupported; otherwise repeat the validated answer.",
    )


class ExtractAnswerOp(BaseModel):
    answer: str = Field(
        default="",
        description="A short final answer extracted from the longer solution, with no extra explanation.",
    )

