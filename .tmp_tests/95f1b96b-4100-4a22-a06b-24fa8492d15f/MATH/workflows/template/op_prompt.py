PLAN_PROMPT = """
You are solving a math problem. Produce a short, high-signal plan before attempting the full solution.

Problem: {problem}

In the "plan" field, provide 3-6 concise steps describing how to solve the problem.
Do not compute the final answer here unless it is needed for the plan.
"""

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your reasoning in detail.
2. In the "answer" field, provide only the final answer concisely.

Problem: {input}
"""

SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

REVIEW_PROMPT = """
Given a math problem and a proposed solution, review the solution carefully.

problem: {problem}
solution: {solution}

If you find a concrete mathematical, logical, or formatting issue that is likely to make the final answer wrong, set "review_result" to false and explain the issue in "feedback".
Otherwise set "review_result" to true and briefly explain why the solution appears correct.
"""

REVISE_PROMPT = """
Given a math problem, an existing solution, and reviewer feedback, revise the solution so it addresses the identified issue.

problem: {problem}
solution: {solution}
feedback: {feedback}

In the "solution" field, return only the revised solution text.
"""

VERIFY_PROMPT = """
Given a math problem and a candidate solution, verify whether the final answer is supported by the reasoning.

problem: {problem}
solution: {solution}

In the "verdict" field, return true if the answer is supported and false otherwise.
In the "reasoning" field, briefly explain the verification result.
In the "corrected_answer" field, provide the validated final answer if clear; otherwise provide the best corrected answer you can infer.
"""

EXTRACT_ANSWER_PROMPT = """
Extract the final short answer from the following math solution.
Return only the concise final answer in the "answer" field, with no explanation.

problem: {problem}
solution: {solution}
"""

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""
