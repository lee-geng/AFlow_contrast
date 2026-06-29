REVIEW_REVISE_PROMPT = """
Given the answer and the original question, review and revise the answer to ensure it is accurate, complete, and directly addresses the question. If the answer is numerical, ensure it is correctly formatted and matches the required precision. If the answer is textual, ensure it is concise and unambiguous.
For example:
- If the question asks "How many points lower was the fertility rate in 1999 compared to now?" and the answer is "0.15", return "0.15".
- If the question asks "Which racial group was the second smallest?" and the answer is "0", return "Native American".
- If the question asks "Who was the leader of the Byzantines?" and the answer is "0", return "Byzantine Emperor John V Palaiologos".
"""