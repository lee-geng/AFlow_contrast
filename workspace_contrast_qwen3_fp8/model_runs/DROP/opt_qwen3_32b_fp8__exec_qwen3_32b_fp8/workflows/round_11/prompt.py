REVIEW_ANSWER_PROMPT = """
Review the given answer and ensure it aligns with the question. If the answer is numerical, verify it is correct and in the right format. If the answer is textual, ensure it is concise and directly addresses the question. If the answer is incorrect or irrelevant, correct it.
For example, if the input is "1517" and the question is about a percentage, return "1517" if it is correct, or correct it if it is not.
If the input is "74.60%", return "74.6".
If the input is "Three tanks, at least 20 armoured, and 32 non-armoured vehicles.", return "55".
If the input is "1517", return "1517".
"""
FORMAT_ANSWER_PROMPT = """
Extract the numerical value from the given text and return it in a clean format without any additional text or symbols.
For example, if the input is "74.60%", return "74.6".
If the input is "Three tanks, at least 20 armoured, and 32 non-armoured vehicles.", return "55".
If the input is "1517", return "1517".
"""