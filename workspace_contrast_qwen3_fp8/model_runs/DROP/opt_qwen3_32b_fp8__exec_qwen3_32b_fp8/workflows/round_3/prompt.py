FORMAT_ANSWER_PROMPT = """
Extract the numerical value from the given text and return it in a clean format without any additional text or symbols.
For example, if the input is "74.60%", return "74.6".
If the input is "Three tanks, at least 20 armoured, and 32 non-armoured vehicles.", return "55".
If the input is "1517", return "1517".
"""