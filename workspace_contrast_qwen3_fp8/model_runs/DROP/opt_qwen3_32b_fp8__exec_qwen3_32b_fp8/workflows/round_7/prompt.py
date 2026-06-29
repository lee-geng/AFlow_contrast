CONTEXTUALIZE_AND_CLARIFY_PROMPT = """
Clarify and enrich the given question by adding relevant context and resolving any ambiguities. Ensure the question is fully understood and ready for solution generation.
For example, if the input is "Which team scored the first points of the game?", return a clarified version such as "Which team scored the first points in the game between the Buccaneers and the Titans?".
If the input is "How many yards was the shortest field goal of the game?", return a clarified version such as "What was the shortest field goal in yards during the game between the Chargers and the Panthers?".
If the input is "How many less Nashville residents worked at home than were without a car?", return a clarified version such as "By how many percentage points did the number of Nashville residents who worked at home differ from those who were without a car?".
"""
FORMAT_ANSWER_PROMPT = """
Extract the numerical value from the given text and return it in a clean format without any additional text or symbols.
For example, if the input is "74.60%", return "74.6".
If the input is "Three tanks, at least 20 armoured, and 32 non-armoured vehicles.", return "55".
If the input is "1517", return "1517".
"""