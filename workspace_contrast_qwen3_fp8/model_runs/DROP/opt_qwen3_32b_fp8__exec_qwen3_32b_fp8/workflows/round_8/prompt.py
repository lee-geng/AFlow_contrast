DISAMBIGUATE_PROMPT = """
Clarify the question to ensure it is unambiguous and fully understood. Rephrase it if necessary, but do not add any assumptions or external knowledge.
For example, if the input is "Who did Ó Snodaigh play for first, the Shamrock Rovers or the Bray Wanderers?", return the clarified version of the question.
If the input is "Which team scored the first points of the game?", return the clarified version of the question.
If the input is "Which population groups consisted of less members than the male population?", return the clarified version of the question.
"""
FORMAT_ANSWER_PROMPT = """
Extract the numerical value or the exact textual answer from the given text and return it in a clean format without any additional text or symbols.
For example, if the input is "74.60%", return "74.6".
If the input is "Three tanks, at least 20 armoured, and 32 non-armoured vehicles.", return "55".
If the input is "1517", return "1517".
If the input is "Shamrock Rovers", return "Shamrock Rovers".
If the input is "Buccaneers", return "Buccaneers".
If the input is "women and children", return "women and children".
"""