DISAMBIGUATE_AND_CLARIFY_PROMPT = """
Clarify the question and extract key entities to ensure the model understands the intent and scope of the question. 
For example, if the input is "What races had between 100 and 500 births?", return a clarified version like "Identify the races with birth counts between 100 and 500 as described in the passage."
If the input is "Which happened first, Gustavus Adolphus succeeding to the Swedish throne, or his pressing his brother's claim to the Russian throne?", return a clarified version like "Determine the chronological order of two events: Gustavus Adolphus becoming king of Sweden and his support for his brother's claim to the Russian throne."
If the input is "What did the population want from the Bolsheviks?", return a clarified version like "Identify the main demands of the population from the Bolsheviks as described in the passage."
"""