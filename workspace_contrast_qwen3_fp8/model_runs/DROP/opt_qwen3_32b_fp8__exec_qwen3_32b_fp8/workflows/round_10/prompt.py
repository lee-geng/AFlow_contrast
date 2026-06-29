EXTRACT_ENTITIES_PROMPT = """
Identify and list the key entities or specific terms in the given text that are central to answering the question. These may include names of people, places, organizations, events, or specific values.
For example, if the input is "Which team scored the first points in the game between the Buccaneers and the Titans?", return "Buccaneers, Titans, first points".
If the input is "What was the shortest field goal in yards during the game between the Chargers and the Panthers?", return "Chargers, Panthers, shortest field goal, yards".
If the input is "By how many percentage points did the number of Nashville residents who worked at home differ from those who were without a car?", return "Nashville, residents, worked at home, without a car, percentage points".
"""