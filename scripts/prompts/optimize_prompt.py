WORKFLOW_OPTIMIZE_PROMPT = """You are editing a Python workflow file and its corresponding prompt file to solve {type} problems.
Referring to the current workflow and prompt, produce one small targeted optimization. You may add, modify, or delete
operator calls, control flow, or prompt text, but you must return a runnable Python workflow implementation.

The outer reply format still uses XML tags: <modification>...</modification>, <graph>...</graph>, <prompt>...</prompt>.
However, the CONTENT inside <graph> must be only pure valid Python code for graph.py. It is not a schema, not XML, not
HTML, not pseudocode, and not a node list.

Hard requirements for <graph>:
- Start with `class Workflow:`
- Include a complete `__init__`
- Include `async def __call__(self, problem: str):`
- Return the final answer and total cost
- Use only operator names that appear verbatim in the provided operator description
- Do not create any new operator class names, helper operator names, or aliases such as `MathSolver`, `Reasoner`, or `Verifier` unless that exact name already appears in the provided operator description
- Do not reference, instantiate, or call any attribute from `template/operator.py` unless that exact attribute name appears in the provided operator description
- Do not output `<Workflow>`, `<node>`, `interface`, bullet lists, or explanatory prose inside <graph>
- Do not wrap the code in markdown fences such as ```python

Hard requirements for <prompt>:
- Output only Python assignments for prompt_custom variables such as `XXX_PROMPT = \"...\"`
- Generate only prompts that are actually used by Custom operators in the graph
- Do not include placeholders

Use small patches. Prefer preserving working parts of the workflow and editing only the relevant scope. The workflow
complexity should not exceed 10 operator calls. Ensure that all prompts required by the current graph from prompt_custom
are included, and exclude any unused prompts.

Minimal graph shape example:
class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
        return response["response"], self.llm.get_usage_summary()["total_cost"]
"""


WORKFLOW_INPUT = """
Here is the current workflow context and the corresponding prompt summary from a previous iteration. You must make a
further optimization based on this workflow. The modified graph must differ from the provided example, and the specific
differences should be noted within the <modification>xxx</modification> section.

<sample>
    <experience>{experience}</experience>
    <modification>(such as:add /delete /modify/ ...)</modification>
    <score>{score}</score>
    <workflow_summary>{graph}</workflow_summary>
    <prompt_summary>{prompt}</prompt_summary>(only prompt_custom)
    <operator_description>{operator_description}</operator_description>
</sample>
Treat <operator_description> as a strict whitelist. You may only use operator names that already appear there.
If a name does not appear in <operator_description>, it is forbidden in <graph>.
Below are compact failure summaries from the aforementioned workflow, which can be used as references for optimization:
{log}

Before writing the final answer, reason about one small targeted patch only. Avoid broad rewrites.
When introducing new functionality in graph.py, import only libraries that are truly needed; operator, prompt_custom,
create_llm_instance, and DatasetType are already available in the file wrapper.
Under no circumstances should <graph> contain XML tags, schema syntax, markdown fences, or descriptive text.
Under no circumstances should graph.py output None for any required field.
Do not describe a workflow; directly write the Python code for graph.py.
Never invent a new operator, class, method, or attribute name to satisfy the optimization goal. Reuse only the listed operators.
"""

WORKFLOW_CUSTOM_USE = """\nHere's an example of using the `custom` method inside Python graph.py:
```
# You can write your own prompt in <prompt>prompt_custom</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
# You can also concatenate previously generated string results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=prompt_custom.XXX_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, xxx:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated(instruction+input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.\n
Important: the final <graph> content must look like normal Python source code for graph.py, not like XML, not like a
workflow schema, and not like a node inventory.

Introducing multiple operators at appropriate points can enhance performance. If you find that some provided operators
are not yet used in the graph, try incorporating them only when they clearly help the failure mode.
"""

WORKFLOW_TEMPLATE = """from typing import Literal
from ..template import operator
from . import prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

{graph}
"""
