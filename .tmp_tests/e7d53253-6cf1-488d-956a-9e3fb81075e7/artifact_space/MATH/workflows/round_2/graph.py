from typing import Literal
from ..template import operator
from . import prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

class Workflow:
    async def __call__(self, problem: str):
        return "ok", self.llm.get_usage_summary()["total_cost"]
