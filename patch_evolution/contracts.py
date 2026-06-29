import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ContractResult:
    ok: bool
    errors: List[str]


class ContractChecker:
    def check(self, output: Any, contract: Dict[str, Any]) -> ContractResult:
        errors = []
        output_requirements = contract.get("output_requirements", [])
        for requirement in output_requirements:
            error = self._check_requirement(output, requirement)
            if error:
                errors.append(error)
        return ContractResult(ok=not errors, errors=errors)

    def _check_requirement(self, output: Any, requirement: Any):
        if isinstance(requirement, str):
            if requirement == "non_empty" and self._is_empty(output):
                return "output is empty"
            if requirement == "json_valid":
                if isinstance(output, (dict, list)):
                    return None
                try:
                    json.loads(str(output))
                except json.JSONDecodeError:
                    return "output is not valid JSON"
            if requirement.startswith("field:"):
                field = requirement.split(":", 1)[1]
                if not isinstance(output, dict) or field not in output:
                    return f"missing required field: {field}"
                if self._is_empty(output.get(field)):
                    return f"empty required field: {field}"
            if requirement.startswith("type:"):
                expected = requirement.split(":", 1)[1]
                if expected == "dict" and not isinstance(output, dict):
                    return "output is not a dict"
                if expected == "str" and not isinstance(output, str):
                    return "output is not a string"
        if isinstance(requirement, dict):
            field = requirement.get("field")
            if field and (not isinstance(output, dict) or field not in output):
                return f"missing required field: {field}"
            if field and self._is_empty(output.get(field)):
                return f"empty required field: {field}"
        return None

    def _is_empty(self, output: Any) -> bool:
        return output is None or output == "" or output == {} or output == []
