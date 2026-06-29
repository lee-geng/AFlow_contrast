import json
import re
from typing import Any, Dict


class FixerExecutor:
    def __init__(self, operators: Dict[str, Any] = None, llm: Any = None):
        self.operators = operators or {}
        self.llm = llm

    async def execute(self, patch_spec: Dict[str, Any], operator_input: Any, original_output: Any, trace_state: Dict[str, Any]) -> Any:
        fixer = patch_spec.get("fixer", {})
        fixer_type = fixer.get("type")
        if fixer_type == "contract_repair":
            return self._contract_repair(fixer, original_output)
        if fixer_type == "answer_normalize":
            return self._answer_normalize(fixer, original_output)
        if fixer_type == "prompt_patch":
            return self._prompt_patch(fixer, original_output)
        if fixer_type == "existing_operator":
            return await self._existing_operator(fixer, operator_input, original_output)
        if fixer_type == "mini_workflow":
            return await self._mini_workflow(fixer, operator_input, original_output, trace_state)
        if fixer_type == "tool_call":
            return self._tool_call(fixer, operator_input, original_output)
        if fixer_type == "llm_repair":
            return await self._llm_repair(fixer, operator_input, original_output, trace_state)
        return original_output

    def _contract_repair(self, fixer: Dict[str, Any], original_output: Any) -> Any:
        definition = fixer.get("definition", "")
        transforms = fixer.get("transforms") or []
        if transforms:
            return self._apply_text_transforms(fixer, original_output)
        if original_output is None or original_output == "" or original_output == {} or original_output == []:
            if "fallback_output" in fixer:
                return fixer["fallback_output"]
        if isinstance(original_output, dict):
            repaired = dict(original_output)
            for field, default in fixer.get("output_mapping", {}).items():
                repaired.setdefault(field, default)
            for field in fixer.get("writes") or []:
                if field not in repaired and "raw_response" in repaired:
                    repaired[field] = str(repaired.get("raw_response") or "").strip()
            return repaired
        if "json" in definition.lower():
            text = str(original_output).strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.replace("json", "", 1).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"response": text}
        if original_output in {"", None}:
            return fixer.get("fallback_output", original_output)
        return original_output

    def _answer_normalize(self, fixer: Dict[str, Any], original_output: Any) -> Any:
        transforms = fixer.get("transforms") or [
            "strip_code_fence",
            "strip_boxed",
            "strip_final_answer_prefix",
            "strip_answer_prefix",
            "trim_terminal_punctuation",
        ]
        normalized = dict(fixer)
        normalized["fields"] = fixer.get("writes") or fixer.get("fields") or ["answer"]
        normalized["transforms"] = transforms
        if isinstance(original_output, dict):
            repaired = dict(original_output)
            for field in normalized["fields"]:
                if field in repaired and isinstance(repaired[field], str):
                    repaired[field] = self._repair_answer_text(repaired[field], transforms)
                elif field not in repaired and isinstance(repaired.get("raw_response"), str):
                    repaired[field] = self._repair_answer_text(repaired["raw_response"], transforms)
            return repaired
        return self._apply_text_transforms(normalized, original_output)

    def _apply_text_transforms(self, fixer: Dict[str, Any], original_output: Any) -> Any:
        transforms = fixer.get("transforms") or []
        fields = fixer.get("fields") or ["answer", "response", "solution", "output"]
        if isinstance(original_output, dict):
            repaired = dict(original_output)
            for field in fields:
                if field in repaired and isinstance(repaired[field], str):
                    repaired[field] = self._repair_answer_text(repaired[field], transforms)
            return repaired
        if isinstance(original_output, str):
            return self._repair_answer_text(original_output, transforms)
        return original_output

    def _repair_answer_text(self, text: str, transforms) -> str:
        repaired = text.strip()
        if "strip_code_fence" in transforms:
            repaired = self._strip_code_fence(repaired)
        if "strip_boxed" in transforms:
            boxed = re.search(r"\\boxed\s*\{([^{}]+)\}", repaired)
            if boxed:
                repaired = boxed.group(1).strip()
        if "strip_answer_prefix" in transforms:
            repaired = re.sub(
                r"^\s*(?:the\s+)?(?:final\s+)?answer\s+(?:is|:)\s*",
                "",
                repaired,
                flags=re.IGNORECASE,
            ).strip()
        if "strip_final_answer_prefix" in transforms:
            repaired = re.sub(
                r"^.*?(?:final\s+answer|answer)\s*(?:is|:)\s*",
                "",
                repaired,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
        if "number_words_to_digits" in transforms:
            repaired = self._number_words_to_digits(repaired)
        if "strip_percent" in transforms:
            repaired = repaired.rstrip("%").strip()
        if "trim_numeric_trailing_zeros" in transforms:
            repaired = self._trim_numeric_trailing_zeros(repaired)
        if "strip_leading_zero_decimal" in transforms:
            repaired = re.sub(r"^(-?)0\.(\d+)$", r"\1.\2", repaired)
        if "million_to_number" in transforms:
            repaired = self._million_to_number(repaired)
        if "trim_terminal_punctuation" in transforms:
            repaired = repaired.strip().strip(" \t\r\n.,:;!?")
        return repaired.strip()

    def _strip_code_fence(self, text: str) -> str:
        match = re.search(r"```(?:\w+)?\s*([\s\S]*?)```", text)
        return match.group(1).strip() if match else text

    def _number_words_to_digits(self, text: str) -> str:
        number_words = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
        }

        def replace(match):
            return number_words[match.group(0).lower()]

        pattern = r"\b(" + "|".join(number_words.keys()) + r")\b"
        return re.sub(pattern, replace, text, flags=re.IGNORECASE)

    def _trim_numeric_trailing_zeros(self, text: str) -> str:
        match = re.fullmatch(r"(-?\d+)\.(\d+)", text.strip())
        if not match:
            return text
        integer, fraction = match.groups()
        fraction = fraction.rstrip("0")
        return integer if not fraction else f"{integer}.{fraction}"

    def _million_to_number(self, text: str) -> str:
        match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s+million", text.strip(), flags=re.IGNORECASE)
        if not match:
            return text
        value = float(match.group(1)) * 1_000_000
        return str(int(value)) if value.is_integer() else str(value)

    def _prompt_patch(self, fixer: Dict[str, Any], original_output: Any) -> Any:
        if isinstance(original_output, dict):
            repaired = dict(original_output)
            repaired.setdefault("patch_note", fixer.get("definition", "prompt_patch"))
            return repaired
        return original_output

    async def _existing_operator(self, fixer: Dict[str, Any], operator_input: Any, original_output: Any) -> Any:
        operator = self.operators.get(fixer.get("name"))
        if operator is None:
            return original_output
        mapped_input = fixer.get("input_mapping", {}).get("input", operator_input)
        result = operator(mapped_input)
        if hasattr(result, "__await__"):
            return await result
        return result

    async def _mini_workflow(self, fixer: Dict[str, Any], operator_input: Any, original_output: Any, trace_state: Dict[str, Any]) -> Any:
        output = original_output
        for step in fixer.get("definition", []):
            if isinstance(step, dict) and step.get("type") == "contract_repair":
                output = self._contract_repair(step, output)
        return output

    def _tool_call(self, fixer: Dict[str, Any], operator_input: Any, original_output: Any) -> Any:
        return fixer.get("fallback_output", original_output)

    async def _llm_repair(
        self,
        fixer: Dict[str, Any],
        operator_input: Any,
        original_output: Any,
        trace_state: Dict[str, Any],
    ) -> Any:
        if self.llm is None:
            return original_output

        fields = fixer.get("writes") or fixer.get("fields") or ["answer", "response"]
        output_field = fields[0] if fields else "answer"
        prompt = self._llm_repair_prompt(
            instruction=fixer.get(
                "definition",
                "Repair the observable output into the shortest scorer-compatible final answer.",
            ),
            operator_input=operator_input,
            original_output=original_output,
            trace_state=trace_state,
            output_field=output_field,
            max_input_chars=int(fixer.get("max_input_chars", 2200)),
            max_output_chars=int(fixer.get("max_output_chars", 1200)),
        )
        try:
            raw = await self.llm(prompt)
            parsed = self._parse_llm_json(raw)
        except Exception:
            return original_output

        if isinstance(original_output, dict):
            repaired = dict(original_output)
            value, resolved_field = self._extract_repaired_field(parsed, fields, output_field)
            if value is not None:
                repaired[resolved_field] = self._normalize_llm_field_value(value)
                return repaired
            if isinstance(parsed, dict):
                if value is not None:
                    repaired[output_field] = self._normalize_llm_field_value(value)
                for key, value in parsed.items():
                    if key in fields and value is not None:
                        repaired[key] = self._normalize_llm_field_value(value)
                return repaired
            if isinstance(parsed, str):
                repaired[output_field] = self._normalize_llm_field_value(parsed)
                return repaired
            return original_output

        value, _ = self._extract_repaired_field(parsed, fields, output_field)
        if value is not None:
            return self._normalize_llm_field_value(value)
        if isinstance(parsed, str):
            return self._normalize_llm_field_value(parsed)
        return original_output

    def _llm_repair_prompt(
        self,
        instruction: str,
        operator_input: Any,
        original_output: Any,
        trace_state: Dict[str, Any],
        output_field: str,
        max_input_chars: int,
        max_output_chars: int,
    ) -> str:
        observable_state = {
            "output_fields": trace_state.get("output_fields"),
            "answer_text": trace_state.get("answer_text"),
            "thought": trace_state.get("thought"),
            "raw_response": trace_state.get("raw_response"),
            "parse_status": trace_state.get("parse_status"),
            "error_message": trace_state.get("error_message"),
        }
        payload = {
            "instruction": instruction,
            "existing_operator_output": self._preview(original_output, max_output_chars),
            "observable_state": self._preview(observable_state, max_output_chars),
            "required_output_field": output_field,
            "operator_input_preview": self._preview(operator_input, max_input_chars),
        }
        payload_text = json.dumps(payload, ensure_ascii=False)
        payload_text = payload_text[: max_input_chars + max_output_chars]
        return (
            "You are a guarded local workflow-output fixer. Repair only by extracting or normalizing information "
            "already present in existing_operator_output, observable_state.thought, or observable_state.raw_response. "
            "You may use operator_input only to identify the requested answer type; do not solve the question from "
            "scratch or introduce facts that are not supported by the existing output. "
            "Do not use gold labels because none are provided. Return strict JSON only, with no markdown. "
            f"The JSON must contain the field {json.dumps(output_field)} as a concise final answer string. "
            "If the field cannot be inferred from the existing output/thought/raw_response, return an empty string "
            f"for {json.dumps(output_field)} so the contract can reject the patch.\n"
            f"{payload_text}"
        )

    def _extract_repaired_field(self, parsed: Any, fields, default_field: str):
        if isinstance(parsed, dict):
            for field in [default_field, *fields]:
                if parsed.get(field) is not None:
                    return parsed.get(field), field
            for container_key in ("repaired_output", "output", "fields", "patch"):
                nested = parsed.get(container_key)
                if isinstance(nested, dict):
                    value, field = self._extract_repaired_field(nested, fields, default_field)
                    if value is not None:
                        return value, field
            if parsed.get("value") is not None:
                return parsed.get("value"), default_field
        if isinstance(parsed, str):
            return parsed, default_field
        return None, default_field

    def _normalize_llm_field_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False).strip()
        text = str(value).strip()
        text = re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer\s+(?:is|:)\s*", "", text, flags=re.IGNORECASE).strip()
        text = text.strip(" \t\r\n:;!?")
        if not re.fullmatch(r"-?(?:\d+(?:\.\d+)?|\.\d+)", text):
            text = text.rstrip(".,")
        return text

    def _parse_llm_json(self, raw: Any) -> Any:
        if isinstance(raw, (dict, list)):
            return raw
        text = str(raw).strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group(0))
            return text.strip().strip('"')

    def _preview(self, value: Any, max_chars: int) -> str:
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False)
            except TypeError:
                text = str(value)
        return text if len(text) <= max_chars else text[: max(0, max_chars - 3)] + "..."
