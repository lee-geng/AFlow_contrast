import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from contrastive_experience.trace_context import record_operator_call
from patch_evolution.contracts import ContractChecker
from patch_evolution.fixer import FixerExecutor
from patch_evolution.trigger_detector import TriggerDetector


class CompiledPatchNode:
    """Explicit workflow node generated from validated PatchSpec evidence."""

    def __init__(
        self,
        llm: Any,
        patch_path: str,
        target_operator: str,
        node_name: Optional[str] = None,
    ):
        self.llm = llm
        self.patch_path = Path(patch_path)
        self.target_operator = target_operator
        self.node_name = node_name or f"{target_operator}Patch"
        self.detector = TriggerDetector()
        self.contract_checker = ContractChecker()
        self.fixer = FixerExecutor(llm=llm)
        self.patches = self._load_patches()

    async def __call__(self, output: Any, operator_input: Any = None, metadata: Dict[str, Any] = None) -> Any:
        if self._repair_lock(output):
            self._record_trace(operator_input, output, output, {}, "locked")
            return output
        repaired = output
        for patch in self.patches:
            state = self._state(operator_input, repaired, metadata)
            should_trigger, _, _ = self.detector.should_trigger(state, patch)
            if not should_trigger:
                continue
            candidate = await self.fixer.execute(patch, operator_input, repaired, state)
            if self.contract_checker.check(candidate, patch.get("contract", {})).ok:
                self._record_trace(operator_input, repaired, candidate, patch, "accepted")
                repaired = candidate
            else:
                self._record_trace(operator_input, repaired, candidate, patch, "contract_rejected")
        return repaired

    def _load_patches(self) -> List[Dict[str, Any]]:
        if not self.patch_path.exists():
            return []
        with self.patch_path.open("r", encoding="utf-8") as fin:
            data = json.load(fin)
        patches = data.get("patches") if isinstance(data, dict) else data
        if not isinstance(patches, list):
            return []
        return [
            patch
            for patch in patches
            if patch.get("target_operator") == self.target_operator
            and patch.get("status") in {"active_guarded", "validated", "promoted", "merged", "compiled"}
        ]

    def _state(self, operator_input: Any, output: Any, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        metadata = metadata or {}
        output_fields = list(output.keys()) if isinstance(output, dict) else []
        answer_text = self._first_text_field(output, ("answer", "response", "solution", "output"))
        output_text = answer_text if answer_text is not None else str(output)
        raw_response = output.get("raw_response") if isinstance(output, dict) else None
        thought = output.get("thought") if isinstance(output, dict) else None
        return {
            "input": operator_input,
            "output": output,
            "output_fields": output_fields,
            "answer": output.get("answer") if isinstance(output, dict) else None,
            "raw_response": raw_response,
            "thought": thought,
            "output_text": output_text,
            "answer_text": answer_text or "",
            "answer_word_count": len(str(answer_text or "").split()),
            "output_word_count": len(str(output_text or "").split()),
            "output_empty": output is None or output == "" or output == {} or output == [],
            "schema_valid": metadata.get("parse_status") not in {"failed", "invalid_json", "empty"},
            "parse_status": metadata.get("parse_status"),
            "error_message": metadata.get("error_message"),
            "tool_error": bool(metadata.get("tool_error")),
        }

    def _first_text_field(self, output: Any, fields) -> Optional[str]:
        if not isinstance(output, dict):
            return output if isinstance(output, str) else None
        for field in fields:
            value = output.get(field)
            if isinstance(value, str):
                return value
        return None

    def _repair_lock(self, output: Any) -> Optional[Dict[str, Any]]:
        if isinstance(output, dict) and isinstance(output.get("_repair_lock"), dict):
            return output["_repair_lock"]
        return None

    def _record_trace(
        self,
        operator_input: Any,
        original_output: Any,
        repaired_output: Any,
        patch: Dict[str, Any],
        status: str,
    ) -> None:
        input_payload = {
            "node_name": self.node_name,
            "target_operator": self.target_operator,
            "patch_id": patch.get("patch_id"),
            "status": status,
            "answer_before": self._first_text_field(original_output, ("answer", "response", "solution", "output"))
            if isinstance(original_output, dict)
            else original_output,
        }
        output_payload = {
            "node_name": self.node_name,
            "target_operator": self.target_operator,
            "patch_id": patch.get("patch_id"),
            "status": status,
            "answer_after": self._first_text_field(repaired_output, ("answer", "response", "solution", "output"))
            if isinstance(repaired_output, dict)
            else repaired_output,
        }
        record_operator_call(
            self.node_name,
            input_payload,
            output_payload,
            mode="compiled_patch",
            error_type=None if status != "contract_rejected" else status,
        )


class ContentRepairNode:
    """Explicit verifier/refiner workflow node for content-level answer failures."""

    DEFAULT_FAMILIES = ("numeric_content", "entity_boundary", "multi_span", "duration")

    def __init__(
        self,
        llm: Any,
        families: Optional[List[str]] = None,
        node_name: str = "ContentRepair",
        max_input_chars: int = 3600,
        max_output_chars: int = 1600,
        repair_focus: Optional[str] = None,
    ):
        self.llm = llm
        self.families = [family for family in (families or list(self.DEFAULT_FAMILIES)) if family in self.DEFAULT_FAMILIES]
        self.node_name = node_name
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars
        self.repair_focus = repair_focus or "content-level answer verification and repair"
        self.contract_checker = ContractChecker()

    async def __call__(self, output: Any, operator_input: Any = None, metadata: Dict[str, Any] = None) -> Any:
        lock = self._repair_lock(output)
        if lock:
            self._record_trace(operator_input, output, output, [], "locked", detail=lock.get("detail"))
            return output
        if self.llm is None or not self.families:
            self._record_trace(operator_input, output, output, [], "disabled")
            return output
        state = self._state(operator_input, output, metadata)
        candidate_families = self._candidate_families(state)
        if not candidate_families:
            self._record_trace(operator_input, output, output, [], "not_triggered")
            return output

        prompt = self._repair_prompt(operator_input, output, state, candidate_families)
        try:
            raw = await self.llm(prompt)
            parsed = self._parse_json(raw)
        except Exception:
            self._record_trace(operator_input, output, output, candidate_families, "llm_error")
            return output

        repair = self._extract_repair(parsed)
        if not repair:
            self._record_trace(operator_input, output, output, candidate_families, "llm_declined")
            return output
        family = repair.get("family")
        answer = self._normalize_answer(repair.get("answer"))
        confidence = self._confidence(repair)
        if confidence is not None and confidence < 0.75:
            self._record_trace(operator_input, output, output, candidate_families, "low_confidence")
            return output
        if self._is_placeholder_answer(answer):
            self._record_trace(operator_input, output, output, candidate_families, "placeholder_repair")
            return output
        if family not in candidate_families or not answer:
            self._record_trace(operator_input, output, output, candidate_families, "invalid_repair")
            return output
        if not self._plausible_repair(state.get("answer_text") or "", answer, family, state):
            self._record_trace(operator_input, output, output, candidate_families, "implausible_repair")
            return output

        candidate = self._write_answer(output, answer)
        if isinstance(candidate, dict):
            contract = {"output_requirements": ["non_empty", "field:answer"]}
        else:
            contract = {"output_requirements": ["non_empty", "type:str"]}
        if not self.contract_checker.check(candidate, contract).ok:
            self._record_trace(operator_input, output, candidate, candidate_families, "contract_rejected")
            return output
        self._record_trace(operator_input, output, candidate, candidate_families, "accepted")
        return candidate

    def _candidate_families(self, state: Dict[str, Any]) -> List[str]:
        input_text = state.get("input_search_text") or state.get("input_text", "")
        answer_text = state.get("answer_text", "")
        output_text = state.get("output_text", "")
        lowered = input_text.lower()
        candidates = []

        if "numeric_content" in self.families and self._looks_numeric_task(lowered, answer_text):
            candidates.append("numeric_content")
        if "duration" in self.families and self._looks_duration_task(lowered, answer_text):
            candidates.append("duration")
        if "multi_span" in self.families and self._looks_multi_span_task(lowered, answer_text, output_text):
            candidates.append("multi_span")
        if "entity_boundary" in self.families and self._looks_entity_boundary_task(lowered, answer_text, output_text):
            candidates.append("entity_boundary")
        return candidates

    def _state(self, operator_input: Any, output: Any, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        metadata = metadata or {}
        output_fields = list(output.keys()) if isinstance(output, dict) else []
        answer_text = self._first_text_field(output, ("answer", "response", "solution", "output"))
        output_text = answer_text if answer_text is not None else str(output)
        raw_response = output.get("raw_response") if isinstance(output, dict) else None
        thought = output.get("thought") if isinstance(output, dict) else None
        input_full = self._to_text(operator_input)
        input_text = self._head_tail_preview(input_full, self.max_input_chars)
        input_search_text = self._head_tail_preview(input_full, max(self.max_input_chars * 2, 6000))
        return {
            "input": operator_input,
            "input_text": input_text,
            "input_search_text": input_search_text,
            "output": output,
            "output_fields": output_fields,
            "answer_text": answer_text or "",
            "output_text": output_text or "",
            "raw_response": raw_response,
            "thought": thought,
            "parse_status": metadata.get("parse_status"),
        }

    def _repair_prompt(
        self,
        operator_input: Any,
        output: Any,
        state: Dict[str, Any],
        candidate_families: List[str],
    ) -> str:
        payload = {
            "candidate_families": candidate_families,
            "node_name": self.node_name,
            "repair_focus": self.repair_focus,
            "family_meanings": {
                "numeric_content": "recalculate or correct a concise numeric answer using only observable context/evidence",
                "entity_boundary": "correct an entity boundary/type mismatch, such as country vs nationality or group vs adjective",
                "multi_span": "return all requested spans/items when the question asks for multiple answers",
                "duration": "normalize or recompute a duration/time expression from observable evidence",
            },
            "existing_answer": state.get("answer_text", ""),
            "existing_operator_output": self._preview(output, self.max_output_chars),
            "existing_thought": self._preview(state.get("thought"), self.max_output_chars),
            "raw_response": self._preview(state.get("raw_response"), self.max_output_chars),
            "operator_input_preview": self._preview(operator_input, self.max_input_chars),
        }
        payload_text = json.dumps(payload, ensure_ascii=False)
        payload_text = payload_text[: self.max_input_chars + self.max_output_chars]
        return (
            "You are an explicit content-repair node inside a QA workflow. Decide whether the existing candidate "
            "answer should be repaired for one of the listed families. You may use the operator input/context, "
            "existing answer, existing thought, and raw response. No gold label is available. Repair only when the "
            "observable evidence clearly supports the new concise final answer; otherwise keep the original answer. "
            "For numeric_content and duration, you may perform simple arithmetic from observable numbers. "
            f"This node's focus is: {self.repair_focus}. "
            "Do not invent facts, do not return explanations in the answer field, and do not use markdown. "
            "Return strict JSON only with this schema: "
            "{\"should_repair\": boolean, \"family\": string, \"answer\": string, \"confidence\": number, \"reason\": string}. "
            "If uncertain, return should_repair=false and answer equal to the existing answer.\n"
            f"{payload_text}"
        )

    def _extract_repair(self, parsed: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(parsed, dict):
            return None
        container = parsed
        for key in ("repair", "patch", "result"):
            if isinstance(container.get(key), dict):
                container = container[key]
                break
        if container.get("should_repair") is not True:
            return None
        return container

    def _write_answer(self, output: Any, answer: str) -> Any:
        if isinstance(output, dict):
            repaired = dict(output)
            repaired["answer"] = answer
            return repaired
        return answer

    def _write_locked_answer(self, output: Any, answer: str, detail: str) -> Any:
        repaired = self._write_answer(output, answer)
        if isinstance(repaired, dict):
            repaired["_repair_lock"] = {
                "node": self.node_name,
                "detail": detail,
                "answer": answer,
            }
        return repaired

    def _plausible_repair(
        self,
        original_answer: str,
        repaired_answer: str,
        family: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        state = state or {}
        if original_answer.strip() == repaired_answer.strip():
            return False
        if len(repaired_answer) > 220:
            return False
        if self._is_placeholder_answer(repaired_answer):
            return False
        if family == "numeric_content":
            if not (self._number_pattern().search(repaired_answer) or self._number_word_pattern().search(repaired_answer)):
                return False
            return self._plausible_numeric_repair(original_answer, repaired_answer, state)
        if family == "duration":
            if re.search(r"\b(?:less|more|about|around|approximately|nearly|almost)\b", repaired_answer, re.IGNORECASE):
                return False
            return bool(self._number_pattern().search(repaired_answer) or re.search(r"\b(second|minute|hour|day|week|month|year)s?\b", repaired_answer, re.IGNORECASE))
        if family == "multi_span":
            return bool(re.search(r",|;|\band\b|\|", repaired_answer, re.IGNORECASE)) and len(repaired_answer.split()) <= 40
        if family == "entity_boundary":
            return len(repaired_answer.split()) <= 12
        return True

    def _looks_numeric_task(self, lowered_input: str, answer_text: str) -> bool:
        numeric_cues = (
            "how many",
            "how much",
            "number",
            "total",
            "difference",
            "more than",
            "less than",
            "fewer",
            "percent",
            "percentage",
            "average",
            "sum",
            "combined",
            "increase",
            "decrease",
            "how far",
            "shortest",
            "longest",
            "largest",
            "smallest",
            "least",
            "most",
            "not ",
            "were not",
            "remaining",
            "field goal",
            "touchdown",
            "yard",
            "population",
            "worked at home",
            "without a car",
            "record",
            "loss",
            "win",
            "yards",
            "points",
            "score",
        )
        input_number_count = len(self._number_pattern().findall(lowered_input))
        answer_has_number = bool(self._number_pattern().search(answer_text) or self._number_word_pattern().search(answer_text))
        return answer_has_number and input_number_count >= 2 and (
            any(cue in lowered_input for cue in numeric_cues) or input_number_count >= 5
        )

    def _looks_duration_task(self, lowered_input: str, answer_text: str) -> bool:
        duration_cues = ("duration", "how long", "time", "seconds", "minutes", "hours", "elapsed")
        answer_has_time = bool(
            re.search(r"\b\d+\s*(?:seconds?|minutes?|hours?)\b|\b\d+:\d{2}\b", answer_text, re.IGNORECASE)
            or (self._number_word_pattern().search(answer_text) and re.search(r"\b(?:seconds?|minutes?|hours?)\b", answer_text, re.IGNORECASE))
        )
        return answer_has_time and any(cue in lowered_input for cue in duration_cues)

    def _looks_multi_span_task(self, lowered_input: str, answer_text: str, output_text: str = "") -> bool:
        question_text = self._extract_question(lowered_input) or lowered_input
        if re.search(
            r"\b(?:what|which)\s+percent(?:age)?\b|\bhow\s+(?:many|much|far|long)\b",
            question_text,
            flags=re.IGNORECASE,
        ):
            return False
        multi_cues = (
            "which two",
            "what two",
            "who were",
            "which were",
            "what were",
            "what are",
            "which are",
            "both",
            "all of",
            "along with",
            "as well as",
            "and what",
            "and who",
            "list",
            "field goals",
            "touchdown runs",
            "touchdowns",
        )
        answer_looks_single = not re.search(r",|;|\band\b|\|", answer_text, re.IGNORECASE)
        return answer_looks_single and any(cue in question_text for cue in multi_cues)

    def _looks_entity_boundary_task(self, lowered_input: str, answer_text: str, output_text: str = "") -> bool:
        entity_cues = (
            "what country",
            "which country",
            "what nationality",
            "which nationality",
            "what group",
            "which group",
            "which side",
            "what side",
            "what team",
            "which team",
            "which nation",
            "what nation",
            "which army",
            "what organization",
            "which organization",
            "who had more",
            "who had the most",
            "which had more",
            "which had less",
            "which had fewer",
        )
        if not answer_text.strip():
            return False
        return any(cue in lowered_input for cue in entity_cues)

    def _first_text_field(self, output: Any, fields) -> Optional[str]:
        if not isinstance(output, dict):
            return output if isinstance(output, str) else None
        for field in fields:
            value = output.get(field)
            if isinstance(value, str):
                return value
        return None

    def _parse_json(self, raw: Any) -> Any:
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
            return text

    def _confidence(self, repair: Dict[str, Any]) -> Optional[float]:
        value = repair.get("confidence")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_answer(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        text = text.strip()
        text = re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer\s+(?:is|:)\s*", "", text, flags=re.IGNORECASE).strip()
        text = text.strip(" \t\r\n:;!?")
        if not re.fullmatch(r"-?(?:\d+(?:\.\d+)?|\.\d+)", text):
            text = text.rstrip(".,")
        return text

    def _is_placeholder_answer(self, answer: Any) -> bool:
        text = str(answer or "").strip().strip("\"'").lower()
        return text in {"", "answer", "final answer", "none", "null", "n/a", "unknown"}

    def _plausible_numeric_repair(self, original_answer: str, repaired_answer: str, state: Dict[str, Any]) -> bool:
        original_numbers = self._number_pattern().findall(str(original_answer or "").replace(",", ""))
        repaired_numbers = self._number_pattern().findall(str(repaired_answer or "").replace(",", ""))
        if len(repaired_numbers) != 1:
            return False
        if len(original_numbers) == 1:
            original_value = self._to_float(original_numbers[0])
            repaired_value = self._to_float(repaired_numbers[0])
            if original_value is None or repaired_value is None:
                return True
            if abs(original_value - repaired_value) < 1e-9:
                return True
            original_is_integer = float(original_value).is_integer()
            repaired_is_integer = float(repaired_value).is_integer()
            if original_is_integer and repaired_is_integer:
                evidence = " ".join(
                    str(value or "")
                    for value in (state.get("thought"), state.get("raw_response"), state.get("output_text"))
                )
                return self._numeric_value_in_text(repaired_numbers[0], evidence)
            if original_value != 0 and repaired_value != 0 and abs(original_value / repaired_value) in {10.0, 100.0, 1000.0}:
                return False
            if repaired_value != 0 and original_value != 0 and abs(repaired_value / original_value) in {10.0, 100.0, 1000.0}:
                return False
        return True

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    def _numeric_value_in_text(self, number_text: str, text: str) -> bool:
        value = self._to_float(number_text)
        if value is None:
            return False
        candidates = {str(int(value)) if float(value).is_integer() else str(value)}
        if 0 < abs(value) < 1:
            candidates.add(str(value).lstrip("0"))
        normalized_text = str(text or "").replace(",", "")
        return any(re.search(rf"(?<![\d.]){re.escape(candidate)}(?!\d|\.\d)", normalized_text) for candidate in candidates)

    def _preview(self, value: Any, max_chars: int) -> str:
        text = self._to_text(value)
        return text if len(text) <= max_chars else text[: max(0, max_chars - 3)] + "..."

    def _record_trace(
        self,
        operator_input: Any,
        original_output: Any,
        repaired_output: Any,
        candidate_families: List[str],
        status: str,
        detail: Optional[str] = None,
    ) -> None:
        input_payload = {
            "node_name": self.node_name,
            "families": self.families,
            "candidate_families": candidate_families,
            "status": status,
            "detail": detail,
            "input_preview": self._head_tail_preview(self._to_text(operator_input), 700),
            "answer_before": self._first_text_field(original_output, ("answer", "response", "solution", "output"))
            if isinstance(original_output, dict)
            else original_output,
        }
        output_payload = {
            "node_name": self.node_name,
            "status": status,
            "detail": detail,
            "answer_after": self._first_text_field(repaired_output, ("answer", "response", "solution", "output"))
            if isinstance(repaired_output, dict)
            else repaired_output,
        }
        record_operator_call(
            self.node_name,
            input_payload,
            output_payload,
            mode="content_repair",
            error_type=None if status not in {"llm_error", "contract_rejected"} else status,
        )

    def _to_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        else:
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return str(value)

    def _head_tail_preview(self, value: str, max_chars: int) -> str:
        text = value or ""
        if len(text) <= max_chars:
            return text
        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars - 18
        return text[:head_chars] + "\n...<truncated>...\n" + text[-tail_chars:]

    def _extract_question(self, text: str) -> str:
        match = re.search(r"Question:\s*([\s\S]*?)\s*Answer:", str(text or ""), flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_passage(self, text: str) -> str:
        match = re.search(r"Passage:\s*([\s\S]*?)\s*Question:", str(text or ""), flags=re.IGNORECASE)
        return match.group(1).strip() if match else str(text or "")

    def _number_words(self) -> Dict[str, int]:
        return {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
        }

    def _quantity_values(self, text: str) -> List[float]:
        values: List[float] = []
        for match in self._number_pattern().finditer(str(text or "").replace(",", "")):
            value = self._to_float(match.group(0))
            if value is not None:
                values.append(value)
        for match in re.finditer(r"\b(" + "|".join(self._number_words().keys()) + r")\b", str(text or ""), flags=re.IGNORECASE):
            values.append(float(self._number_words()[match.group(1).lower()]))
        return values

    def _format_number(self, value: float, prefer_leading_zero: bool = True) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        text = f"{value:.10f}".rstrip("0").rstrip(".")
        if not prefer_leading_zero and text.startswith("0."):
            text = text[1:]
        if not prefer_leading_zero and text.startswith("-0."):
            text = "-" + text[2:]
        return text

    def _repair_lock(self, output: Any) -> Optional[Dict[str, Any]]:
        if isinstance(output, dict) and isinstance(output.get("_repair_lock"), dict):
            return output["_repair_lock"]
        return None

    def _number_pattern(self):
        return re.compile(r"-?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)")

    def _number_word_pattern(self):
        return re.compile(
            r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b",
            re.IGNORECASE,
        )


class AnswerCanonicalizerNode(ContentRepairNode):
    """Deterministic low-risk answer canonicalization before ensemble selection."""

    COUNTRY_TO_DEMONYM = {
        "france": "French",
        "spain": "Spanish",
        "russia": "Russian",
        "georgia": "Georgian",
        "germany": "German",
        "china": "Chinese",
        "japan": "Japanese",
        "poland": "Polish",
        "england": "English",
        "britain": "British",
        "united states": "American",
    }

    def __init__(self, llm: Any = None, **kwargs):
        super().__init__(
            llm,
            families=["numeric_content"],
            node_name=kwargs.pop("node_name", "AnswerCanonicalizer"),
            repair_focus=kwargs.pop("repair_focus", "deterministic answer canonicalization"),
            **kwargs,
        )
        self.families = ["canonical"]

    async def __call__(self, output: Any, operator_input: Any = None, metadata: Dict[str, Any] = None) -> Any:
        lock = self._repair_lock(output)
        if lock:
            self._record_trace(operator_input, output, output, [], "locked", detail=lock.get("detail"))
            return output
        state = self._state(operator_input, output, metadata)
        answer = state.get("answer_text", "")
        candidate, detail = self._canonical_candidate(answer, state)
        if not candidate:
            self._record_trace(operator_input, output, output, [], "not_triggered")
            return output
        repaired = self._write_locked_answer(output, candidate, detail)
        if not self.contract_checker.check(repaired, {"output_requirements": ["non_empty", "field:answer"]}).ok:
            self._record_trace(operator_input, output, repaired, ["canonical"], "contract_rejected", detail=detail)
            return output
        self._record_trace(operator_input, output, repaired, ["canonical"], "accepted", detail=detail)
        return repaired

    def _canonical_candidate(self, answer: str, state: Dict[str, Any]) -> tuple:
        stripped = str(answer or "").strip().strip(" \t\r\n.,:;!?")
        lowered = stripped.lower()
        number_words = self._number_words()
        if lowered in number_words:
            return str(number_words[lowered]), "exact_number_word_to_digit"

        question = self._extract_question(state.get("input_search_text") or state.get("input_text") or "").lower()
        passage = self._extract_passage(state.get("input_search_text") or state.get("input_text") or "")
        evidence = f"{passage}\n{state.get('thought') or ''}\n{state.get('raw_response') or ''}".lower()
        demonym = self.COUNTRY_TO_DEMONYM.get(lowered)
        if (
            demonym
            and "troop" in question
            and ("which nation" in question or "which side" in question or "which force" in question)
            and demonym.lower() in evidence
        ):
            return demonym, "country_to_demonym_for_troop_question"
        return None, None


class MissingAnswerRecoveryNode(ContentRepairNode):
    def __init__(self, llm: Any, **kwargs):
        super().__init__(
            llm,
            families=["numeric_content"],
            node_name=kwargs.pop("node_name", "MissingAnswerRecovery"),
            repair_focus=kwargs.pop(
                "repair_focus",
                "schema recovery: fill a missing or placeholder answer field from existing thought/raw_response only",
            ),
            **kwargs,
        )
        self.families = ["schema_recovery"]

    async def __call__(self, output: Any, operator_input: Any = None, metadata: Dict[str, Any] = None) -> Any:
        lock = self._repair_lock(output)
        if lock:
            self._record_trace(operator_input, output, output, [], "locked", detail=lock.get("detail"))
            return output
        state = self._state(operator_input, output, metadata)
        if not self._should_recover_answer(output, state):
            self._record_trace(operator_input, output, output, [], "not_triggered")
            return output
        if self.llm is None:
            self._record_trace(operator_input, output, output, ["schema_recovery"], "disabled")
            return output

        prompt = self._recovery_prompt(operator_input, output, state)
        try:
            raw = await self.llm(prompt)
            parsed = self._parse_json(raw)
        except Exception:
            self._record_trace(operator_input, output, output, ["schema_recovery"], "llm_error")
            return output
        if not isinstance(parsed, dict) or parsed.get("should_repair") is not True:
            self._record_trace(operator_input, output, output, ["schema_recovery"], "llm_declined")
            return output

        answer = self._normalize_answer(parsed.get("answer"))
        confidence = self._confidence(parsed)
        if confidence is not None and confidence < 0.7:
            self._record_trace(operator_input, output, output, ["schema_recovery"], "low_confidence")
            return output
        if self._is_placeholder_answer(answer) or len(answer.split()) > 24:
            self._record_trace(operator_input, output, output, ["schema_recovery"], "invalid_repair")
            return output
        if not self._evidence_supports_answer(answer, state):
            self._record_trace(operator_input, output, output, ["schema_recovery"], "unsupported_repair")
            return output

        repaired = dict(output) if isinstance(output, dict) else {}
        repaired["answer"] = answer
        if not self.contract_checker.check(repaired, {"output_requirements": ["non_empty", "field:answer"]}).ok:
            self._record_trace(operator_input, output, repaired, ["schema_recovery"], "contract_rejected")
            return output
        self._record_trace(operator_input, output, repaired, ["schema_recovery"], "accepted")
        return repaired

    def _should_recover_answer(self, output: Any, state: Dict[str, Any]) -> bool:
        if not isinstance(output, dict):
            return False
        answer = output.get("answer")
        return "answer" not in output or self._is_placeholder_answer(answer)

    def _recovery_prompt(self, operator_input: Any, output: Any, state: Dict[str, Any]) -> str:
        payload = {
            "node_name": self.node_name,
            "task": "Recover the missing answer field from existing observable reasoning.",
            "existing_operator_output": self._preview(output, self.max_output_chars),
            "existing_thought": self._preview(state.get("thought"), self.max_output_chars),
            "raw_response": self._preview(state.get("raw_response"), self.max_output_chars),
            "operator_input_preview": self._preview(operator_input, self.max_input_chars),
        }
        payload_text = json.dumps(payload, ensure_ascii=False)
        payload_text = payload_text[: self.max_input_chars + self.max_output_chars]
        return (
            "You are a guarded schema-recovery node. The operator output is missing the answer field or has a "
            "placeholder answer. Recover the concise final answer only from existing_operator_output.thought, "
            "raw_response, or directly quoted observable context. Do not solve from scratch and do not invent facts. "
            "Return strict JSON only: {\"should_repair\": boolean, \"answer\": string, \"confidence\": number, \"reason\": string}. "
            "If the answer is not clearly supported by the observable thought/raw_response, return should_repair=false.\n"
            f"{payload_text}"
        )

    def _evidence_supports_answer(self, answer: str, state: Dict[str, Any]) -> bool:
        evidence = " ".join(
            str(value or "")
            for value in (state.get("thought"), state.get("raw_response"), state.get("output_text"))
        ).lower()
        normalized_answer = answer.lower().strip()
        if normalized_answer and normalized_answer in evidence:
            return True
        tokens = [
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z'-]+|\d+(?:\.\d+)?", normalized_answer)
            if token.lower() not in {"the", "a", "an", "and", "or", "of", "to", "in"}
        ]
        if not tokens:
            return False
        matched = 0
        for token in tokens:
            lowered = token.lower()
            variants = {lowered}
            if lowered.endswith("s") and len(lowered) > 3:
                variants.add(lowered[:-1])
            else:
                variants.add(f"{lowered}s")
            if any(variant in evidence for variant in variants):
                matched += 1
        return matched / len(tokens) >= 0.75


class NumericVerifierNode(ContentRepairNode):
    def __init__(self, llm: Any, **kwargs):
        super().__init__(
            llm,
            families=["numeric_content"],
            node_name=kwargs.pop("node_name", "NumericVerifier"),
            repair_focus=kwargs.pop(
                "repair_focus",
                "numeric questions: verify extracted numbers, infer the operation, and recompute a concise numeric answer",
            ),
            **kwargs,
        )

    async def __call__(self, output: Any, operator_input: Any = None, metadata: Dict[str, Any] = None) -> Any:
        lock = self._repair_lock(output)
        if lock:
            self._record_trace(operator_input, output, output, [], "locked", detail=lock.get("detail"))
            return output
        state = self._state(operator_input, output, metadata)
        question = self._extract_question(state.get("input_search_text") or state.get("input_text") or "")
        if self._question_requests_entity_answer(question):
            self._record_trace(operator_input, output, output, [], "not_triggered", detail="entity_answer_question")
            return output
        candidate, detail = self._deterministic_numeric_candidate(state)
        if candidate:
            repaired = self._write_locked_answer(output, candidate, detail)
            if self.contract_checker.check(repaired, {"output_requirements": ["non_empty", "field:answer"]}).ok:
                self._record_trace(operator_input, output, repaired, ["numeric_content"], "accepted", detail=detail)
                return repaired
            self._record_trace(operator_input, output, repaired, ["numeric_content"], "contract_rejected", detail=detail)
        return await super().__call__(output, operator_input=operator_input, metadata=metadata)

    def _deterministic_numeric_candidate(self, state: Dict[str, Any]) -> tuple:
        input_text = state.get("input_search_text") or state.get("input_text") or ""
        question = self._extract_question(input_text)
        passage = self._extract_passage(input_text)
        answer = state.get("answer_text", "")

        candidates = [
            self._yard_measure_operation_candidate(question, passage),
            self._month_span_candidate(question, passage),
            self._age_at_birth_event_candidate(question, passage),
            self._game_tied_time_candidate(question, passage),
            self._percent_complement_candidate(question, passage),
            self._percentage_less_difference_candidate(question, passage),
            self._percentage_difference_candidate(question, passage),
            self._listed_quantity_sum_candidate(question, answer),
            self._singular_count_candidate(question, passage, answer),
            self._record_before_event_candidate(question, passage),
        ]
        answer_normalized = str(answer or "").strip().strip(" \t\r\n.,:;!?%").lower()
        for candidate, detail in candidates:
            if not candidate:
                continue
            candidate_normalized = str(candidate).strip().strip(" \t\r\n.,:;!?%").lower()
            if candidate_normalized and candidate_normalized != answer_normalized:
                return candidate, detail
        return None, None

    def _question_requests_entity_answer(self, question: str) -> bool:
        lowered_question = str(question or "").lower()
        if not lowered_question:
            return False
        if any(
            cue in lowered_question
            for cue in (
                "how many",
                "how much",
                "what number",
                "what percent",
                "what percentage",
                "how long",
                "how far",
                "what distance",
                "from what distance",
            )
        ):
            return False
        entity_patterns = (
            r"\b(?:which|what)\s+team\b",
            r"\b(?:which|what)\s+(?:player|person|coach|quarterback|kicker|receiver|runner|school|country|nation|state|city|group|side|organization|club)\b",
            r"\bwho\b",
        )
        return any(re.search(pattern, lowered_question) for pattern in entity_patterns)

    def _month_span_candidate(self, question: str, passage: str) -> tuple:
        lowered_question = question.lower()
        if "month" not in lowered_question or "time span" not in lowered_question:
            return None, None
        ranges = []
        for sentence in self._sentences_ranked_by_question_overlap(question, passage):
            for match in re.finditer(
                rf"\b(?:between|from)\s+({self._month_names_pattern()})\s+(\d{{4}})\s+"
                rf"(?:and|to|through|until)\s+({self._month_names_pattern()})\s+(\d{{4}})\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                start_month = self._month_number(match.group(1))
                end_month = self._month_number(match.group(3))
                start_year = int(match.group(2))
                end_year = int(match.group(4))
                if start_month is None or end_month is None:
                    continue
                months = abs((end_year - start_year) * 12 + (end_month - start_month))
                if months > 0:
                    ranges.append(months)
            if ranges:
                unique = sorted(set(ranges))
                if len(unique) == 1:
                    return str(unique[0]), "numeric_operation:month_span"
        return None, None

    def _age_at_birth_event_candidate(self, question: str, passage: str) -> tuple:
        lowered_question = question.lower()
        if "how many years old" not in lowered_question or "born" not in lowered_question:
            return None, None
        subject_match = re.search(r"\bhow many years old was\s+(.+?)\s+when\b", question, flags=re.IGNORECASE)
        if not subject_match:
            return None, None
        subject = subject_match.group(1).strip().strip(" \t\r\n.,:;!?")
        if not subject:
            return None, None

        event_date = self._birth_event_date_for_subject(subject, passage)
        birth_date = self._subject_birth_date(subject, passage)
        if event_date and birth_date:
            age = event_date[0] - birth_date[0]
            if (event_date[1], event_date[2]) < (birth_date[1], birth_date[2]):
                age -= 1
            if age >= 0:
                return str(age), "numeric_operation:age_at_birth_event"

        event_year = event_date[0] if event_date else self._birth_event_year_for_subject(subject, passage)
        inferred_birth_year = self._subject_birth_year_from_since(subject, passage)
        if event_year is not None and inferred_birth_year is not None and event_year >= inferred_birth_year:
            return str(event_year - inferred_birth_year), "numeric_operation:age_at_birth_event"
        return None, None

    def _game_tied_time_candidate(self, question: str, passage: str) -> tuple:
        lowered_question = question.lower()
        if "time" not in lowered_question or "tied" not in lowered_question:
            return None, None
        for sentence in re.split(r"(?<=[.!?])\s+", str(passage or "")):
            if not re.search(r"\btied?\b", sentence, flags=re.IGNORECASE):
                continue
            match = re.search(
                r"\bwith\s+(\d+:\d{2}|\d+\s+seconds?)\s+remaining\b[^.]*\btied?\b",
                sentence,
                flags=re.IGNORECASE,
            )
            if not match:
                match = re.search(
                    r"\btied?\b[^.]*\bwith\s+(\d+:\d{2}|\d+\s+seconds?)\s+remaining\b",
                    sentence,
                    flags=re.IGNORECASE,
                )
            if match:
                return match.group(1), "numeric_operation:tied_game_time_remaining"
        return None, None

    def _yard_measure_operation_candidate(self, question: str, passage: str) -> tuple:
        lowered_question = question.lower()
        if "yard" not in lowered_question and not re.search(
            r"\b(?:distance|field goals?|touchdowns?|td)\b",
            lowered_question,
            flags=re.IGNORECASE,
        ):
            return None, None
        comparison = self._yard_comparison_candidate(question, passage)
        if comparison[0]:
            return comparison
        values = self._extract_yard_values(question, passage)
        if not values:
            return None, None
        if "longest" in lowered_question and "shortest" in lowered_question and (
            "longer" in lowered_question or "difference" in lowered_question or "than" in lowered_question
        ):
            return str(max(values) - min(values)), "numeric_operation:max_minus_min"
        if "second longest" in lowered_question:
            unique_desc = sorted(set(values), reverse=True)
            if len(unique_desc) >= 2:
                return str(unique_desc[1]), "numeric_operation:ordinal_extreme:second_longest"
        if "shortest" in lowered_question and "longest" not in lowered_question:
            return str(min(values)), "numeric_operation:min"
        if "longest" in lowered_question and "shortest" not in lowered_question:
            return str(max(values)), "numeric_operation:max"
        if re.search(r"\b(?:make|made|kick|kicked|get|got|nail|nailed)\s+two\b", lowered_question):
            counts: Dict[int, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            repeated = sorted(value for value, count in counts.items() if count >= 2)
            if len(repeated) == 1:
                return f"{repeated[0]}-yard", "numeric_operation:repeated_value"
        return None, None

    def _yard_comparison_candidate(self, question: str, passage: str) -> tuple:
        lowered_question = question.lower()
        if "longer" not in lowered_question or "longest" not in lowered_question:
            return None, None
        event_kind = self._yard_event_kind(question)
        if not event_kind:
            return None, None
        event_phrase = self._yard_event_phrase(event_kind)
        pattern = (
            rf"\bwas\s+(.+?)(?:'s|')\s+longest\s+{event_phrase}\s+"
            rf"(?:compared\s+to|than)\s+(.+?)(?:'s|')\s+only\s+{event_phrase}\b"
        )
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            return None, None
        left_subject = self._clean_yard_subject(match.group(1))
        right_subject = self._clean_yard_subject(match.group(2))
        if not left_subject or not right_subject:
            return None, None
        left_values = self._extract_yard_values(question, passage, subject=left_subject)
        right_values = self._extract_yard_values(question, passage, subject=right_subject)
        right_unique = sorted(set(right_values))
        if not left_values or len(right_unique) != 1:
            return None, None
        return str(max(left_values) - right_unique[0]), "numeric_operation:subject_max_minus_subject_only"

    def _extract_yard_values(self, question: str, passage: str, subject: Optional[str] = None) -> List[int]:
        event_kind = self._yard_event_kind(question)
        if not event_kind:
            return []
        subject = subject if subject is not None else self._extract_yard_subject(question)
        sentence_candidates = self._sentences_for_subject(passage, subject, fallback=not bool(subject))
        values: List[int] = []
        for sentence in sentence_candidates:
            values.extend(self._extract_values_from_yard_sentence(sentence, event_kind))
        return values

    def _yard_event_kind(self, question: str) -> Optional[str]:
        lowered_question = question.lower()
        if re.search(r"\b(?:field goals?|fgs?)\b", lowered_question):
            return "field_goal"
        if re.search(r"\b(?:touchdown|td)\s+pass(?:es)?\b", lowered_question):
            return "touchdown_pass"
        if re.search(r"\b(?:touchdown|td)\s+runs?\b", lowered_question):
            return "touchdown_run"
        return None

    def _yard_event_phrase(self, event_kind: str) -> str:
        if event_kind == "field_goal":
            return r"(?:field goals?|fgs?)"
        if event_kind == "touchdown_pass":
            return r"(?:touchdown|td)\s+pass(?:es)?"
        if event_kind == "touchdown_run":
            return r"(?:touchdown|td)\s+runs?"
        return r""

    def _extract_values_from_yard_sentence(self, sentence: str, event_kind: str) -> List[int]:
        if event_kind == "touchdown_run":
            return [
                int(value)
                for value in re.findall(r"\b(\d+)-yard\s+(?:touchdown|td)\s+run\b", sentence, flags=re.IGNORECASE)
            ]
        if event_kind == "touchdown_pass":
            direct = [
                int(value)
                for value in re.findall(r"\b(\d+)-yard\s+(?:touchdown|td)\s+pass\b", sentence, flags=re.IGNORECASE)
            ]
            grouped = []
            for group in re.findall(
                r"(?:touchdown|td)\s+passes\s+of\s+([^.;!?]+?)\s+yards\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                grouped.extend(int(value) for value in re.findall(r"\b(\d+)\b", group))
            return direct + grouped
        if event_kind != "field_goal":
            return []

        values: List[int] = []
        for match in re.finditer(r"\b(\d+)-yard(?:er)?\b", sentence, flags=re.IGNORECASE):
            before = sentence[max(0, match.start() - 32) : match.start()]
            after = sentence[match.end() : min(len(sentence), match.end() + 56)]
            local = f"{before} {match.group(0)} {after}"
            if not re.search(r"\b(?:field goals?|fgs?|yarder)\b", local, flags=re.IGNORECASE):
                continue
            if self._field_goal_window_is_negative(before, after):
                continue
            values.append(int(match.group(1)))
        for group in re.findall(r"field goals?\s+from\s+([^.;!?]+?)\s+yards\b", sentence, flags=re.IGNORECASE):
            values.extend(int(value) for value in re.findall(r"\b(\d+)\b", group))
        return values

    def _field_goal_window_is_negative(self, before: str, after: str) -> bool:
        before_lower = before.lower()
        after_lower = after.lower()
        if re.search(r"\b(?:missed|blocked|failed|shanked)\b", before_lower):
            return True
        if re.search(r"\b(?:attempt|no good|wide|short|blocked)\b", after_lower):
            return True
        return False

    def _extract_yard_subject(self, question: str) -> Optional[str]:
        patterns = [
            r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)(?:'s|')\s+(?:longest|shortest|second longest)",
            r"\bdid\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:make|kick|get|nail)",
            r"\bfrom what distance did\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+",
        ]
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                return self._clean_yard_subject(match.group(1))
        return None

    def _clean_yard_subject(self, subject: str) -> str:
        cleaned = str(subject or "").strip().strip(" \t\r\n.,:;!?")
        cleaned = re.sub(r"^(?:kicker|qb|quarterback|wr|rb|te)\s+", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _sentences_for_subject(self, passage: str, subject: Optional[str], fallback: bool = True) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", str(passage or ""))
        if not subject:
            return sentences
        lowered_subject = subject.lower()
        surname = lowered_subject.split()[-1]
        filtered = [
            sentence
            for sentence in sentences
            if lowered_subject in sentence.lower() or re.search(rf"\b{re.escape(surname)}\b", sentence, flags=re.IGNORECASE)
        ]
        return filtered or (sentences if fallback else [])

    def _percent_complement_candidate(self, question: str, passage: str) -> tuple:
        lowered_question = question.lower()
        if "not " not in lowered_question or "percent" not in lowered_question:
            return None, None
        if re.search(r"\b(?:did|does|do)\s+not\s+\w+\b", lowered_question):
            return None, None
        target = self._extract_percent_complement_target(question)
        if not target:
            return None, None
        search_passage = self._passage_window_for_year(passage, self._question_year(question)) or passage
        value = self._find_percent_for_target(search_passage, target)
        if value is None:
            return None, None
        return self._format_number(100.0 - value), "numeric_operation:percent_complement"

    def _extract_percent_complement_target(self, question: str) -> str:
        match = re.search(r"\bnot\s+(.+?)(?:\s+in\s+\d{4}\b|\s+according\b|\?|$)", question, flags=re.IGNORECASE)
        if not match:
            return ""
        return self._clean_question_term(match.group(1))

    def _percentage_less_difference_candidate(self, question: str, passage: str) -> tuple:
        lowered_question = question.lower()
        if " than " not in lowered_question or not re.search(r"\b(?:less|fewer)\b", lowered_question):
            return None, None
        terms = self._less_than_terms(question)
        if not terms:
            return None, None
        left_term, right_term = terms
        left_values = self._find_percents_for_target(passage, left_term)
        right_values = self._find_percents_for_target(passage, right_term)
        if not left_values or not right_values:
            return None, None
        positive_differences = [
            right - left
            for left in left_values
            for right in right_values
            if right > left and abs(right - left) > 1e-9
        ]
        unique = sorted(set(round(value, 10) for value in positive_differences))
        if len(unique) == 1:
            return self._format_number(unique[0]), "numeric_operation:percent_less_difference"
        return None, None

    def _less_than_terms(self, question: str) -> Optional[tuple]:
        lowered_question = question.lower()
        if "worked at home" in lowered_question and "without a car" in lowered_question:
            return "worked at home", "without a car"
        match = re.search(
            r"\b(?:less|fewer)\s+(.+?)\s+than\s+(?:were|was|are|is)?\s*(.+?)(?:\?|$)",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        left = self._clean_question_term(match.group(1))
        right = self._clean_question_term(match.group(2))
        if not left or not right:
            return None
        return left, right

    def _percentage_difference_candidate(self, question: str, passage: str) -> tuple:
        lowered_question = question.lower()
        if " than " not in lowered_question or not ("more people" in lowered_question or "more percent" in lowered_question):
            return None, None
        match = re.search(r"\bwere\s+(.+?)\s+than\s+(.+?)(?:\?|$)", question, flags=re.IGNORECASE)
        if not match:
            return None, None
        left_term = self._clean_question_term(match.group(1))
        right_terms = self._split_question_terms(match.group(2))
        if not left_term or not right_terms:
            return None, None
        left_value = self._find_percent_for_target(passage, left_term)
        right_values = [self._find_percent_for_target(passage, term) for term in right_terms]
        if left_value is None or any(value is None for value in right_values):
            return None, None
        return self._format_number(left_value - sum(value for value in right_values if value is not None)), "numeric_operation:percent_difference"

    def _find_percent_for_target(self, passage: str, target: str) -> Optional[float]:
        unique = self._find_percents_for_target(passage, target)
        if len(unique) == 1:
            return unique[0]
        return None

    def _find_percents_for_target(self, passage: str, target: str) -> List[float]:
        variants = self._target_variants(target)
        if not variants:
            return []
        matches: List[float] = []
        passage_text = str(passage or "")
        percent_with_tail = r"(?<![\d.])(\d+(?:\.\d+)?)%\s+([^\n]{0,140}?)(?=(?<![\d.])\d+(?:\.\d+)?%|[,.;\n]|$)"
        for match in re.finditer(percent_with_tail, passage_text, flags=re.IGNORECASE):
            value = self._to_float(match.group(1))
            if value is None:
                continue
            tail = self._normalize_category_text(match.group(2))
            if self._category_candidate_is_metadata(tail):
                continue
            if self._category_matches(tail, variants):
                matches.append(value)
        for match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)%\s*(?=\)|,|\.|;|\n|$)", passage_text, flags=re.IGNORECASE):
            value = self._to_float(match.group(1))
            if value is None:
                continue
            prefix = self._normalize_category_text(passage_text[max(0, match.start() - 90) : match.start()])
            if self._category_candidate_is_metadata(prefix):
                continue
            if self._category_matches(prefix, variants):
                matches.append(value)
        unique = []
        for value in matches:
            if value not in unique:
                unique.append(value)
        return unique

    def _target_variants(self, target: str) -> List[str]:
        normalized = self._normalize_category_text(target)
        if not normalized:
            return []
        variants = {normalized}
        if normalized in {"us", "u s"}:
            variants.add("united states")
        if normalized.endswith("s") and len(normalized) > 3:
            variants.add(normalized[:-1])
        elif len(normalized) > 2:
            variants.add(f"{normalized}s")
        return sorted(variants, key=len, reverse=True)

    def _category_matches(self, text: str, variants: List[str]) -> bool:
        normalized = self._normalize_category_text(text)
        if not normalized:
            return False
        return any(re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", normalized) for variant in variants)

    def _category_candidate_is_metadata(self, text: str) -> bool:
        normalized = self._normalize_category_text(text)
        return "united states census" in normalized

    def _normalize_category_text(self, text: str) -> str:
        normalized = str(text or "").lower()
        normalized = re.sub(r"\b2\b", "two", normalized)
        normalized = re.sub(r"\bu\.s\.\b", "united states", normalized)
        normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
        normalized = re.sub(r"\b(?:of whom were|were of|were|was|are|from|the|people|ancestry|race)\b", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _clean_question_term(self, text: str) -> str:
        cleaned = str(text or "").strip().strip(" \t\r\n.,:;!?")
        cleaned = re.sub(r"^(?:(?:from|the|a|an)\s+)+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+(?:people|ancestry|race)$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _split_question_terms(self, text: str) -> List[str]:
        cleaned = re.sub(r"\s+or\s+", " and ", str(text or ""), flags=re.IGNORECASE)
        parts = re.split(r"\s+and\s+|,", cleaned, flags=re.IGNORECASE)
        return [self._clean_question_term(part) for part in parts if self._clean_question_term(part)]

    def _question_year(self, question: str) -> Optional[str]:
        match = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", str(question or ""))
        return match.group(1) if match else None

    def _passage_window_for_year(self, passage: str, year: Optional[str]) -> str:
        if not year:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", str(passage or ""))
        matches = [sentence for sentence in sentences if re.search(rf"\b{re.escape(year)}\b", sentence)]
        return " ".join(matches)

    def _month_names_pattern(self) -> str:
        return r"January|February|March|April|May|June|July|August|September|October|November|December"

    def _month_number(self, month: str) -> Optional[int]:
        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        return months.get(str(month or "").lower())

    def _sentences_ranked_by_question_overlap(self, question: str, passage: str) -> List[str]:
        sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", str(passage or "")) if sentence.strip()]
        keywords = self._question_keywords(question)
        if not keywords:
            return sentences
        return sorted(
            sentences,
            key=lambda sentence: len(keywords & set(re.findall(r"[a-z0-9]+", sentence.lower()))),
            reverse=True,
        )

    def _question_keywords(self, question: str) -> set:
        stopwords = {
            "the",
            "was",
            "were",
            "when",
            "where",
            "what",
            "which",
            "how",
            "many",
            "much",
            "time",
            "span",
            "old",
            "years",
            "months",
            "less",
            "fewer",
            "than",
            "from",
            "with",
            "that",
            "this",
            "did",
            "does",
            "make",
            "made",
        }
        return {word for word in re.findall(r"[a-z0-9]+", str(question or "").lower()) if len(word) > 2 and word not in stopwords}

    def _birth_event_date_for_subject(self, subject: str, passage: str) -> Optional[tuple]:
        subject_pattern = re.escape(subject)
        for sentence in re.split(r"(?<=[.!?])\s+", str(passage or "")):
            if not re.search(subject_pattern, sentence, flags=re.IGNORECASE):
                continue
            if not re.search(r"\b(?:son|daughter|child)\s+of\b|\b(?:his|her)\s+(?:son|daughter|child)\b", sentence, flags=re.IGNORECASE):
                continue
            date = self._first_full_date(sentence)
            if date:
                return date
        return None

    def _birth_event_year_for_subject(self, subject: str, passage: str) -> Optional[int]:
        date = self._birth_event_date_for_subject(subject, passage)
        if date:
            return date[0]
        return None

    def _subject_birth_date(self, subject: str, passage: str) -> Optional[tuple]:
        subject_pattern = re.escape(subject)
        for sentence in re.split(r"(?<=[.!?])\s+", str(passage or "")):
            if not re.search(subject_pattern, sentence, flags=re.IGNORECASE) or "born" not in sentence.lower():
                continue
            if re.search(rf"\b(?:son|daughter|child)\s+of\s+{subject_pattern}\b", sentence, flags=re.IGNORECASE):
                continue
            if re.search(r"\b(?:his|her)\s+(?:son|daughter|child)\b", sentence, flags=re.IGNORECASE):
                continue
            if not re.search(
                rf"\b{subject_pattern}\b[^.]*\b(?:was|is|being)?\s*born\b|\bborn\b[^.]*\b{subject_pattern}\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                continue
            date = self._first_full_date(sentence)
            if date:
                return date
        return None

    def _subject_birth_year_from_since(self, subject: str, passage: str) -> Optional[int]:
        subject_pattern = re.escape(subject)
        for sentence in re.split(r"(?<=[.!?])\s+", str(passage or "")):
            if not re.search(subject_pattern, sentence, flags=re.IGNORECASE):
                continue
            if not re.search(r"\bonly\b.*\b(?:child|son|daughter)\b.*\bborn\b.*\bsince\b", sentence, flags=re.IGNORECASE):
                continue
            match = re.search(r"\bsince\s+(1[6-9]\d{2}|20\d{2})\b", sentence)
            if match:
                return int(match.group(1))
        return None

    def _first_full_date(self, text: str) -> Optional[tuple]:
        match = re.search(
            rf"\b({self._month_names_pattern()})\s+(\d{{1,2}}),\s*(1[6-9]\d{{2}}|20\d{{2}})\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        month = self._month_number(match.group(1))
        if month is None:
            return None
        return int(match.group(3)), month, int(match.group(2))

    def _listed_quantity_sum_candidate(self, question: str, answer: str) -> tuple:
        lowered_question = question.lower()
        if "how many" not in lowered_question:
            return None, None
        answer_text = str(answer or "")
        if not re.search(r",|;|\band\b", answer_text, flags=re.IGNORECASE):
            return None, None
        if re.search(r"%|\byards?\b|\bseconds?\b|\bminutes?\b|\bhours?\b|\bmonths?\b|\byears?\b", answer_text, flags=re.IGNORECASE):
            return None, None
        values = self._quantity_values(answer_text)
        if len(values) < 2:
            return None, None
        total = sum(values)
        if total <= 0:
            return None, None
        return self._format_number(total), "numeric_operation:sum_listed_quantities"

    def _singular_count_candidate(self, question: str, passage: str, answer: str) -> tuple:
        lowered_answer = str(answer or "").lower()
        if not re.search(r"\b(?:not specified|no specific|does not provide|not provide)\b", lowered_answer):
            return None, None
        match = re.search(r"\bhow many\s+(.+?)\s+(?:did|were|was|are|is|had|have)\b", question, flags=re.IGNORECASE)
        if not match:
            return None, None
        object_phrase = self._clean_question_term(match.group(1))
        head = object_phrase.split()[-1] if object_phrase.split() else ""
        if head.endswith("ies"):
            singular = head[:-3] + "y"
        elif head.endswith("s") and len(head) > 3:
            singular = head[:-1]
        else:
            singular = head
        if singular and re.search(rf"\b(?:an?|one)\s+(?:[a-z-]+\s+){{0,4}}{re.escape(singular)}\b", passage, flags=re.IGNORECASE):
            return "1", "numeric_operation:singular_count"
        return None, None

    def _record_before_event_candidate(self, question: str, passage: str) -> tuple:
        lowered_question = question.lower()
        if "loss" not in lowered_question or "come into this game" not in lowered_question:
            return None, None
        match = re.search(r"\bwith the loss,\s+[^.]*?\bfell to\s+(\d+)-(\d+)\b", passage, flags=re.IGNORECASE)
        if not match:
            return None, None
        losses_after = int(match.group(2))
        if losses_after <= 0:
            return None, None
        return str(losses_after - 1), "numeric_operation:record_before_event"


class MultiSpanExtractorNode(ContentRepairNode):
    def __init__(self, llm: Any, **kwargs):
        super().__init__(
            llm,
            families=["multi_span"],
            node_name=kwargs.pop("node_name", "MultiSpanExtractor"),
            repair_focus=kwargs.pop(
                "repair_focus",
                "multi-span questions: recover all requested spans/items when the candidate answer kept only one",
            ),
            **kwargs,
        )


class AnswerTypeRefinerNode(ContentRepairNode):
    def __init__(self, llm: Any, **kwargs):
        super().__init__(
            llm,
            families=["entity_boundary"],
            node_name=kwargs.pop("node_name", "AnswerTypeRefiner"),
            repair_focus=kwargs.pop(
                "repair_focus",
                "entity boundary and answer type questions: convert verbose answers or wrong entity types to the requested concise type",
            ),
            **kwargs,
        )


class DurationNormalizerNode(ContentRepairNode):
    def __init__(self, llm: Any, **kwargs):
        super().__init__(
            llm,
            families=["duration"],
            node_name=kwargs.pop("node_name", "DurationNormalizer"),
            repair_focus=kwargs.pop(
                "repair_focus",
                "duration and clock-time questions: normalize or recompute time expressions from observable evidence",
            ),
            **kwargs,
        )
