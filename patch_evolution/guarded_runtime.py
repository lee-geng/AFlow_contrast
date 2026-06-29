import contextvars
from dataclasses import dataclass
from typing import Any, Dict, Optional

from patch_evolution.contracts import ContractChecker
from patch_evolution.fixer import FixerExecutor
from patch_evolution.patch_registry import PatchRegistry
from patch_evolution.trigger_detector import TriggerDetector


_CURRENT_PATCH_RUNTIME = contextvars.ContextVar("patch_evolution_runtime", default=None)


@dataclass
class PatchRuntimeConfig:
    enabled: bool = False
    registry_dir: str = "results/patch_evolution"
    llm: Any = None


class GuardedRuntime:
    def __init__(self, registry: PatchRegistry, llm: Any = None):
        self.registry = registry
        self.detector = TriggerDetector()
        self.contract_checker = ContractChecker()
        self.fixer = FixerExecutor(llm=llm)

    async def apply(self, operator_name: str, operator_input: Any, original_output: Any, metadata: Dict[str, Any] = None) -> Any:
        output = original_output
        patches = self.registry.query_active_by_target(operator_name)
        for patch in patches:
            self.registry.update_telemetry(patch["patch_id"], {"checked_count": 1})
            state = self._state(operator_input, output, metadata)
            should_trigger, risk, info = self.detector.should_trigger(state, patch)
            if not should_trigger:
                continue
            self.registry.update_telemetry(patch["patch_id"], {"activation_count": 1})
            fixed_output = await self.fixer.execute(patch, operator_input, output, state)
            contract_result = self.contract_checker.check(fixed_output, patch.get("contract", {}))
            if not contract_result.ok:
                self.registry.update_telemetry(patch["patch_id"], {"rollback_count": 1})
                continue
            self.registry.update_telemetry(patch["patch_id"], {"accepted_count": 1})
            output = fixed_output
        return output

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
            "verifier_score": metadata.get("verifier_score"),
            "confidence": metadata.get("confidence"),
        }

    def _first_text_field(self, output: Any, fields) -> Optional[str]:
        if not isinstance(output, dict):
            return output if isinstance(output, str) else None
        for field in fields:
            value = output.get(field)
            if isinstance(value, str):
                return value
        return None


def set_patch_runtime(config: PatchRuntimeConfig):
    if not config or not config.enabled:
        return None
    runtime = GuardedRuntime(PatchRegistry(config.registry_dir), llm=config.llm)
    return _CURRENT_PATCH_RUNTIME.set(runtime)


def reset_patch_runtime(token) -> None:
    if token is not None:
        _CURRENT_PATCH_RUNTIME.reset(token)


def get_patch_runtime() -> Optional[GuardedRuntime]:
    return _CURRENT_PATCH_RUNTIME.get()


async def guarded_apply_operator_output(
    operator_name: str,
    operator_input: Any,
    original_output: Any,
    metadata: Dict[str, Any] = None,
) -> Any:
    runtime = get_patch_runtime()
    if runtime is None:
        return original_output
    return await runtime.apply(operator_name, operator_input, original_output, metadata=metadata)
