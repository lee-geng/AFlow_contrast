import ast
import re
from dataclasses import dataclass, field
from typing import List, Literal


GuardMode = Literal["off", "repair_then_continue", "fail_fast"]


@dataclass
class WorkflowGuardResult:
    valid: bool
    graph: str
    repaired: bool
    issues: List[str] = field(default_factory=list)


class WorkflowGuard:
    """Lightweight guardrail for generated workflow code before evaluation."""

    def __init__(self, mode: GuardMode = "repair_then_continue"):
        self.mode = mode

    def ensure(self, graph: str) -> WorkflowGuardResult:
        if self.mode == "off":
            return WorkflowGuardResult(valid=True, graph=graph, repaired=False, issues=[])

        issues: List[str] = []
        repaired = False
        fixed = graph

        fixed, changed, s_issues = self._sanitize_graph_code(fixed)
        repaired = repaired or changed
        issues.extend(s_issues)

        fixed, changed, c_issues = self._repair_common_operator_mismatch(fixed)
        repaired = repaired or changed
        issues.extend(c_issues)

        fixed, changed, r_issues = self._repair_return_shape(fixed)
        repaired = repaired or changed
        issues.extend(r_issues)

        parse_ok, parse_err = self._is_parseable(fixed)
        if not parse_ok:
            issues.append(f"syntax_error_after_repair: {parse_err}")
            return WorkflowGuardResult(valid=False, graph=fixed, repaired=repaired, issues=issues)

        checks_ok, check_issues = self._check_required_interfaces(fixed)
        issues.extend(check_issues)
        return WorkflowGuardResult(valid=checks_ok, graph=fixed, repaired=repaired, issues=issues)

    def sanitize_graph_text(self, graph: str) -> str:
        fixed, _, _ = self._sanitize_graph_code(graph)
        return fixed

    def _sanitize_graph_code(self, graph: str):
        issues: List[str] = []
        if graph is None:
            return "", False, issues

        original = str(graph)
        fixed = original.strip()

        tag_match = re.search(r"<graph>(.*?)</graph>", fixed, flags=re.DOTALL | re.IGNORECASE)
        if tag_match:
            fixed = tag_match.group(1).strip()

        fixed = re.sub(r"^\s*```(?:python)?\s*", "", fixed, flags=re.IGNORECASE)
        fixed = re.sub(r"\s*```\s*$", "", fixed)
        fixed = fixed.replace("\r\n", "\n").replace("\r", "\n").strip()

        lines = fixed.splitlines()
        start_idx = self._find_code_start(lines)
        if start_idx > 0:
            fixed = "\n".join(lines[start_idx:]).strip()

        salvaged = self._salvage_parseable_python(fixed)
        if salvaged:
            fixed = salvaged

        changed = fixed != original.strip()
        if changed:
            issues.append("auto_repaired:sanitized_graph_text")
        return fixed, changed, issues

    @staticmethod
    def _find_code_start(lines: List[str]) -> int:
        starters = (
            "from ",
            "import ",
            "class Workflow",
            "class ",
            "@",
            "#",
        )
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(starters):
                return idx
        return 0

    def _salvage_parseable_python(self, graph: str) -> str:
        candidates = [graph.strip()]
        lines = graph.splitlines()

        workflow_idx = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("class Workflow"):
                workflow_idx = idx
                break
        if workflow_idx is not None:
            candidates.append("\n".join(lines[workflow_idx:]).strip())

        for candidate in candidates:
            if not candidate:
                continue
            if self._is_parseable(candidate)[0]:
                return candidate

            trimmed = candidate.splitlines()
            while trimmed:
                current = "\n".join(trimmed).strip()
                if current and self._is_parseable(current)[0]:
                    return current
                last = trimmed[-1].strip()
                if last and not self._looks_like_python_line(last):
                    trimmed.pop()
                    continue
                break
        return graph

    @staticmethod
    def _looks_like_python_line(line: str) -> bool:
        python_prefixes = (
            "from ",
            "import ",
            "class ",
            "def ",
            "async def ",
            "return ",
            "if ",
            "elif ",
            "else:",
            "for ",
            "while ",
            "try:",
            "except ",
            "finally:",
            "with ",
            "@",
            "#",
            "pass",
            "break",
            "continue",
            "raise ",
        )
        if line.startswith(python_prefixes):
            return True
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", line):
            return True
        if re.match(r"^[)\]}]+[,)]*$", line):
            return True
        if line.endswith((":", ")", "]", "}", ",", '"""', "'''")):
            return True
        if line.startswith((")", "]", "}")):
            return True
        return False

    def _repair_common_operator_mismatch(self, graph: str):
        issues: List[str] = []
        changed = False
        fixed = graph

        patterns = [
            (r"operator\.Programmer\(\s*\)", "operator.Programmer(self.llm)", "programmer_missing_llm"),
            (r"operator\.ScEnsemble\(\s*\)", "operator.ScEnsemble(self.llm)", "ensemble_missing_llm"),
            (
                r"operator\.ScEnsemble\(\s*solutions\s*=.*?\)",
                "operator.ScEnsemble(self.llm)",
                "ensemble_ctor_misused_with_runtime_kwargs",
            ),
            (
                r"operator\.ScEnsemble\(\s*self\.llm\s*,\s*[^)]*(?:solutions|problem)\s*=.*?\)",
                "operator.ScEnsemble(self.llm)",
                "ensemble_ctor_misused_with_mixed_kwargs",
            ),
            (r"self\.ensemble\.sc_ensemble\(", "self.ensemble(", "ensemble_call_wrong_method"),
        ]
        for pattern, repl, label in patterns:
            new_fixed, n = re.subn(pattern, repl, fixed, flags=re.DOTALL)
            if n > 0:
                fixed = new_fixed
                changed = True
                issues.append(f"auto_repaired:{label}")

        return fixed, changed, issues

    def _repair_return_shape(self, graph: str):
        changed = False
        issues: List[str] = []
        fixed = graph

        # Keep output + cost if a three-item return accidentally appears.
        pattern = (
            r"return\s+(.+?),\s*(.+?),\s*(self\.llm\.get_usage_summary\(\)\[\"total_cost\"\])"
        )
        repl = r"return \1, \3"
        new_fixed, n = re.subn(pattern, repl, fixed)
        if n > 0:
            fixed = new_fixed
            changed = True
            issues.append("auto_repaired:return_tuple_len_3_to_len_2")

        return fixed, changed, issues

    def _is_parseable(self, graph: str):
        try:
            ast.parse(graph)
            return True, ""
        except SyntaxError as exc:
            return False, str(exc)

    def _check_required_interfaces(self, graph: str):
        issues: List[str] = []
        try:
            tree = ast.parse(graph)
        except SyntaxError as exc:
            return False, [f"syntax_error:{exc}"]

        workflow_cls = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "Workflow":
                workflow_cls = node
                break
        if workflow_cls is None:
            return False, ["missing_Workflow_class"]

        call_fn = None
        for node in workflow_cls.body:
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "__call__":
                call_fn = node
                break
        if call_fn is None:
            return False, ["missing_async___call__"]

        has_valid_return = False
        for node in ast.walk(call_fn):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 2:
                has_valid_return = True
                break
        if not has_valid_return:
            issues.append("return_shape_must_be_(output,cost)")

        return len(issues) == 0, issues
