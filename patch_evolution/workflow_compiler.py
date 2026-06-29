import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from patch_evolution.consolidator import OfflineConsolidator
from patch_evolution.patch_registry import PatchRegistry


COMPILE_STATUSES = {"active_guarded", "validated", "promoted", "merged"}
SUPPORTED_COMPILE_TARGETS = {"AnswerGenerate", "ScEnsemble"}


@dataclass
class WorkflowCompilationResult:
    source_workflow_dir: str
    compiled_workflow_dir: str
    compiled_patch_count: int
    skipped_patch_count: int
    inserted_nodes: List[Dict[str, Any]]
    compiled_patch_ids: List[str]
    skipped_patches: List[Dict[str, Any]]
    edit_manifest_path: str
    compiled_patches_path: str
    content_repair_families: List[str]
    content_repair_batches: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_workflow_dir": self.source_workflow_dir,
            "compiled_workflow_dir": self.compiled_workflow_dir,
            "compiled_patch_count": self.compiled_patch_count,
            "skipped_patch_count": self.skipped_patch_count,
            "inserted_nodes": self.inserted_nodes,
            "compiled_patch_ids": self.compiled_patch_ids,
            "skipped_patches": self.skipped_patches,
            "edit_manifest_path": self.edit_manifest_path,
            "compiled_patches_path": self.compiled_patches_path,
            "content_repair_families": self.content_repair_families,
            "content_repair_batches": self.content_repair_batches,
        }


class WorkflowPatchCompiler:
    DEFAULT_CONTENT_REPAIR_FAMILIES = ["numeric_content", "entity_boundary", "multi_span", "duration"]
    CONTENT_REPAIR_BATCHES = {
        "schema": {
            "node_name": "missing_answer_recovery",
            "class_name": "MissingAnswerRecoveryNode",
            "families": ["schema_recovery"],
        },
        "canonical": {
            "node_name": "answer_canonicalizer",
            "class_name": "AnswerCanonicalizerNode",
            "families": ["canonical"],
        },
        "numeric": {
            "node_name": "numeric_verifier",
            "class_name": "NumericVerifierNode",
            "families": ["numeric_content"],
        },
        "multispan": {
            "node_name": "multi_span_extractor",
            "class_name": "MultiSpanExtractorNode",
            "families": ["multi_span"],
        },
        "entity": {
            "node_name": "answer_type_refiner",
            "class_name": "AnswerTypeRefinerNode",
            "families": ["entity_boundary"],
        },
        "duration": {
            "node_name": "duration_normalizer",
            "class_name": "DurationNormalizerNode",
            "families": ["duration"],
        },
    }
    CONTENT_REPAIR_BATCH_ALIASES = {
        "all": ["schema", "canonical", "numeric", "multispan", "entity", "duration"],
        "default": ["schema", "canonical", "numeric", "multispan", "entity", "duration"],
        "canon": ["canonical"],
        "canonicalizer": ["canonical"],
        "answer-canonicalizer": ["canonical"],
        "answer_canonicalizer": ["canonical"],
        "missing": ["schema"],
        "missing-answer": ["schema"],
        "missing_answer": ["schema"],
        "schema-recovery": ["schema"],
        "schema_recovery": ["schema"],
        "numeric_content": ["numeric"],
        "numeric_verifier": ["numeric"],
        "multi_span": ["multispan"],
        "multi-span": ["multispan"],
        "multi_span_extractor": ["multispan"],
        "multispan_extractor": ["multispan"],
        "entity_boundary": ["entity"],
        "answer_type": ["entity"],
        "answer_type_refiner": ["entity"],
        "time": ["duration"],
        "duration_normalizer": ["duration"],
    }

    def __init__(
        self,
        min_credit: float = 0.0,
        max_patches: int = 8,
        compile_content_nodes: bool = False,
        content_repair_families: Optional[List[str]] = None,
        content_repair_batches: Optional[List[str]] = None,
    ):
        self.min_credit = min_credit
        self.max_patches = max(1, max_patches)
        self.content_repair_batches = self._normalize_content_repair_batches(content_repair_batches)
        self.compile_content_nodes = compile_content_nodes or bool(self.content_repair_batches)
        self.content_repair_families = content_repair_families or list(self.DEFAULT_CONTENT_REPAIR_FAMILIES)
        self.consolidator = OfflineConsolidator()

    def compile_from_registry(
        self,
        workflow_dir: str,
        patch_registry_dir: str,
        output_dir: str,
        dataset: str,
        run_id: str,
        patch_ids: Optional[List[str]] = None,
    ) -> WorkflowCompilationResult:
        registry = PatchRegistry(patch_registry_dir)
        patches = registry.load_all()
        return self.compile(
            workflow_dir=workflow_dir,
            patches=patches,
            output_dir=output_dir,
            dataset=dataset,
            run_id=run_id,
            patch_ids=patch_ids,
        )

    def compile(
        self,
        workflow_dir: str,
        patches: List[Dict[str, Any]],
        output_dir: str,
        dataset: str,
        run_id: str,
        patch_ids: Optional[List[str]] = None,
    ) -> WorkflowCompilationResult:
        source_dir = Path(workflow_dir).resolve()
        if not (source_dir / "graph.py").exists():
            raise FileNotFoundError(f"Workflow graph not found: {source_dir / 'graph.py'}")

        selected, skipped = self._select_patches(patches, patch_ids=patch_ids)
        source_compiled_patches = self._load_source_compiled_patches(source_dir)
        compiled_dir = Path(output_dir).resolve() / dataset / run_id / f"{source_dir.name}_compiled"
        if compiled_dir.exists():
            shutil.rmtree(compiled_dir)
        shutil.copytree(source_dir, compiled_dir)

        compiled_patches = self._merge_compiled_patches(
            source_compiled_patches,
            [self._mark_compiled(patch) for patch in selected],
        )
        compiled_patches_path = compiled_dir / "compiled_patches.json"
        with compiled_patches_path.open("w", encoding="utf-8") as fout:
            json.dump({"patches": compiled_patches}, fout, ensure_ascii=False, indent=2)

        graph_path = compiled_dir / "graph.py"
        graph_text = graph_path.read_text(encoding="utf-8")
        graph_text, inserted_nodes, graph_skips = self._compile_graph(graph_text, compiled_patches)
        graph_path.write_text(graph_text, encoding="utf-8")
        skipped.extend(graph_skips)

        edit_manifest = {
            "edit_type": "patch_compiled_workflow",
            "source_workflow_dir": str(source_dir),
            "compiled_workflow_dir": str(compiled_dir),
            "dataset": dataset,
            "run_id": run_id,
            "inserted_nodes": inserted_nodes,
            "compiled_patch_ids": [patch["patch_id"] for patch in compiled_patches],
            "skipped_patches": skipped,
            "content_repair_families": self.content_repair_families if self.compile_content_nodes else [],
            "content_repair_batches": self.content_repair_batches,
            "notes": [
                "This workflow contains explicit patch nodes compiled from validated runtime patch evidence.",
                "When enabled, content repair nodes are explicit guarded workflow nodes rather than runtime registry hooks.",
                "It can be evaluated without enabling the guarded runtime patch registry.",
            ],
        }
        edit_manifest_path = compiled_dir / "workflow_edit_manifest.json"
        with edit_manifest_path.open("w", encoding="utf-8") as fout:
            json.dump(edit_manifest, fout, ensure_ascii=False, indent=2)

        return WorkflowCompilationResult(
            source_workflow_dir=str(source_dir),
            compiled_workflow_dir=str(compiled_dir),
            compiled_patch_count=len(compiled_patches),
            skipped_patch_count=len(skipped),
            inserted_nodes=inserted_nodes,
            compiled_patch_ids=[patch["patch_id"] for patch in compiled_patches],
            skipped_patches=skipped,
            edit_manifest_path=str(edit_manifest_path),
            compiled_patches_path=str(compiled_patches_path),
            content_repair_families=self.content_repair_families if self.compile_content_nodes else [],
            content_repair_batches=self.content_repair_batches,
        )

    def _normalize_content_repair_batches(self, batches: Optional[List[str]]) -> List[str]:
        if not batches:
            return []
        normalized = []
        for raw_batch in batches:
            batch = str(raw_batch).strip().lower().replace("_", "-")
            batch = batch.replace("multi-span", "multispan")
            if not batch:
                continue
            aliases = self.CONTENT_REPAIR_BATCH_ALIASES.get(batch) or self.CONTENT_REPAIR_BATCH_ALIASES.get(batch.replace("-", "_"))
            expanded = aliases or [batch]
            for item in expanded:
                canonical = item.replace("_", "-").replace("multi-span", "multispan")
                if canonical in self.CONTENT_REPAIR_BATCHES and canonical not in normalized:
                    normalized.append(canonical)
        return normalized

    def _select_patches(
        self,
        patches: List[Dict[str, Any]],
        patch_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        requested = set(patch_ids or [])
        selected = []
        skipped = []
        for patch in patches:
            patch_id = patch.get("patch_id")
            if requested and patch_id not in requested:
                continue
            if patch.get("status") not in COMPILE_STATUSES:
                skipped.append({"patch_id": patch_id, "reason": f"status:{patch.get('status')}"})
                continue
            target = patch.get("target_operator")
            if target not in SUPPORTED_COMPILE_TARGETS:
                skipped.append({"patch_id": patch_id, "reason": f"unsupported_target:{target}"})
                continue
            decision = self.consolidator.decide_patch(patch)
            if decision.get("decision") == "prune" or float(decision.get("credit") or 0.0) < self.min_credit:
                skipped.append(
                    {
                        "patch_id": patch_id,
                        "reason": f"low_credit:{decision.get('decision')}",
                        "credit": decision.get("credit", 0.0),
                    }
                )
                continue
            selected.append((float(decision.get("credit") or 0.0), patch))

        selected.sort(key=lambda item: item[0], reverse=True)
        limited = [patch for _, patch in selected[: self.max_patches]]
        for _, patch in selected[self.max_patches :]:
            skipped.append({"patch_id": patch.get("patch_id"), "reason": "max_patches_limit"})
        return limited, skipped

    def _mark_compiled(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        compiled = json.loads(json.dumps(patch, ensure_ascii=False))
        compiled["status"] = "compiled"
        compiled.setdefault("generation", {})
        compiled["generation"]["compiled_into_workflow"] = True
        return compiled

    def _load_source_compiled_patches(self, source_dir: Path) -> List[Dict[str, Any]]:
        patch_path = source_dir / "compiled_patches.json"
        if not patch_path.exists():
            return []
        try:
            with patch_path.open("r", encoding="utf-8") as fin:
                data = json.load(fin)
        except (OSError, json.JSONDecodeError):
            return []
        patches = data.get("patches") if isinstance(data, dict) else data
        if not isinstance(patches, list):
            return []
        preserved = []
        allowed_statuses = COMPILE_STATUSES | {"compiled"}
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            if patch.get("target_operator") not in SUPPORTED_COMPILE_TARGETS:
                continue
            if patch.get("status") not in allowed_statuses:
                continue
            preserved.append(self._mark_compiled(patch))
        return preserved

    def _merge_compiled_patches(
        self,
        source_patches: List[Dict[str, Any]],
        selected_patches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        for patch in [*source_patches, *selected_patches]:
            patch_id = patch.get("patch_id")
            patch_key = patch.get("patch_key")
            dedupe_key = patch_id or patch_key
            if dedupe_key and dedupe_key in seen:
                continue
            if dedupe_key:
                seen.add(dedupe_key)
            merged.append(patch)
        return merged

    def _compile_graph(
        self,
        graph_text: str,
        patches: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        targets = sorted({patch.get("target_operator") for patch in patches})
        inserted_nodes = []
        skipped = []
        text = self._ensure_imports(graph_text)
        if patches:
            text = self._ensure_patch_path(text)

        if self.compile_content_nodes:
            if self.content_repair_batches:
                text, inserted, batch_skips = self._insert_content_repair_batch_nodes(text)
                skipped.extend(batch_skips)
                inserted_nodes.extend(inserted)
            else:
                text, inserted = self._insert_content_repair_node(text)
                if inserted:
                    inserted_nodes.append(
                        {
                            "node_name": "content_repair",
                            "target_operator": "AnswerGenerate",
                            "position": "after AnswerGenerate before answer_generate_patch and ScEnsemble",
                            "families": self.content_repair_families,
                            "edit_kind": "insert_node",
                        }
                    )
                else:
                    skipped.append({"patch_id": None, "reason": "could_not_insert_content_repair_node"})

        if "AnswerGenerate" in targets:
            text, inserted = self._insert_answer_generate_node(text)
            if inserted:
                inserted_nodes.append(
                    {
                        "node_name": "answer_generate_patch",
                        "target_operator": "AnswerGenerate",
                        "position": "after content repair nodes before ScEnsemble",
                    }
                )
            else:
                skipped.extend(
                    [
                        {"patch_id": patch.get("patch_id"), "reason": "could_not_insert_answer_generate_node"}
                        for patch in patches
                        if patch.get("target_operator") == "AnswerGenerate"
                    ]
                )

        if "ScEnsemble" in targets:
            text, inserted = self._insert_scensemble_node(text)
            if inserted:
                inserted_nodes.append(
                    {
                        "node_name": "sc_ensemble_patch",
                        "target_operator": "ScEnsemble",
                        "position": "after ScEnsemble selection before return",
                    }
                )
            else:
                skipped.extend(
                    [
                        {"patch_id": patch.get("patch_id"), "reason": "could_not_insert_scensemble_node"}
                        for patch in patches
                        if patch.get("target_operator") == "ScEnsemble"
                    ]
                )
        return text, inserted_nodes, skipped

    def _ensure_imports(self, graph_text: str) -> str:
        text = graph_text
        if "from pathlib import Path" not in text:
            text = "from pathlib import Path\n" + text
        required = ["CompiledPatchNode"]
        if self.compile_content_nodes and not self.content_repair_batches:
            required.append("ContentRepairNode")
        if self.content_repair_batches:
            for batch in self.content_repair_batches:
                class_name = self.CONTENT_REPAIR_BATCHES[batch]["class_name"]
                if class_name not in required:
                    required.append(class_name)
        import_match = re.search(r"^from patch_evolution\.workflow_nodes import ([^\n]+)\n", text, flags=re.MULTILINE)
        if import_match:
            names = [name.strip() for name in import_match.group(1).split(",") if name.strip()]
            for name in required:
                if name not in names:
                    names.append(name)
            text = text[: import_match.start()] + f"from patch_evolution.workflow_nodes import {', '.join(names)}\n" + text[import_match.end() :]
        else:
            marker = "from scripts.async_llm import create_llm_instance\n"
            import_line = f"from patch_evolution.workflow_nodes import {', '.join(required)}\n"
            if marker in text:
                text = text.replace(marker, marker + import_line, 1)
            else:
                text = import_line + text
        return text

    def _ensure_patch_path(self, graph_text: str) -> str:
        if "self._compiled_patch_path" in graph_text:
            return graph_text
        pattern = r"(self\.llm\s*=\s*create_llm_instance\(llm_config\)\n)"
        replacement = "\\1        self._compiled_patch_path = Path(__file__).with_name(\"compiled_patches.json\")\n"
        return re.sub(pattern, replacement, graph_text, count=1)

    def _insert_answer_generate_node(self, graph_text: str) -> Tuple[str, bool]:
        text = graph_text
        if "self.answer_generate_patch" not in text:
            marker = "        self.answer_gen = operator.AnswerGenerate(self.llm)\n"
            insertion = (
                marker
                + "        self.answer_generate_patch = CompiledPatchNode(\n"
                + "            self.llm,\n"
                + "            self._compiled_patch_path,\n"
                + "            target_operator=\"AnswerGenerate\",\n"
                + "            node_name=\"AnswerGeneratePatch\",\n"
                + "        )\n"
            )
            if marker not in text:
                return text, False
            text = text.replace(marker, insertion, 1)

        if "await self.answer_generate_patch(solution, operator_input=problem)" in text:
            return text, True
        insertion = (
            "        solutions = [\n"
            "            await self.answer_generate_patch(solution, operator_input=problem)\n"
            "            for solution in solutions\n"
            "        ]\n"
        )
        markers = [
            "        # Extract the 'answer' field from each solution\n",
            '        # Extract the "answer" field from each solution\n',
            "        solution_list =",
        ]
        for marker in markers:
            if marker in text:
                if marker.startswith("        solution_list"):
                    text = text.replace(marker, insertion + marker, 1)
                else:
                    text = text.replace(marker, insertion + marker, 1)
                return text, True
        return text, False

    def _insert_content_repair_node(self, graph_text: str) -> Tuple[str, bool]:
        text = graph_text
        if "self.content_repair" not in text:
            marker = "        self.answer_gen = operator.AnswerGenerate(self.llm)\n"
            families_text = json.dumps(self.content_repair_families, ensure_ascii=False)
            insertion = (
                marker
                + "        self.content_repair = ContentRepairNode(\n"
                + "            self.llm,\n"
                + f"            families={families_text},\n"
                + "            node_name=\"ContentRepair\",\n"
                + "        )\n"
            )
            if marker not in text:
                return text, False
            text = text.replace(marker, insertion, 1)

        if "await self.content_repair(solution, operator_input=problem)" in text:
            return text, True

        content_block = (
            "        solutions = [\n"
            "            await self.content_repair(solution, operator_input=problem)\n"
            "            for solution in solutions\n"
            "        ]\n"
        )
        answer_patch_block = (
            "        solutions = [\n"
            "            await self.answer_generate_patch(solution, operator_input=problem)\n"
            "            for solution in solutions\n"
            "        ]\n"
        )
        if answer_patch_block in text:
            text = text.replace(answer_patch_block, answer_patch_block + content_block, 1)
            return text, True

        markers = [
            "        # Extract the 'answer' field from each solution\n",
            '        # Extract the "answer" field from each solution\n',
            "        solution_list =",
        ]
        for marker in markers:
            if marker in text:
                text = text.replace(marker, content_block + marker, 1)
                return text, True
        return text, False

    def _insert_content_repair_batch_nodes(self, graph_text: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        text = graph_text
        inserted_nodes = []
        skipped = []
        for batch in self.content_repair_batches:
            spec = self.CONTENT_REPAIR_BATCHES[batch]
            node_name = spec["node_name"]
            class_name = spec["class_name"]
            if f"self.{node_name}" not in text:
                marker = "        self.answer_gen = operator.AnswerGenerate(self.llm)\n"
                insertion = (
                    marker
                    + f"        self.{node_name} = {class_name}(\n"
                    + "            self.llm,\n"
                    + f"            node_name={json.dumps(class_name.replace('Node', ''))},\n"
                    + "        )\n"
                )
                if marker not in text:
                    skipped.append({"patch_id": None, "reason": f"could_not_insert_{node_name}_definition"})
                    continue
                text = text.replace(marker, insertion, 1)

            call = f"await self.{node_name}(solution, operator_input=problem)"
            if call not in text:
                block = (
                    "        solutions = [\n"
                    f"            {call}\n"
                    "            for solution in solutions\n"
                    "        ]\n"
                )
                text, ok = self._insert_solution_repair_block(text, block)
                if not ok:
                    skipped.append({"patch_id": None, "reason": f"could_not_insert_{node_name}_call"})
                    continue
            inserted_nodes.append(
                {
                    "node_name": node_name,
                    "target_operator": "AnswerGenerate",
                    "position": "after AnswerGenerate before answer_generate_patch and ScEnsemble",
                    "families": spec["families"],
                    "batch": batch,
                    "class_name": class_name,
                    "edit_kind": "insert_node",
                }
            )
        return text, inserted_nodes, skipped

    def _insert_solution_repair_block(self, graph_text: str, block: str) -> Tuple[str, bool]:
        text = graph_text
        existing_content_calls = list(
            re.finditer(
                r"        solutions = \[\n            await self\.(?:answer_generate_patch|content_repair|missing_answer_recovery|answer_canonicalizer|numeric_verifier|multi_span_extractor|answer_type_refiner|duration_normalizer)\(solution, operator_input=problem\)\n            for solution in solutions\n        \]\n",
                text,
            )
        )
        if existing_content_calls:
            last = existing_content_calls[-1]
            return text[: last.end()] + block + text[last.end() :], True

        markers = [
            "        # Extract the 'answer' field from each solution\n",
            '        # Extract the "answer" field from each solution\n',
            "        solution_list =",
        ]
        for marker in markers:
            if marker in text:
                return text.replace(marker, block + marker, 1), True
        return text, False

    def _insert_scensemble_node(self, graph_text: str) -> Tuple[str, bool]:
        text = graph_text
        if "self.sc_ensemble_patch" not in text:
            marker = "        self.ensemble = operator.ScEnsemble(self.llm)\n"
            insertion = (
                marker
                + "        self.sc_ensemble_patch = CompiledPatchNode(\n"
                + "            self.llm,\n"
                + "            self._compiled_patch_path,\n"
                + "            target_operator=\"ScEnsemble\",\n"
                + "            node_name=\"ScEnsemblePatch\",\n"
                + "        )\n"
            )
            if marker not in text:
                return text, False
            text = text.replace(marker, insertion, 1)

        if "await self.sc_ensemble_patch(final_answer, operator_input=problem)" in text:
            return text, True
        pattern = r"(        final_answer\s*=\s*await self\.ensemble\([^\n]+\)\n)"
        replacement = "\\1        final_answer = await self.sc_ensemble_patch(final_answer, operator_input=problem)\n"
        text, count = re.subn(pattern, replacement, text, count=1)
        return text, count > 0
