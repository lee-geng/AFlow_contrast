import json
import os
import re
import time
import traceback
from typing import List

from scripts.prompts.optimize_prompt import (
    WORKFLOW_CUSTOM_USE,
    WORKFLOW_INPUT,
    WORKFLOW_OPTIMIZE_PROMPT,
    WORKFLOW_TEMPLATE,
)
from scripts.logs import logger


class GraphUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def _module_prefix(self) -> str:
        return self.root_path.replace("\\", ".").replace("/", ".")

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def load_graph(self, round_number: int, workflows_path: str):
        workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{workflows_path}.round_{round_number}.graph"

        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "Workflow")
            return graph_class
        except ImportError as e:
            logger.error(f"Error loading graph for round {round_number}: {e}")
            raise

    def read_graph_files(self, round_number: int, workflows_path: str):
        prompt_file_path = os.path.join(workflows_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(workflows_path, f"round_{round_number}", "graph.py")

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            logger.error(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        pattern = r"class Workflow:.+"
        return re.findall(pattern, graph_load, re.DOTALL)

    def load_operators_description(self, operators: List[str]) -> str:
        path = f"{self.root_path}/workflows/template/operator.json"
        operators_description = ""
        for id, operator in enumerate(operators):
            operator_description = self._load_operator_description(id + 1, operator, path)
            operators_description += f"{operator_description}\n"
        return operators_description

    def _load_operator_description(self, id: int, operator_name: str, file_path: str) -> str:
        with open(file_path, "r") as f:
            operator_data = json.load(f)
            matched_data = operator_data[operator_name]
            desc = matched_data["description"]
            interface = matched_data["interface"]
            return f"{id}. {operator_name}: {desc}, with interface {interface})."

    def create_graph_optimize_prompt(
        self,
        experience: str,
        score: float,
        graph: str,
        prompt: str,
        operator_description: str,
        type: str,
        log_data: str,
    ) -> str:
        graph_input = WORKFLOW_INPUT.format(
            experience=experience,
            score=score,
            graph=graph,
            prompt=prompt,
            operator_description=operator_description,
            type=type,
            log=log_data,
        )
        graph_system = WORKFLOW_OPTIMIZE_PROMPT.format(type=type)
        return graph_input + WORKFLOW_CUSTOM_USE + graph_system

    def create_memory_guided_optimize_prompt(
        self,
        objective: str,
        workflow_context: str,
        failure_evidence: str,
        memory_guidance: str,
        constraints: str,
        output_requirements: str,
        type: str,
    ) -> str:
        graph_system = WORKFLOW_OPTIMIZE_PROMPT.format(type=type)
        sections = [
            "[Optimization Objective]",
            objective,
            "",
            "[Current Workflow Context]",
            workflow_context,
            "",
            "[Failure Evidence]",
            failure_evidence or "No failure evidence available.",
            "",
            memory_guidance or "[Memory-Guided Optimization]\nNo relevant memory found.\n",
            "",
            "[Constraints]",
            constraints,
            "",
            "[Required Output]",
            output_requirements,
            "",
            WORKFLOW_CUSTOM_USE,
            graph_system,
        ]
        return "\n".join(sections)

    def summarize_workflow(
        self,
        graph_content: str,
        prompt_content: str,
        graph_char_limit: int = 1200,
        prompt_char_limit: int = 800,
    ) -> tuple[str, str]:
        node_calls = re.findall(r"await self\.([a-zA-Z_][a-zA-Z0-9_]*)\(", graph_content)
        ordered_nodes: List[str] = []
        for item in node_calls:
            if item not in ordered_nodes:
                ordered_nodes.append(item)

        graph_body = re.sub(r"\s+", " ", graph_content).strip()
        prompt_body = re.sub(r"\s+", " ", prompt_content).strip()
        graph_summary = (
            f"Workflow nodes: {', '.join(ordered_nodes) if ordered_nodes else 'unknown'}. "
            f"Workflow code preview: {graph_body[:graph_char_limit]}"
        )
        prompt_vars = re.findall(r"([A-Z][A-Z0-9_]+)\s*=", prompt_content)
        prompt_summary = (
            f"Prompt vars: {', '.join(sorted(set(prompt_vars))) if prompt_vars else 'none'}. "
            f"Prompt preview: {prompt_body[:prompt_char_limit]}"
        )
        return graph_summary, prompt_summary

    async def get_graph_optimize_response(self, graph_optimize_node):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                response = graph_optimize_node.instruct_content.model_dump()
                return response
            except Exception as e:
                retries += 1
                logger.error(f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})")
                if retries == max_retries:
                    logger.info("Maximum retries reached. Skipping this sample.")
                    break
                traceback.print_exc()
                time.sleep(5)
        return None

    def write_graph_files(self, directory: str, response: dict, round_number: int, dataset: str):
        graph_body = self._prepare_graph_body_for_write(response["graph"])
        graph = WORKFLOW_TEMPLATE.format(
            graph=graph_body,
            round=round_number,
            dataset=dataset,
            module_prefix=self._module_prefix(),
        )

        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        with open(os.path.join(directory, "prompt.py"), "w", encoding="utf-8") as file:
            file.write(self._prepare_prompt_body_for_write(response["prompt"]))

        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")

    def _prepare_graph_body_for_write(self, graph_code: str) -> str:
        text = str(graph_code or "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"^\s*```(?:python)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)

        managed_import_patterns = [
            r"^\s*from typing import Literal\s*$",
            r"^\s*import [A-Za-z0-9_\.]+\.workflows\.template\.operator as operator\s*$",
            r"^\s*import [A-Za-z0-9_\.]+\.workflows\.round_\d+\.prompt as prompt_custom\s*$",
            r"^\s*from \.\.template import operator\s*$",
            r"^\s*from \. import prompt as prompt_custom\s*$",
            r"^\s*from scripts\.async_llm import create_llm_instance\s*$",
            r"^\s*from scripts\.evaluator import DatasetType\s*$",
        ]
        filtered_lines = []
        for line in text.splitlines():
            if any(re.match(pattern, line.strip()) for pattern in managed_import_patterns):
                continue
            filtered_lines.append(line)
        cleaned = "\n".join(filtered_lines).strip()
        return cleaned

    def _prepare_prompt_body_for_write(self, prompt_code: str) -> str:
        text = str(prompt_code or "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"^\s*```(?:python)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
        text = text.strip()

        has_real_assignment = bool(re.search(r"(?m)^[A-Z][A-Z0-9_]*\s*=", text))
        has_commented_assignment = bool(re.search(r"(?m)^\s*#\s*[A-Z][A-Z0-9_]*\s*=", text))
        if not has_real_assignment and has_commented_assignment:
            uncommented_lines = []
            for line in text.splitlines():
                uncommented_lines.append(re.sub(r"^\s*#\s?", "", line))
            text = "\n".join(uncommented_lines).strip()

        return text
