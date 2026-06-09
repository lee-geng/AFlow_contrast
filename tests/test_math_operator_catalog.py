import json
import unittest
from pathlib import Path


class MathOperatorCatalogTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.operator_json = self.repo_root / "workspace_template_ab" / "MATH" / "workflows" / "template" / "operator.json"
        self.operator_py = self.repo_root / "workspace_template_ab" / "MATH" / "workflows" / "template" / "operator.py"

    def test_math_template_exposes_explicit_operators(self):
        payload = json.loads(self.operator_json.read_text(encoding="utf-8"))
        expected = {
            "Custom",
            "Plan",
            "AnswerGenerate",
            "Review",
            "Revise",
            "Verify",
            "ExtractAnswer",
            "ScEnsemble",
            "Programmer",
        }
        self.assertTrue(expected.issubset(set(payload.keys())))

    def test_operator_module_declares_matching_classes(self):
        text = self.operator_py.read_text(encoding="utf-8")
        for class_name in [
            "class Plan(",
            "class AnswerGenerate(",
            "class Review(",
            "class Revise(",
            "class Verify(",
            "class ExtractAnswer(",
        ]:
            self.assertIn(class_name, text)


if __name__ == "__main__":
    unittest.main()
