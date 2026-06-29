import json
import unittest
from pathlib import Path

from contrastive_experience.divergence import compute_operator_divergence, js_divergence
from contrastive_experience.experience_library import ExperienceLibrary
from contrastive_experience.grouping import group_traces, load_trace_records
from contrastive_experience.localization import localize_bottleneck
from contrastive_experience.prototypes import select_prototypes
from contrastive_experience.report import diagnose_trace_file, write_diagnosis_outputs
from contrastive_experience.trace_abstraction import (
    classify_output_style,
    classify_schema_type,
    infer_answer_type,
)
from contrastive_experience.verification import compute_net_utility, summarize_verification


def scratch_dir(name):
    path = Path.cwd() / "logs" / "contrastive_experience_tests" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_sample(sample_id, output, score, expected="Duke University"):
    return {
        "workflow_id": "round_1",
        "round_id": 1,
        "sample_id": sample_id,
        "dataset": "MATH",
        "prediction": output,
        "expected": expected,
        "sample_score": score,
        "operator_traces": [
            {
                "call_index": 0,
                "operator_id": "AnswerExtract#0",
                "operator_type": "Custom",
                "input_schema_type": "free_text",
                "input_output_style": "full_sentence",
                "schema_type": "free_text",
                "output_style": classify_output_style(output),
                "answer_type": infer_answer_type(output),
                "output_length": len(output.split()),
                "parse_status": "ok",
                "input_preview": "question",
                "output_preview": output,
            }
        ],
    }


class ContrastiveExperienceTests(unittest.TestCase):
    def test_trace_abstraction(self):
        self.assertEqual(classify_schema_type({"answer": "Duke"}), "json_dict")
        self.assertEqual(classify_schema_type("[1, 2]"), "json_list")
        self.assertEqual(classify_schema_type("{bad json"), "invalid_json")
        self.assertEqual(classify_output_style("Duke University"), "concise_entity")
        self.assertEqual(classify_output_style("The answer is Duke University."), "full_sentence")
        self.assertEqual(classify_output_style("\\boxed{Poland}"), "boxed_answer")
        self.assertEqual(infer_answer_type("2015"), "number")
        self.assertEqual(infer_answer_type("yes"), "yes_no")

    def test_js_divergence(self):
        self.assertEqual(js_divergence({"a": 1.0}, {"a": 1.0}), 0.0)
        self.assertGreater(js_divergence({"a": 1.0}, {"b": 1.0}), 0.9)

    def test_grouping_divergence_localization_and_prototypes(self):
        records = [
            make_sample("s1", "Duke University", 1),
            make_sample("s2", "Poland", 1, expected="Poland"),
            make_sample("s3", "2015", 1, expected="2015"),
            make_sample("f1", "The answer is Duke University.", 0),
            make_sample("f2", "\\boxed{Poland}", 0, expected="Poland"),
            make_sample("f3", "Polish independence", 0, expected="Poland"),
        ]
        grouped = group_traces(records)
        self.assertEqual(grouped["stats"]["success_count"], 3)
        self.assertEqual(grouped["stats"]["failure_count"], 3)

        divergence = compute_operator_divergence(grouped["success_traces"], grouped["failure_traces"])
        self.assertEqual(divergence[0]["operator_id"], "AnswerExtract#0")
        self.assertGreater(divergence[0]["output_style_div"], 0.0)

        localization = localize_bottleneck(divergence)
        self.assertEqual(localization["candidate_operator"], "AnswerExtract#0")
        self.assertIn("Candidate bottleneck", localization["evidence"])

        prototypes = select_prototypes(
            grouped["success_traces"],
            grouped["failure_traces"],
            localization["candidate_operator"],
        )
        self.assertGreaterEqual(len(prototypes["failure_prototypes"]), 1)
        self.assertGreaterEqual(len(prototypes["success_prototypes"]), 1)

    def test_report_files(self):
        records = [
            make_sample("s1", "Duke University", 1),
            make_sample("f1", "The answer is Duke University.", 0),
        ]
        tmp = scratch_dir("report")
        trace_file = tmp / "trace.jsonl"
        with trace_file.open("w", encoding="utf-8") as fout:
            for record in records:
                fout.write(json.dumps(record) + "\n")
        loaded = load_trace_records(trace_file)
        self.assertEqual(len(loaded), 2)
        report = diagnose_trace_file(trace_file)
        write_diagnosis_outputs(trace_file, report)
        self.assertTrue((tmp / "divergence_report.json").exists())
        self.assertTrue((tmp / "prototypes.json").exists())
        self.assertTrue((tmp / "attribution_prompt.txt").exists())

    def test_experience_library_and_verification(self):
        tmp = scratch_dir("library")
        library = ExperienceLibrary("MATH", root_dir=tmp)
        library.save({"node_experience": [], "edge_edit_experience": [], "operator_edit_applicability": [], "failed_edits": []})
        data = library.append("node_experience", {"workflow_id": "W1", "score": 0.7})
        self.assertEqual(data["node_experience"][0]["workflow_id"], "W1")
        self.assertTrue((tmp / "MATH" / "experience.json").exists())

        utility = compute_net_utility(0.5, 0.1, 0.1, 0.0)
        self.assertAlmostEqual(utility, 0.35)
        summary = summarize_verification(0.5, 0.1, 0.1, 0.0)
        self.assertTrue(summary["accepted"])


if __name__ == "__main__":
    unittest.main()
