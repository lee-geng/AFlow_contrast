import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from analysis.blame_attributor import RuleBasedBlameAttributor
from data.build_revision_dataset import build_samples, write_jsonl
from data.export_dpo_dataset import export_dpo
from models.workflow_revision_model import RevisionModelConfig, WorkflowRevisionModel
from scripts.optimizer_utils.revision_prior import RevisionPriorController, append_prior_log


class RevisionPipelineTests(unittest.TestCase):
    def _mk_tmp(self) -> Path:
        root = Path(__file__).resolve().parents[1] / ".tmp_tests" / str(uuid.uuid4())
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _mock_nodes(self):
        from data.aflow_adapters import WorkflowNode

        nodes = {
            "p1": WorkflowNode(
                node_id="p1",
                parent_id=None,
                workflow="class Workflow:\n    async def __call__(self, problem):\n        x = await self.retrieve(problem)\n        y = await self.generate(x)",
                edit_text="",
                score_mean=0.6,
                score_std=0.01,
                task_family="MATH",
                task_summary="math reasoning",
                log_feedback={"failure_summary": "answer unsupported without verification", "trace_summary": "retrieve then answer"},
            ),
            "c_pos": WorkflowNode(
                node_id="c_pos",
                parent_id="p1",
                workflow="class Workflow:\n    async def __call__(self, problem):\n        x = await self.retrieve(problem)\n        v = await self.review(x)\n        y = await self.generate(v)",
                edit_text="insert verification block after retrieval",
                score_mean=0.68,
                score_std=0.02,
                task_family="MATH",
                task_summary="math reasoning",
                log_feedback={},
            ),
            "c_neg": WorkflowNode(
                node_id="c_neg",
                parent_id="p1",
                workflow="class Workflow:\n    async def __call__(self, problem):\n        y = await self.generate(problem)",
                edit_text="only rewrite answer prompt wording",
                score_mean=0.58,
                score_std=0.02,
                task_family="MATH",
                task_summary="math reasoning",
                log_feedback={},
            ),
        }
        return nodes

    def test_build_revision_pairs(self):
        rows, stats = build_samples(
            nodes=self._mock_nodes(),
            pos_threshold=0.03,
            neg_threshold=0.0,
            max_score_std=0.2,
            drop_unknown_blame_if_weak_feedback=False,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["num_sibling_pairs"], 1)
        self.assertGreater(rows[0]["chosen_edit"]["delta_score"], 0.03)
        self.assertLessEqual(rows[0]["rejected_edit"]["delta_score"], 0.0)

    def test_blame_output_structure(self):
        attributor = RuleBasedBlameAttributor()
        blame = attributor.attribute(
            parent_workflow=self._mock_nodes()["p1"].workflow,
            execution_feedback={"failure_summary": "unsupported answer lacks verification", "trace_summary": "retrieve->answer"},
            edit_text="insert verification block",
        )
        self.assertIn(blame.blame_type, {"verification_missing", "unknown", "retrieval_error", "planning_error", "tool_usage_error", "patch_incomplete"})
        self.assertTrue(blame.target_block.startswith("b"))

    def test_train_and_generate_revision_model(self):
        rows, _ = build_samples(
            nodes=self._mock_nodes(),
            pos_threshold=0.03,
            neg_threshold=0.0,
            max_score_std=0.2,
            drop_unknown_blame_if_weak_feedback=False,
        )
        model = WorkflowRevisionModel(RevisionModelConfig(feature_dim=256, lora_rank=4))
        model.train_sft(rows=rows, epochs=1, learning_rate=0.3, batch_size=1)
        proposal = model.generate_proposal(rows[0])
        self.assertTrue(model.valid_proposal_format(proposal))

    def test_export_dpo(self):
        rows, _ = build_samples(
            nodes=self._mock_nodes(),
            pos_threshold=0.03,
            neg_threshold=0.0,
            max_score_std=0.2,
            drop_unknown_blame_if_weak_feedback=False,
        )
        dpo = export_dpo(rows)
        self.assertEqual(len(dpo), 1)
        self.assertIn("prompt", dpo[0])
        self.assertIn("chosen", dpo[0])
        self.assertIn("rejected", dpo[0])

    def test_prior_invocation_and_logging(self):
        rows, _ = build_samples(
            nodes=self._mock_nodes(),
            pos_threshold=0.03,
            neg_threshold=0.0,
            max_score_std=0.2,
            drop_unknown_blame_if_weak_feedback=False,
        )
        model = WorkflowRevisionModel(RevisionModelConfig(feature_dim=256, lora_rank=4))
        model.train_sft(rows=rows, epochs=1, learning_rate=0.3, batch_size=1)

        tmp = self._mk_tmp()
        ckpt = tmp / "ckpt"
        model.save(str(ckpt))

        controller = RevisionPriorController(model_dir=str(ckpt), mode="generate_then_search", rank_candidates=2)
        context = controller.build_context(
            dataset="MATH",
            parent_round=1,
            parent_workflow=rows[0]["parent_workflow"],
            parent_score=float(rows[0]["parent_score"]),
            log_data=rows[0]["execution_feedback"]["failure_summary"],
            task_summary="math reasoning",
        )
        proposal = controller.generate_prior_edit(context)
        self.assertIn("edit_text", proposal)

        round_dir = tmp / "round_2"
        round_dir.mkdir(parents=True, exist_ok=True)
        append_prior_log(str(round_dir), {"prior_enabled": True, "executed": True, "delta_score": 0.1})
        log_file = round_dir / "revision_prior_log.json"
        self.assertTrue(log_file.exists())
        payload = json.loads(log_file.read_text(encoding="utf-8"))
        self.assertEqual(len(payload), 1)
        self.assertTrue(payload[0]["prior_enabled"])

    def test_cli_dataset_builder_and_dpo_export(self):
        tmp = self._mk_tmp()
        tree_path = tmp / "tree.json"
        output_dir = tmp / "out"
        dpo_path = tmp / "dpo.jsonl"

        tree = {
            "nodes": [
                {"id": "p1", "workflow": "wf parent", "score_mean": 0.5, "task_family": "MATH", "task_summary": "math"},
                {"id": "c1", "parent_id": "p1", "workflow": "wf child good", "edit": "insert verify", "score_mean": 0.56, "score_std": 0.02},
                {"id": "c2", "parent_id": "p1", "workflow": "wf child bad", "edit": "rewrite prompt", "score_mean": 0.49, "score_std": 0.02},
            ]
        }
        tree_path.write_text(json.dumps(tree, ensure_ascii=False), encoding="utf-8")

        cmd1 = [
            sys.executable,
            "data/build_revision_dataset.py",
            "--input_path",
            str(tree_path),
            "--output_dir",
            str(output_dir),
            "--task_family",
            "MATH",
            "--val_ratio",
            "0.0",
            "--keep_unknown_blame",
        ]
        subprocess.run(cmd1, cwd=Path(__file__).resolve().parents[1], check=True)

        train_path = output_dir / "train_revision.jsonl"
        self.assertTrue(train_path.exists())

        cmd2 = [
            sys.executable,
            "data/export_dpo_dataset.py",
            "--input_path",
            str(train_path),
            "--output_path",
            str(dpo_path),
        ]
        subprocess.run(cmd2, cwd=Path(__file__).resolve().parents[1], check=True)
        self.assertTrue(dpo_path.exists())


if __name__ == "__main__":
    unittest.main()
