import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from data.build_contrastive_dataset import build_contrastive_dataset, load_tree_nodes, write_jsonl


class ContrastiveDatasetTests(unittest.TestCase):
    def _make_workspace_tmp(self) -> Path:
        root = Path(__file__).resolve().parents[1] / ".tmp_tests" / str(uuid.uuid4())
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_build_sibling_and_fallback_pairs(self) -> None:
        mock_tree = {
            "nodes": [
                {"id": "p1", "workflow": "parent workflow one", "score_mean": 0.50, "task_family": "MATH", "task_summary": "math reasoning"},
                {"id": "c1", "parent_id": "p1", "workflow": "child workflow pos", "edit": "add self-check", "score_mean": 0.56},
                {"id": "c2", "parent_id": "p1", "workflow": "child workflow neg", "edit": "remove tool", "score_mean": 0.48},
                {"id": "c3", "parent_id": "p1", "workflow": "child workflow neutral", "edit": "tiny tweak", "score_mean": 0.52},
                {"id": "p2", "workflow": "parent workflow two", "score_mean": 0.40, "task_family": "MATH", "task_summary": "math reasoning"},
                {"id": "c4", "parent_id": "p2", "workflow": "child workflow only-positive", "edit": "better decomposition", "score_mean": 0.45},
            ]
        }

        tmp = self._make_workspace_tmp()
        tree_path = tmp / "mock_tree.json"
        tree_path.write_text(json.dumps(mock_tree, ensure_ascii=False), encoding="utf-8")

        nodes = load_tree_nodes(tree_path, task_family="MATH")
        samples = build_contrastive_dataset(nodes)
        self.assertEqual(len(samples), 2)

        sibling = [s for s in samples if s.sample_type == "sibling_pair"]
        fallback = [s for s in samples if s.sample_type == "parent_child_fallback"]
        self.assertEqual(len(sibling), 1)
        self.assertEqual(len(fallback), 1)

        self.assertGreater(sibling[0].positive_delta_score, 0.03)
        self.assertLessEqual(sibling[0].negative_delta_score, 0.0)

        out_path = tmp / "contrastive.jsonl"
        write_jsonl(samples, out_path)
        lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)

        parsed = [json.loads(line) for line in lines]
        required_fields = {
            "task_family",
            "parent_id",
            "parent_workflow",
            "positive_edit",
            "positive_child_workflow",
            "positive_delta_score",
            "negative_edit",
            "negative_child_workflow",
            "negative_delta_score",
        }
        for row in parsed:
            self.assertTrue(required_fields.issubset(set(row.keys())))

    def test_training_script_runs_one_epoch(self) -> None:
        sample = {
            "task_family": "MATH",
            "task_summary": "math reasoning",
            "parent_id": "p1",
            "parent_workflow": "parent",
            "parent_workflow_summary": "parent",
            "positive_edit": "add verification",
            "positive_child_workflow": "pos child",
            "positive_delta_score": 0.05,
            "negative_edit": "remove verification",
            "negative_child_workflow": "neg child",
            "negative_delta_score": -0.02,
            "sample_type": "sibling_pair",
        }

        tmp = self._make_workspace_tmp()
        train_path = tmp / "train.jsonl"
        val_path = tmp / "val.jsonl"
        out_dir = tmp / "ckpt"

        train_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
        val_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

        cmd = [
            sys.executable,
            "train/train_edit_ranker.py",
            "--train_path",
            str(train_path),
            "--val_path",
            str(val_path),
            "--output_dir",
            str(out_dir),
            "--epochs",
            "1",
            "--batch_size",
            "1",
        ]
        subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

        self.assertTrue((out_dir / "edit_scorer.npz").exists())
        self.assertTrue((out_dir / "config.json").exists())
        self.assertTrue((out_dir / "train_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
