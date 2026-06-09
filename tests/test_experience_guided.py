import shutil
import unittest
import uuid
from pathlib import Path

from scripts.optimizer_utils.experience_guided import (
    AttributionAnalyzer,
    AttributionEvent,
    CandidateEvaluationSummary,
    ExecutionTrace,
    ExperienceCompressor,
    ExperienceGuidedConfig,
    ExperienceGuidedEvaluationProtocol,
    ExperienceGuidedValidationSampler,
    ExperienceGuidedController,
    MemoryGuidance,
    MemoryStore,
    OptimizationStateQuery,
    TaskEmbedder,
)


class ExperienceGuidedTests(unittest.IsolatedAsyncioTestCase):
    def _mk_tmp(self) -> Path:
        root = Path(__file__).resolve().parents[1] / ".tmp_tests" / str(uuid.uuid4())
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _workflow_source(self) -> str:
        return (
            "class Workflow:\n"
            "    async def __call__(self, problem):\n"
            "        draft = await self.custom(problem)\n"
            "        review = await self.review(draft)\n"
            "        answer = await self.ensemble(review)\n"
            "        return answer\n"
        )

    def test_attribution_analyzer_failure_and_success(self):
        analyzer = AttributionAnalyzer()
        workflow = self._workflow_source()
        failure_trace = ExecutionTrace(
            task_id="12",
            workflow_id="wf_1",
            workflow_version="1",
            task_pattern="geometry",
            node_execution_sequence=["b1:custom", "b2:review", "b3:ensemble"],
            node_summaries=[],
            final_output="Here is a revised version with better readability.",
            ground_truth="\\boxed{42}",
            score=0.0,
            outcome="failure",
            token_cost=0.02,
            runtime_seconds=1.1,
            evaluator_feedback={"question": "Prove the triangle statement", "expected_output": "\\boxed{42}"},
        )
        failure = analyzer.analyze(workflow, failure_trace)
        self.assertEqual(failure.outcome, "failure")
        self.assertTrue(failure.failed_node)
        self.assertIn(failure.failure_type, {"weak_revision", "verification_missing", "reasoning_error"})
        self.assertTrue(failure.patch_target)

        success_trace = ExecutionTrace(
            task_id="13",
            workflow_id="wf_1",
            workflow_version="1",
            task_pattern="counting",
            node_execution_sequence=["b1:custom", "b2:review", "b3:ensemble"],
            node_summaries=[],
            final_output="Therefore the answer is \\boxed{144}.",
            ground_truth="\\boxed{144}",
            score=1.0,
            outcome="success",
            token_cost=0.01,
            runtime_seconds=0.8,
            evaluator_feedback={"question": "How many ways", "expected_output": "\\boxed{144}"},
        )
        success = analyzer.analyze(workflow, success_trace)
        self.assertEqual(success.outcome, "success")
        self.assertTrue(success.contributing_nodes)
        self.assertTrue(success.successful_subworkflow)

    def test_task_embedder_fallback_clusters_related_tasks(self):
        embedder = TaskEmbedder(enabled=False, dim=32)
        geometry_a = embedder.embed_text("Triangle area with side lengths and angle")
        geometry_b = embedder.embed_text("Find the angle of a triangle from its sides")
        code_task = embedder.embed_text("Write a Python function that sorts a list")
        self.assertEqual(len(geometry_a), 32)
        self.assertGreater(embedder.similarity(geometry_a, geometry_b), embedder.similarity(geometry_a, code_task))
        self.assertTrue(embedder.cluster_id(geometry_a).startswith("emb_"))

    def test_experience_compressor_promotes_and_dedupes(self):
        compressor = ExperienceCompressor(min_support=2, min_confidence=0.5)
        events = [
            AttributionEvent(
                event_id="e1",
                task_id="1",
                workflow_id="wf_1",
                workflow_version="1",
                task_pattern="geometry",
                outcome="failure",
                attributed_nodes=["b1:custom"],
                failed_node="b2:review",
                failure_type="weak_revision",
                patch_target="b2:review",
                patch_suggestion="Add explicit verification after b2:review.",
                confidence=0.8,
            ),
            AttributionEvent(
                event_id="e2",
                task_id="2",
                workflow_id="wf_2",
                workflow_version="2",
                task_pattern="geometry",
                outcome="failure",
                attributed_nodes=["b1:custom"],
                failed_node="b2:review",
                failure_type="weak_revision",
                patch_target="b2:review",
                patch_suggestion="Add explicit verification after b2:review.",
                confidence=0.82,
            ),
            AttributionEvent(
                event_id="e3",
                task_id="3",
                workflow_id="wf_3",
                workflow_version="3",
                task_pattern="geometry",
                outcome="success",
                attributed_nodes=["b2:review", "b3:ensemble"],
                contributing_nodes=["b2:review", "b3:ensemble"],
                successful_subworkflow=["b2:review", "b3:ensemble"],
                confidence=0.9,
            ),
        ]
        memories = compressor.compress(events)
        self.assertEqual(len(memories), 1)
        self.assertGreaterEqual(memories[0].support["failure_cases"], 2)
        self.assertIn(memories[0].status, {"candidate", "validated"})

        low_support = compressor.compress(events[:1])
        self.assertEqual(low_support, [])

    def test_memory_store_save_load_and_query(self):
        root = self._mk_tmp()
        store = MemoryStore(str(root))
        trace = ExecutionTrace(
            task_id="1",
            workflow_id="wf_1",
            workflow_version="1",
            task_pattern="geometry",
            node_execution_sequence=["b1:custom"],
            node_summaries=[],
            final_output="42",
            ground_truth="24",
            score=0.0,
            outcome="failure",
            token_cost=0.01,
            runtime_seconds=0.5,
        )
        event = AttributionEvent(
            event_id="e1",
            task_id="1",
            workflow_id="wf_1",
            workflow_version="1",
            task_pattern="geometry",
            outcome="failure",
            attributed_nodes=["b1:custom"],
            failed_node="b1:custom",
            failure_type="reasoning_error",
            patch_target="b1:custom",
            patch_suggestion="Add check",
            confidence=0.75,
        )
        store.add_trace(trace)
        store.add_event(event)
        self.assertEqual(len(store.load_traces()), 1)
        self.assertEqual(len(store.load_events()), 1)

        compressor = ExperienceCompressor(min_support=1, min_confidence=0.1)
        memories = store.compress(compressor)
        self.assertEqual(len(memories), 1)
        queried = store.query(task_pattern="geometry", failure_type="reasoning_error", failed_node="b1:custom")
        self.assertEqual(len(queried), 1)
        embedding_query = store.query(task_embedding=[0.0] * 64, limit=5)
        self.assertIsInstance(embedding_query, list)

    def test_memory_store_query_by_embedding(self):
        root = self._mk_tmp()
        store = MemoryStore(str(root))
        embedder = TaskEmbedder(enabled=False, dim=32)
        event = AttributionEvent(
            event_id="e1",
            task_id="1",
            workflow_id="wf_1",
            workflow_version="1",
            task_pattern="emb_11110000",
            task_text="Triangle area with side lengths",
            task_embedding=embedder.embed_text("Triangle area with side lengths"),
            task_cluster="emb_11110000",
            outcome="failure",
            attributed_nodes=["b1:custom"],
            failed_node="b1:custom",
            failure_type="reasoning_error",
            patch_target="b1:custom",
            patch_suggestion="Add check",
            confidence=0.75,
        )
        store.add_event(event)
        memories = store.compress(ExperienceCompressor(min_support=1, min_confidence=0.1))
        self.assertEqual(len(memories), 1)
        queried = store.query(
            task_embedding=embedder.embed_text("Compute a triangle area from edges"),
            similarity_threshold=0.1,
            limit=3,
        )
        self.assertEqual(len(queried), 1)

    def test_optimization_state_query_and_memory_guidance(self):
        root = self._mk_tmp()
        config = ExperienceGuidedConfig(
            enabled=True,
            store_path=str(root),
            min_support=1,
            min_confidence=0.1,
            task_embedding_enabled=False,
            use_optimization_state_memory_query=True,
        )
        controller = ExperienceGuidedController(dataset="MATH", config=config)
        controller.store.add_event(
            AttributionEvent(
                event_id="f1",
                task_id="1",
                workflow_id="MATH_round_1",
                workflow_version="1",
                task_pattern="emb_a",
                outcome="failure",
                attributed_nodes=["b1:custom"],
                failed_node="b2:review",
                failure_type="verification_missing",
                patch_target="b2:review",
                patch_suggestion="Insert explicit answer verification after b2:review.",
                confidence=0.9,
            )
        )
        controller.store.add_event(
            AttributionEvent(
                event_id="s1",
                task_id="2",
                workflow_id="MATH_round_1",
                workflow_version="1",
                task_pattern="emb_a",
                outcome="success",
                attributed_nodes=["b2:review", "b3:ensemble"],
                successful_subworkflow=["b2:review", "b3:ensemble"],
                reuse_suggestion="Preserve b2:review -> b3:ensemble",
                confidence=0.9,
            )
        )
        controller.store.add_event(
            AttributionEvent(
                event_id="r1",
                task_id="3",
                workflow_id="MATH_round_9",
                workflow_version="9",
                task_pattern="emb_b",
                outcome="failure",
                attributed_nodes=["b1:custom"],
                failed_node="b4:custom",
                failure_type="verification_missing",
                patch_target="b4:custom",
                patch_suggestion="Broad prompt expansion around b4:custom.",
                confidence=0.95,
            )
        )
        memories = controller.compress()
        for memory in memories:
            if memory.failed_node == "b2:review":
                controller.store.update_status(memory.memory_id, "validated")
            if memory.successful_subworkflow:
                controller.store.update_status(memory.memory_id, "validated")
            elif memory.failed_node == "b4:custom":
                controller.store.update_status(memory.memory_id, "rejected")
        query: OptimizationStateQuery = controller.build_optimization_state_query(
            parent_round=1,
            parent_workflow_id="MATH_round_1",
            score=0.2,
            failure_cards=[
                {
                    "failure_type": "verification_missing",
                    "node_name": "b2:review",
                    "patch_target": "b2:review",
                    "mismatch_info": "expected=42 vs extracted=41",
                    "extract_result": "41",
                }
            ],
            graph_summary="Workflow nodes: custom, review, ensemble",
            prompt_summary="Prompt vars: XXX_PROMPT",
            recent_failed_modifications=["broad prompt expansion"],
            recent_successful_modifications=["preserve review then ensemble"],
        )
        self.assertIn("verification_missing", query.failure_types)
        self.assertIn("b2:review", query.failed_nodes)
        self.assertIn("41", query.extract_result_summary)
        guidance: MemoryGuidance = controller.build_memory_guidance(query)
        self.assertIn("b2:review", guidance.focus_nodes)
        self.assertIn("b2:review", guidance.preserve_nodes)
        self.assertTrue(any("Broad prompt expansion" in item for item in guidance.avoid_nodes_or_patches))
        text = controller.build_memory_guidance_text(guidance)
        self.assertIn("Validated memory:", text)
        self.assertIn("Rejected memory:", text)

    def test_validation_sampler_selects_targeted_guard_and_anchor(self):
        config = ExperienceGuidedConfig(
            enabled=True,
            store_path="unused",
            adaptive_validation_enabled=True,
            targeted_failure_size=2,
            success_guard_size=2,
            anchor_size=3,
            task_embedding_enabled=False,
            task_embedding_dim=32,
        )
        sampler = ExperienceGuidedValidationSampler(config)
        records = [
            {"_task_id": 0, "problem": "Triangle area problem"},
            {"_task_id": 1, "problem": "How many ways to choose"},
            {"_task_id": 2, "problem": "Solve the equation"},
            {"_task_id": 3, "problem": "Random probability"},
        ]
        parent_outcomes = {
            "0": AttributionEvent("e0", "0", "wf", "1", "geometry", "failure", ["b1"], failed_node="b1"),
            "1": AttributionEvent("e1", "1", "wf", "1", "counting", "success", ["b1"]),
            "2": AttributionEvent("e2", "2", "wf", "1", "algebra", "failure", ["b1"], failed_node="b1"),
            "3": AttributionEvent("e3", "3", "wf", "1", "counting", "success", ["b1"]),
        }
        memories = []
        plan_a = sampler.plan(records, memories, parent_outcomes)
        plan_b = sampler.plan(records, memories, parent_outcomes)
        self.assertEqual(len(plan_a.targeted_failure_set), 2)
        self.assertEqual(len(plan_a.success_guard_set), 2)
        self.assertEqual(plan_a.anchor_set, plan_b.anchor_set)

    async def test_protocol_stages_and_comparable_score(self):
        config = ExperienceGuidedConfig(
            enabled=True,
            store_path="unused",
            adaptive_validation_enabled=True,
            targeted_failure_size=2,
            success_guard_size=1,
            anchor_size=2,
            full_validation_top_k=1,
            task_embedding_enabled=False,
            task_embedding_dim=32,
        )
        protocol = ExperienceGuidedEvaluationProtocol(config)
        parent_outcomes = {
            "0": AttributionEvent("e0", "0", "wf_parent", "1", "geometry", "failure", ["b1"]),
            "1": AttributionEvent("e1", "1", "wf_parent", "1", "counting", "success", ["b1"]),
        }
        plan = ExperienceGuidedValidationSampler(config).plan(
            [{"_task_id": 0, "problem": "Triangle"}, {"_task_id": 1, "problem": "How many ways"}, {"_task_id": 2, "problem": "Equation"}],
            [],
            parent_outcomes,
        )

        async def runner(indices, stage_name, return_details):
            score_map = {"targeted": 0.8, "anchor": 0.7, "full": 0.75}
            results = []
            for idx in indices:
                score = 1.0 if idx in {0, 1} else 0.0
                if stage_name == "anchor":
                    score = 1.0 if idx in {0, 1} else 0.0
                if stage_name == "full":
                    score = 1.0 if idx in {0, 1} else 0.0
                results.append(
                    {
                        "task_index": idx,
                        "problem": {"_task_id": idx},
                        "result": ("q", "pred", "exp", score, 0.01),
                        "runtime_seconds": 0.1,
                    }
                )
            return {
                "score": score_map[stage_name],
                "avg_cost": 0.01,
                "total_cost": 0.01 * max(1, len(indices)),
                "details": {
                    "score": score_map[stage_name],
                    "avg_cost": 0.01,
                    "total_cost": 0.01 * max(1, len(indices)),
                    "columns": ["question", "prediction", "expected_output", "score", "cost"],
                    "results": results,
                },
                "task_ids": [str(i) for i in indices],
            }

        summary: CandidateEvaluationSummary = await protocol.evaluate_candidate(
            plan=plan,
            subset_runner=runner,
            parent_outcomes=parent_outcomes,
            historical_top_scores=[0.4],
        )
        self.assertIn("targeted", summary.stages_run)
        self.assertIn("anchor", summary.stages_run)
        self.assertIn(summary.score_source, {"anchor", "full"})
        self.assertGreaterEqual(summary.selection_score, summary.anchor_score - 1.0)
        self.assertNotEqual(summary.score_source, "screened_out")

    def test_config_can_disable_staged_validation(self):
        config = ExperienceGuidedConfig(
            enabled=True,
            adaptive_validation_enabled=True,
            use_staged_validation=False,
        )
        self.assertFalse(config.use_staged_validation)


if __name__ == "__main__":
    unittest.main()
