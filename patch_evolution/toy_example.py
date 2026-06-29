import argparse
import asyncio
import json
from pathlib import Path

from patch_evolution.consolidator import OfflineConsolidator
from patch_evolution.guarded_runtime import GuardedRuntime
from patch_evolution.patch_registry import PatchRegistry
from patch_evolution.patch_synthesizer import PatchSynthesizer
from patch_evolution.probe import ProbeBeforeEdit
from patch_evolution.trace_ir import NodeState, OperatorState, TraceIR, TraceMetadata


def parse_args():
    parser = argparse.ArgumentParser(description="Run a no-LLM Patch-as-Hypothesis toy example")
    parser.add_argument("--output_dir", default="logs/patch_evolution_toy")
    return parser.parse_args()


def make_failed_trace() -> TraceIR:
    return TraceIR(
        metadata=TraceMetadata(
            trace_id="toy_trace_empty_solver",
            task_name="toy",
            sample_id="toy_sample_1",
            workflow_id="toy_workflow",
            success=False,
        ),
        input="Retrieve -> Solver -> Format toy input",
        final_output="",
        nodes=[
            NodeState(
                node_id="Solver#0",
                operator=OperatorState(
                    operator_name="Solver",
                    operator_input="toy evidence",
                    operator_output="",
                    metadata={"parse_status": "empty"},
                ),
            )
        ],
    )


async def main_async():
    args = parse_args()
    output_dir = Path(args.output_dir)
    registry = PatchRegistry(str(output_dir))
    trace = make_failed_trace()
    diagnosis = ProbeBeforeEdit().diagnose(trace)
    patch = (await PatchSynthesizer().synthesize(trace, diagnosis))[0]
    patch["status"] = "active_guarded"
    patch["fixer"]["fallback_output"] = {"response": "fallback answer from guarded patch"}
    registry.add_patch(patch)

    runtime = GuardedRuntime(registry)
    patched_output = await runtime.apply("Solver", "toy evidence", {}, metadata={"parse_status": "empty"})
    patch_after = registry.load_patch(patch["patch_id"])
    consolidation = OfflineConsolidator().consolidate([patch_after])

    result = {
        "trace": trace.to_dict(),
        "diagnosis": diagnosis,
        "patch_path": str(registry.patch_path(patch["patch_id"])),
        "patched_output": patched_output,
        "telemetry": patch_after["telemetry"],
        "consolidation": consolidation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "toy_result.json").open("w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

