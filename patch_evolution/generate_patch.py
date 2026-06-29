import argparse
import asyncio
import json
from pathlib import Path

from patch_evolution.patch_registry import PatchRegistry
from patch_evolution.patch_synthesizer import PatchSynthesizer
from patch_evolution.probe import ProbeBeforeEdit
from patch_evolution.trace_ir import TraceIR, contrastive_record_to_trace_ir


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PatchSpec from a failed trace")
    parser.add_argument("--trace", required=True, help="TraceIR JSON or contrastive JSONL file")
    parser.add_argument("--output_dir", default="results/patch_evolution")
    return parser.parse_args()


def load_trace(path: str) -> TraceIR:
    trace_path = Path(path)
    if trace_path.suffix == ".jsonl":
        with trace_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                record = json.loads(line)
                if record.get("final_result") == "failure" or float(record.get("sample_score") or 0) == 0:
                    return contrastive_record_to_trace_ir(record)
        raise ValueError("No failed trace found in JSONL file")
    return TraceIR.load_json(path)


async def main_async():
    args = parse_args()
    trace = load_trace(args.trace)
    diagnosis = ProbeBeforeEdit().diagnose(trace)
    patches = await PatchSynthesizer().synthesize(trace, diagnosis)
    registry = PatchRegistry(args.output_dir)
    for patch in patches:
        registry.add_patch(patch)
        print(json.dumps(patch, ensure_ascii=False, indent=2))


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

