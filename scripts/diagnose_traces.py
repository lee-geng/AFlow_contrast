import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contrastive_experience.report import diagnose_trace_file, write_diagnosis_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose contrastive AFlow traces")
    parser.add_argument("trace_file", help="Path to trace JSONL file")
    parser.add_argument("--success_threshold", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    report = diagnose_trace_file(args.trace_file, success_threshold=args.success_threshold)
    write_diagnosis_outputs(args.trace_file, report, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "stats": report["stats"],
                "candidate_bottleneck": report["localization"].get("candidate_operator"),
                "confidence": report["localization"].get("confidence"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
