import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contrastive_experience.experience_library import update_library_from_verification


def _load_json_arg(value: str):
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as fin:
            return json.load(fin)
    return json.loads(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Update contrastive experience library")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--child", required=True)
    parser.add_argument("--edit_description", required=True, help="JSON string or JSON file path")
    parser.add_argument("--verification_results", required=True, help="JSON string or JSON file path")
    parser.add_argument("--root_dir", default="experience_library")
    return parser.parse_args()


def main():
    args = parse_args()
    data = update_library_from_verification(
        dataset=args.dataset,
        parent=args.parent,
        child=args.child,
        edit_description=_load_json_arg(args.edit_description),
        verification_results=_load_json_arg(args.verification_results),
        root_dir=args.root_dir,
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
