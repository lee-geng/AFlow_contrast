import argparse
import json

from patch_evolution.patch_registry import PatchRegistry


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect active guarded patches for a task/operator")
    parser.add_argument("--registry", default="results/patch_evolution")
    parser.add_argument("--target_operator", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    registry = PatchRegistry(args.registry)
    patches = registry.query_active_by_target(args.target_operator)
    print(json.dumps({"target_operator": args.target_operator, "active_patches": patches}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

