import argparse
import json

from patch_evolution.consolidator import OfflineConsolidator
from patch_evolution.patch_registry import PatchRegistry


def parse_args():
    parser = argparse.ArgumentParser(description="Consolidate Patch-as-Hypothesis registry")
    parser.add_argument("--registry", default="results/patch_evolution")
    return parser.parse_args()


def main():
    args = parse_args()
    registry = PatchRegistry(args.registry)
    patches = registry.load_all()
    result = OfflineConsolidator().consolidate(patches)
    output = registry.consolidated_dir / "consolidation_decisions.json"
    with output.open("w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

