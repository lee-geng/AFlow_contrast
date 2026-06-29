import argparse
import json

from patch_evolution.patch_spec import load_patch, validate_contract_definition, validate_patch_schema
from patch_evolution.shadow_validation import ShadowValidator


def parse_args():
    parser = argparse.ArgumentParser(description="Validate PatchSpec schema and optional shadow metrics")
    parser.add_argument("--patch", required=True)
    parser.add_argument("--shadow_pool", default=None, help="Optional JSON file with failed_traces/success_traces states")
    return parser.parse_args()


def main():
    args = parse_args()
    patch = load_patch(args.patch)
    ok, errors = validate_patch_schema(patch)
    contract_ok, contract_errors = validate_contract_definition(patch)
    result = {"schema_ok": ok, "schema_errors": errors, "contract_ok": contract_ok, "contract_errors": contract_errors}
    if args.shadow_pool:
        with open(args.shadow_pool, "r", encoding="utf-8") as fin:
            pool = json.load(fin)
        result["shadow_validation"] = ShadowValidator().validate(
            patch,
            pool.get("failed_traces", []),
            pool.get("success_traces", []),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

