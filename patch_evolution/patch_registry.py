import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from patch_evolution.patch_spec import clone_patch


ACTIVE_STATUSES = {"validated", "active_guarded"}


class PatchRegistry:
    def __init__(self, root_dir: str = "results/patch_evolution"):
        self.root_dir = Path(root_dir)
        self.patch_dir = self.root_dir / "patches"
        self.telemetry_dir = self.root_dir / "telemetry"
        self.validation_dir = self.root_dir / "validation"
        self.consolidated_dir = self.root_dir / "consolidated"
        for path in (self.patch_dir, self.telemetry_dir, self.validation_dir, self.consolidated_dir):
            path.mkdir(parents=True, exist_ok=True)

    def patch_path(self, patch_id: str) -> Path:
        return self.patch_dir / f"{patch_id}.json"

    def save_patch(self, patch: Dict[str, Any]) -> None:
        patch = clone_patch(patch)
        with self.patch_path(patch["patch_id"]).open("w", encoding="utf-8") as fout:
            json.dump(patch, fout, ensure_ascii=False, indent=2)

    def load_patch(self, patch_id: str) -> Dict[str, Any]:
        with self.patch_path(patch_id).open("r", encoding="utf-8") as fin:
            return json.load(fin)

    def has_patch(self, patch_id: str) -> bool:
        return self.patch_path(patch_id).exists()

    def load_all(self) -> List[Dict[str, Any]]:
        patches = []
        for path in sorted(self.patch_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as fin:
                patches.append(json.load(fin))
        return patches

    def find_by_patch_key(self, patch_key: str) -> Optional[Dict[str, Any]]:
        if not patch_key:
            return None
        for patch in self.load_all():
            if patch.get("patch_key") == patch_key:
                return patch
        return None

    def add_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        self.save_patch(patch)
        return patch

    def query_active_by_target(self, target_operator: str) -> List[Dict[str, Any]]:
        matches = []
        for patch in self.load_all():
            if patch.get("status") in ACTIVE_STATUSES and patch.get("target_operator") == target_operator:
                matches.append(patch)
        return matches

    def update_telemetry(self, patch_id: str, increments: Dict[str, float]) -> Dict[str, Any]:
        patch = self.load_patch(patch_id)
        telemetry = patch.setdefault("telemetry", {})
        for key, value in increments.items():
            telemetry[key] = telemetry.get(key, 0) + value
        self.save_patch(patch)
        return patch

    def archive_disabled(self) -> List[str]:
        archived = []
        for patch in self.load_all():
            if patch.get("status") == "disabled":
                patch["status"] = "archived"
                self.save_patch(patch)
                archived.append(patch["patch_id"])
        return archived
