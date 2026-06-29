import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic_core import to_jsonable_python


DEFAULT_LIBRARY = {
    "node_experience": [],
    "edge_edit_experience": [],
    "operator_edit_applicability": [],
    "failed_edits": [],
}


class ExperienceLibrary:
    def __init__(self, dataset: str, root_dir: str = "experience_library"):
        self.dataset = dataset
        self.path = Path(root_dir) / dataset / "experience.json"

    def load(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.path.exists():
            return {key: list(value) for key, value in DEFAULT_LIBRARY.items()}
        with self.path.open("r", encoding="utf-8") as fin:
            data = json.load(fin)
        for key in DEFAULT_LIBRARY:
            data.setdefault(key, [])
        return data

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fout:
            json.dump(data, fout, ensure_ascii=False, indent=2, default=to_jsonable_python)

    def append(self, section: str, record: Dict[str, Any]) -> Dict[str, Any]:
        if section not in DEFAULT_LIBRARY:
            raise ValueError(f"Unknown experience library section: {section}")
        data = self.load()
        data[section].append(record)
        self.save(data)
        return data


def update_library_from_verification(
    dataset: str,
    parent: str,
    child: str,
    edit_description: Dict[str, Any],
    verification_results: Dict[str, Any],
    root_dir: str = "experience_library",
) -> Dict[str, Any]:
    library = ExperienceLibrary(dataset, root_dir=root_dir)
    record = {
        "parent": parent,
        "child": child,
        **edit_description,
        **verification_results,
    }
    section = "edge_edit_experience" if verification_results.get("accepted") else "failed_edits"
    return library.append(section, record)

