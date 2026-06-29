import json
import threading
from pathlib import Path
from typing import Any, Dict, Union

from pydantic_core import to_jsonable_python


class TraceRecorder:
    def __init__(self, trace_path: Union[str, Path]):
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=to_jsonable_python)
        with self._lock:
            with self.trace_path.open("a", encoding="utf-8") as fout:
                fout.write(line + "\n")


def preview_value(value: Any, max_chars: int = 512, full_io: bool = False) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=to_jsonable_python)
        except TypeError:
            text = str(value)

    if full_io or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def default_run_id() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")
