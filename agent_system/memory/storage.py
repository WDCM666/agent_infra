import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class JsonlMemoryStorage:
    """Append-only JSONL storage for long-term agent memories."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def extend(self, records: Iterable[Dict[str, Any]]) -> None:
        for record in records:
            self.append(record)

    def load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
