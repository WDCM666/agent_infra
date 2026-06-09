import json
from pathlib import Path
from typing import Iterable, List

from .base import Skill


class SkillBank:
    def __init__(self, skills: Iterable[Skill] = None):
        self.skills: List[Skill] = list(skills or [])

    def add(self, skill: Skill) -> None:
        self.skills.append(skill)

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Skill(**item) for item in data)

    def to_json(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [skill.__dict__ for skill in self.skills]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
