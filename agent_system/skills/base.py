from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Skill:
    name: str
    description: str
    steps: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
        steps = "\n".join(f"- {step}" for step in self.steps)
        return f"{self.name}: {self.description}\n{steps}".strip()
