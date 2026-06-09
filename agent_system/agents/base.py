from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentOutput:
    action: str
    prompt: Optional[str] = None
    thought: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Interface for prompt-based and trainable agent baselines."""

    def __init__(self, prompt_builder=None, action_parser=None):
        self.prompt_builder = prompt_builder
        self.action_parser = action_parser

    @abstractmethod
    def act(self, observation: Any, **kwargs) -> AgentOutput:
        """Return the next action for a single environment observation."""
        raise NotImplementedError

    def build_prompt(self, observation: Any, **kwargs) -> str:
        if self.prompt_builder is None:
            return str(observation)
        return self.prompt_builder.build(observation=observation, **kwargs)

    def parse_action(self, model_output: str) -> str:
        if self.action_parser is None:
            return model_output.strip()
        return self.action_parser.parse(model_output)
