from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStateTracker(ABC):
    def __init__(self):
        self.state: Dict[str, Any] = {}

    def reset(self, initial_state=None):
        self.state = dict(initial_state or {})
        return self.state

    @abstractmethod
    def update(self, observation, action=None, info=None):
        raise NotImplementedError

    def snapshot(self):
        return dict(self.state)
