from abc import ABC, abstractmethod


class BasePromptBuilder(ABC):
    @abstractmethod
    def build(self, observation, **kwargs) -> str:
        raise NotImplementedError

    def format_list(self, title, items) -> str:
        if not items:
            return ""
        body = "\n".join(f"- {item}" for item in items)
        return f"{title}:\n{body}"
