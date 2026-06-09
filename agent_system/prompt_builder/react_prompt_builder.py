from .base import BasePromptBuilder


class ReActPromptBuilder(BasePromptBuilder):
    def build(self, observation, history=None, **kwargs) -> str:
        parts = [
            "Use Thought and Action to solve the task.",
            self.format_list("History", history or []),
            f"Observation:\n{observation}",
            "Thought:",
        ]
        return "\n\n".join(part for part in parts if part)
