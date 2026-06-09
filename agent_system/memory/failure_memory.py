from .experience_memory import ExperienceMemory


class FailureMemory(ExperienceMemory):
    """Long-term memory for failed attempts and avoidable action patterns."""

    def add_failure(self, observation, action, reason=None, **metadata) -> None:
        self.add({
            "observation": observation,
            "action": action,
            "reason": reason,
            **metadata,
        })
