from .base import AgentOutput, BaseAgent


class DirectAgent(BaseAgent):
    """Direct action baseline without explicit reasoning or retrieval."""

    def act(self, observation, model_output=None, **kwargs) -> AgentOutput:
        prompt = self.build_prompt(observation, **kwargs)
        action = self.parse_action(model_output if model_output is not None else prompt)
        return AgentOutput(action=action, prompt=prompt, metadata={"agent": "direct"})
