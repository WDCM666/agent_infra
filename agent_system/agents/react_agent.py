from .base import AgentOutput, BaseAgent


class ReActAgent(BaseAgent):
    """ReAct baseline that keeps thought and action channels explicit."""

    def act(self, observation, model_output=None, **kwargs) -> AgentOutput:
        prompt = self.build_prompt(observation, **kwargs)
        text = model_output if model_output is not None else prompt
        action = self.parse_action(text)
        thought = text.split("Action:", 1)[0].strip() if "Action:" in text else None
        return AgentOutput(action=action, prompt=prompt, thought=thought, metadata={"agent": "react"})
