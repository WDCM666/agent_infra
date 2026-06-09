from .base import AgentOutput, BaseAgent


class ReflexionAgent(BaseAgent):
    """Baseline that injects past reflections into the next prompt."""

    def __init__(self, prompt_builder=None, action_parser=None):
        super().__init__(prompt_builder=prompt_builder, action_parser=action_parser)
        self.reflections = []

    def add_reflection(self, reflection: str) -> None:
        if reflection:
            self.reflections.append(reflection)

    def act(self, observation, model_output=None, **kwargs) -> AgentOutput:
        prompt = self.build_prompt(observation, reflections=self.reflections, **kwargs)
        action = self.parse_action(model_output if model_output is not None else prompt)
        return AgentOutput(action=action, prompt=prompt, metadata={"agent": "reflexion", "num_reflections": len(self.reflections)})
