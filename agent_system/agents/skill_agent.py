from .base import AgentOutput, BaseAgent


class SkillAgent(BaseAgent):
    """Baseline that retrieves reusable skills before action generation."""

    def __init__(self, skill_retriever=None, prompt_builder=None, action_parser=None):
        super().__init__(prompt_builder=prompt_builder, action_parser=action_parser)
        self.skill_retriever = skill_retriever

    def retrieve_skills(self, query, **kwargs):
        if self.skill_retriever is None:
            return []
        return self.skill_retriever.retrieve(query=query, **kwargs)

    def act(self, observation, model_output=None, **kwargs) -> AgentOutput:
        skills = self.retrieve_skills(observation, **kwargs)
        prompt = self.build_prompt(observation, skills=skills, **kwargs)
        action = self.parse_action(model_output if model_output is not None else prompt)
        return AgentOutput(action=action, prompt=prompt, metadata={"agent": "skill", "num_skills": len(skills)})
