from .react_prompt_builder import ReActPromptBuilder


class SkillPromptBuilder(ReActPromptBuilder):
    def build(self, observation, skills=None, **kwargs) -> str:
        skill_text = [skill.to_prompt() if hasattr(skill, "to_prompt") else str(skill) for skill in skills or []]
        prompt = super().build(observation, **kwargs)
        skill_block = self.format_list("Useful skills", skill_text)
        return "\n\n".join(part for part in [skill_block, prompt] if part)
