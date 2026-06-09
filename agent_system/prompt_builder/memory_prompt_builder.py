from .react_prompt_builder import ReActPromptBuilder


class MemoryPromptBuilder(ReActPromptBuilder):
    def build(self, observation, memories=None, **kwargs) -> str:
        memory_text = [str(memory) for memory in memories or []]
        prompt = super().build(observation, **kwargs)
        memory_block = self.format_list("Relevant memories", memory_text)
        return "\n\n".join(part for part in [memory_block, prompt] if part)
