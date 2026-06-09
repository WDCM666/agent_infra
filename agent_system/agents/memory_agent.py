from .base import AgentOutput, BaseAgent


class MemoryAgent(BaseAgent):
    """Baseline that retrieves memory records before constructing prompts."""

    def __init__(self, memory=None, retriever=None, prompt_builder=None, action_parser=None):
        super().__init__(prompt_builder=prompt_builder, action_parser=action_parser)
        self.memory = memory
        self.retriever = retriever

    def retrieve_memory(self, query, **kwargs):
        if self.retriever is not None:
            return self.retriever.retrieve(query=query, **kwargs)
        if self.memory is not None and hasattr(self.memory, "search"):
            return self.memory.search(query=query, **kwargs)
        return []

    def act(self, observation, model_output=None, **kwargs) -> AgentOutput:
        memories = self.retrieve_memory(observation, **kwargs)
        prompt = self.build_prompt(observation, memories=memories, **kwargs)
        action = self.parse_action(model_output if model_output is not None else prompt)
        return AgentOutput(action=action, prompt=prompt, metadata={"agent": "memory", "num_memories": len(memories)})
