from agent_system.action import ActionNormalizer, ActionParser, ActionValidator
from agent_system.memory import ExperienceMemory
from agent_system.prompt_builder import MemoryPromptBuilder, SkillPromptBuilder
from agent_system.skills import Skill, SkillBank, SkillRetriever
from agent_system.state import ALFWorldStateTracker


def test_action_pipeline():
    parser = ActionParser()
    normalizer = ActionNormalizer()
    validator = ActionValidator(["go north"])
    action = normalizer.normalize(parser.parse("Thought: inspect\nAction:   go   north"))
    assert action == "go north"
    assert validator.validate(action)


def test_memory_retrieval():
    memory = ExperienceMemory()
    memory.add({"task": "find apple", "action": "open fridge"})
    assert memory.search("apple", top_k=1)[0]["action"] == "open fridge"


def test_skill_retrieval_and_prompt():
    bank = SkillBank([Skill(name="search", description="search products", tags=["webshop"])])
    skills = SkillRetriever(bank).retrieve("webshop product search", top_k=1)
    prompt = SkillPromptBuilder().build("Need a red shirt", skills=skills)
    assert "Useful skills" in prompt


def test_memory_prompt_and_state_tracker():
    prompt = MemoryPromptBuilder().build("where am I?", memories=["avoid invalid pickup"])
    assert "Relevant memories" in prompt
    tracker = ALFWorldStateTracker()
    state = tracker.update("in kitchen", action="look")
    assert state["env_name"] == "alfworld"
