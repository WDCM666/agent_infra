class SkillEvaluator:
    """Tracks whether retrieved skills help solve tasks."""

    def evaluate(self, skill, trajectory=None, success=None, reward=None, **kwargs):
        return {
            "skill": skill.name,
            "success": success,
            "reward": reward,
            "used": trajectory is not None,
        }
