from .base import Skill


class SkillUpdater:
    """Creates or updates skills from trajectories."""

    def propose(self, name, description, trajectory=None, tags=None, **metadata):
        steps = []
        if trajectory:
            steps = [str(item.get("action", item)) for item in trajectory]
        return Skill(name=name, description=description, steps=steps, tags=tags or [], metadata=metadata)
