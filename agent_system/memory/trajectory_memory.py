from .memory import SimpleMemory


class TrajectoryMemory(SimpleMemory):
    """Stores per-environment trajectories and exposes recent interaction history."""

    def export_trajectories(self):
        return self._data or []
