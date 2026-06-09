from .base import BaseStateTracker
from .state_updater import StateUpdater


class ALFWorldStateTracker(BaseStateTracker):
    def __init__(self, updater=None):
        super().__init__()
        self.updater = updater or StateUpdater()

    def update(self, observation, action=None, info=None):
        self.state = self.updater.update(self.state, observation=observation, action=action, info=info)
        self.state["env_name"] = "alfworld"
        return self.snapshot()
