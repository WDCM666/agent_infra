class StateUpdater:
    """Generic state updater used when no environment-specific tracker exists."""

    def update(self, state, observation=None, action=None, info=None):
        next_state = dict(state or {})
        next_state.update({
            "last_observation": observation,
            "last_action": action,
            "last_info": info or {},
        })
        return next_state
