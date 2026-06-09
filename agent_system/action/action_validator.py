class ActionValidator:
    """Validates actions against optional allowed action sets."""

    def __init__(self, allowed_actions=None):
        self.allowed_actions = set(allowed_actions or [])

    def validate(self, action: str, allowed_actions=None) -> bool:
        candidates = set(allowed_actions or self.allowed_actions)
        if not candidates:
            return bool(str(action).strip())
        return str(action).strip() in candidates
