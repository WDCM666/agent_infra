class ActionNormalizer:
    """Normalizes model actions before environment projection."""

    def normalize(self, action: str) -> str:
        return " ".join(str(action).strip().split())
