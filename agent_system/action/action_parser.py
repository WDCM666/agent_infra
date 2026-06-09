import re


class ActionParser:
    """Extracts executable actions from model text."""

    def parse(self, text: str) -> str:
        text = str(text).strip()
        match = re.search(r"Action\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().splitlines()[0]
        return text.splitlines()[-1].strip() if text else ""
