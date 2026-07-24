import re


class ActionParser:
    """Extracts executable actions from model text."""

    @staticmethod
    def _clean_action(action: str) -> str:
        return str(action).strip().lstrip("> ").strip()

    def parse(self, text: str) -> str:
        text = str(text).strip()
        # 1) Try <action>...</action> XML-style tags
        match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return self._clean_action(match.group(1))
        # 2) Try "Action: ..." markdown-style
        match = re.search(r"Action\s*:\s*(.*)", text, flags=re.IGNORECASE)
        if match:
            return self._clean_action(match.group(1).strip().splitlines()[0])
        # 3) Fallback — return the last non-empty line
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return self._clean_action(lines[-1]) if lines else ""
