class CostTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens += int(prompt_tokens)
        self.completion_tokens += int(completion_tokens)

    def snapshot(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }
