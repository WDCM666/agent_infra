from collections import Counter


class FailureAnalyzer:
    def summarize(self, episodes):
        reasons = [episode.get("failure_reason", "unknown") for episode in episodes if not episode.get("success")]
        return dict(Counter(reasons))
