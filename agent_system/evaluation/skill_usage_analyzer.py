from collections import Counter


class SkillUsageAnalyzer:
    def summarize(self, episodes):
        counter = Counter()
        for episode in episodes:
            counter.update(episode.get("skills", []))
        return dict(counter)
