class MemoryUsageAnalyzer:
    def summarize(self, episodes):
        counts = [len(episode.get("memories", [])) for episode in episodes]
        return {"avg_retrieved_memories": sum(counts) / len(counts) if counts else 0.0}
