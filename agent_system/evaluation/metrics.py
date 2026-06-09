class AgentMetrics:
    @staticmethod
    def success_rate(successes):
        values = list(successes)
        return sum(float(x) for x in values) / len(values) if values else 0.0

    @staticmethod
    def average_steps(step_counts):
        values = list(step_counts)
        return sum(values) / len(values) if values else 0.0
