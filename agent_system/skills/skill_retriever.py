class SkillRetriever:
    """Lexical skill retriever for baseline experiments."""

    def __init__(self, skill_bank):
        self.skill_bank = skill_bank

    def retrieve(self, query, top_k: int = 3, **kwargs):
        query_terms = set(str(query).lower().split())
        scored = []
        for skill in self.skill_bank.skills:
            text = " ".join([skill.name, skill.description, " ".join(skill.tags), " ".join(skill.steps)]).lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [skill for _, skill in scored[:top_k]]
