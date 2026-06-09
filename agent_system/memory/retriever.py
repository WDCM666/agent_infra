from typing import Any, Dict, List


class KeywordMemoryRetriever:
    """Simple lexical retriever; replace with embeddings for serious baselines."""

    def __init__(self, records=None):
        self.records = list(records or [])

    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def retrieve(self, query, top_k: int = 3, **kwargs) -> List[Dict[str, Any]]:
        query_terms = set(str(query).lower().split())
        scored = []
        for record in self.records:
            text = " ".join(str(v) for v in record.values()).lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]
