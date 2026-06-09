from typing import Any, Dict, List

from .retriever import KeywordMemoryRetriever


class ExperienceMemory:
    """Long-term memory for successful reusable experiences."""

    def __init__(self, records=None, retriever=None):
        self.records: List[Dict[str, Any]] = list(records or [])
        self.retriever = retriever or KeywordMemoryRetriever(self.records)

    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        self.retriever.add(record)

    def search(self, query, top_k: int = 3, **kwargs):
        return self.retriever.retrieve(query=query, top_k=top_k, **kwargs)
