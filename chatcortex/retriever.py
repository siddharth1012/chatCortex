#Vector retrieval

from typing import List


class Retriever:
    def __init__(self):
        self._documents: List[str] = []
    
    def index(self, documents: List[str]):
        self._documents.extend(documents)
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        # v0.1: Simple retrievel
        return self._documents[:k]