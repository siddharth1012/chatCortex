#Configuration handling

from dataclasses import dataclass
from typing import Optional

@dataclass
class ChatCortexConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db: str = "faiss"
    chunk_size: int = 500
    chunk_overlap: int = 100
    llm_model: str = "local"
    persist_path: Optional[str] = None
