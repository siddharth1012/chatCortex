#Document ingestion

from pathlib import Path
from typing import List

from .exceptions import IngestionError


class DocumentIngestor:
    def load_from_path(self, path: str) -> List[str]:
        p = Path(path)

        if not p.exists():
            raise IngestionError(f"Path does exist: {path}")
        
        documents: List[str] = []

        if p.is_file() and p.suffix == ".txt":
            documents.append(p.read_text(encoding="utf-8"))
        elif p.is_dir():
            for file in p.rglob("*.txt"):
                documents.append(file.read_text(encoding="utf-8"))
        else:
            raise IngestionError("Only .txt files are supported in v0.1")
    
        return documents