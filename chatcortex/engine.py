#Core Engine (Public API)

from typing import Dict, Optional

from .config import ChatCortexConfig
from .ingestion import DocumentIngestor
from .retriever import Retriever
from .session import ChatSession
from .exceptions import SessionError


class ChatCortexEngine:
    def __init__(self, config: ChatCortexConfig | None = None):
        self.config = config or ChatCortexConfig()
        self.ingestor = DocumentIngestor()
        self.retriever = Retriever()
        self.sessions: Dict[str, ChatSession] = {}
    
    def ingest(self, source: str):
        documents = self.ingestor.load_from_path(source)
        self.retriever.index(documents)
    
    def chat(self, message: str, session_id: Optional[str] = None) -> str:
        session = self._get_or_create_session(session_id)
        session.add_user_message(message)

        context = self.retriever.retrieve(message)
        response = self._generate_response(message, context)

        session.add_assistant_message(response)
        return response, session.session_id

    def reset_session(self, session_id: str):
        if session_id not in self.sessions:
            raise SessionError("Session not found")
        self.sessions[session_id].clear()
    
    def _get_or_create_session(self, session_id: Optional[str]) -> ChatSession:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        session = ChatSession(session_id)
        self.sessions[session.session_id] = session
        return session
    
    def _generate_response(self, message: str, context: list[str]) -> str:
        # v0.1 initial setup
        return f"Answer Based on {len(context)} document(s)."