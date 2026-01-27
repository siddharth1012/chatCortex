#Chat session and memoy

from typing import List, Dict
from uuid import uuid4


class ChatSession:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid4())
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})
    
    def add_assistant_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})
    
    def clear(self):
        self.history.clear()