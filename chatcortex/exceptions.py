#Custom exceptions

class ChatCortexError(Exception):
    """Base exception for ChatCortex"""

class ConfigurationError(ChatCortexError):
    """Raised when configuration is invalid"""

class IngestionError(ChatCortexError):
    """Raised when document ingestion fails"""

class RetrievalError(ChatCortexError):
    """Raised when retrieval fails"""

class SessionError(ChatCortexError):
    """Raised for session-related issues"""

