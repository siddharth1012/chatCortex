# ChatCortex

ChatCortex is a **privacy-first, local document chatbot engine** designed to be embedded into any application.  
It enables teams to build document-aware conversational systems **without relying on cloud services**, while keeping all data within their own infrastructure.

ChatCortex is **library-first**, **API-agnostic**, and designed for production and enterprise use cases.

---

## Overview

Most document-based chat systems are tightly coupled to user interfaces, depend on external cloud services, or are difficult to integrate into existing platforms.

ChatCortex focuses exclusively on the **core infrastructure layer**, allowing developers to integrate document chat capabilities into their own systems with minimal overhead.

---

## Core Principles

- **Privacy-first**: All ingestion, retrieval, and inference occur locally  
- **Library-first design**: Use directly as a Python SDK or wrap with custom APIs  
- **Frontend-agnostic**: Compatible with any UI or client application  
- **Backend-oriented**: Built as infrastructure, not a demo application  
- **Simple and stable API**: Small public surface area intended for long-term stability  

---

## Use Cases

ChatCortex can be used to build:

- Internal enterprise document chatbots  
- Knowledge base assistants  
- Policy, HR, and legal document Q&A systems  
- On-premise or air-gapped AI assistants  
- Custom chat APIs for proprietary frontends  

---

## Quick Start

```bash
pip install chatcortex

from chatcortex import ChatCortexEngine

engine = ChatCortexEngine()

# Ingest documents (v0.1 supports .txt files)
engine.ingest("./docs")

# Ask a question
response = engine.chat("What is covered in these documents?")
print(response)
```
--- 

## Public API (v0.1)

```
ChatCortexEngine(
    config: ChatCortexConfig | None = None
)

engine.ingest(source: str)
engine.chat(message: str, session_id: Optional[str] = None) -> str
engine.reset_session(session_id: str)
```
---

## Configuration

```
from chatcortex.config import ChatCortexConfig
from chatcortex import ChatCortexEngine

config = ChatCortexConfig(
    chunk_size=500,
    chunk_overlap=100
)

engine = ChatCortexEngine(config)
```

---

## Design Philosophy

ChatCortex enforces a clear separation of responsibilities:

| Layer       | Responsibility                                    |
| ----------- | ------------------------------------------------- |
| Core Engine | Document ingestion, retrieval, session management |
| API Layer   | Optional adapters (REST, gRPC, etc.)              |
| Frontend    | Fully owned by the integrator                     |


You are free to:

Use ChatCortex directly as a Python library

Wrap it with your own API layer

Integrate it into existing systems

Ignore any optional adapters

---

## Roadmap
v0.1 -> Core engine -> Local document ingestion -> Session handling

## Planned

PDF ingestion -> Embeddings and vector database support (FAISS, Chroma) -> Streaming responses -> External connectors (S3, SharePoint) -> Optional FastAPI service wrapper

---

## Development Setup
```
git clone https://github.com/yourusername/chatcortex
cd chatcortex
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## License

MIT License

You are free to use, modify, and integrate ChatCortex into commercial and internal products.

---

## Project Goals

ChatCortex is built as infrastructure, not a demo application.

It is intended for teams that require a reusable, local-first document chatbot engine
without repeatedly rebuilding the same backend components.
