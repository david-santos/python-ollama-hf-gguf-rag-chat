# Python Ollama HF GGUF RAG Chat

A RAG (Retrieval Augmented Generation) chat application using Ollama and PGVector. This is a Python port of the [Spring AI example](../spring-ai/spring-ai-ollama-hf-gguf-rag-chat).

## Features

- **PDF Document Ingestion**: Automatically loads and processes PDF documents on startup
- **Vector Search**: Uses PGVector for semantic similarity search
- **Conversation Memory**: Maintains separate conversation contexts per session
- **Local LLMs**: Uses Ollama with Qwen3 models (chat + embeddings)
- **REST API**: FastAPI with automatic OpenAPI documentation

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Ollama](https://ollama.ai/) installed and running
- PostgreSQL with pgvector extension (or use Docker Compose)

## Quick Start

### 1. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

### 2. Install Dependencies

If you don't have uv installed:
```bash
brew install uv
```

Then install project dependencies:
```bash
uv sync
```

### 3. Pull Ollama Models

The application uses these models from HuggingFace via Ollama:

```bash
ollama pull hf.co/Qwen/Qwen3-8B-GGUF
ollama pull hf.co/Qwen/Qwen3-Embedding-8B-GGUF
```

### 4. Configure Environment (Optional)

Copy the example environment file and customize if needed:

```bash
cp .env.example .env
```

### 5. Run the Application

```bash
uv run uvicorn app.main:app --reload
```

The application will:
1. Load the PDF document from `resources/2025-nfl-rulebook-final.pdf`
2. Split it into chunks and compute embeddings
3. Store embeddings in PGVector
4. Start the API server on http://localhost:8000

To stop the application, press `Ctrl+C` in the terminal where uvicorn is running.

## API Usage

### Ask a Question

```bash
curl localhost:8000/ask -H "Content-type: application/json" -d '{"question": "What is roughing the passer?"}'
```

### Follow-up Question (Uses Conversation Memory)

```bash
curl localhost:8000/ask -H "Content-type: application/json" -d '{"question": "What is the penalty for that?"}'
```

### Separate Conversation Context

Use the `X_CONV_ID` header to maintain separate conversations:

```bash
curl localhost:8000/ask -H "Content-type: application/json" -H "X_CONV_ID: user1" -d '{"question": "What is a touchdown?"}'
```

### Health Check

```bash
curl localhost:8000/health
```

### OpenAPI Documentation

Visit http://localhost:8000/docs for interactive API documentation.

## Configuration

Configuration is done via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_CHAT_MODEL` | `hf.co/Qwen/Qwen3-8B-GGUF` | Chat model |
| `OLLAMA_EMBEDDING_MODEL` | `hf.co/Qwen/Qwen3-Embedding-8B-GGUF` | Embedding model |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `postgres` | Database name |
| `DOCUMENT_PATH` | `resources/2025-nfl-rulebook-final.pdf` | Document to ingest |

## Project Structure

```
python-ollama-hf-gguf-rag-chat/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Pydantic Settings
│   ├── dependencies.py          # Dependency injection
│   ├── api/routes/ask.py        # POST /ask endpoint
│   ├── models/schemas.py        # Request/response models
│   ├── services/
│   │   ├── chat_service.py      # RAG chat logic
│   │   ├── memory_service.py    # Conversation memory
│   │   └── vector_store_service.py
│   ├── etl/document_loader.py   # PDF ingestion pipeline
│   └── core/
│       ├── logging.py           # Structured logging
│       └── lifespan.py          # Startup ETL
├── resources/                   # PDF documents
├── docker-compose.yml           # PostgreSQL + pgvector
└── pyproject.toml              # Dependencies
```

## Technology Stack

| Component | Library | Spring AI Equivalent |
|-----------|---------|---------------------|
| Web Framework | FastAPI | Spring Boot Web |
| LLM Framework | LangChain | Spring AI |
| PDF Parsing | PyMuPDF | Apache Tika |
| Vector Store | langchain-postgres | PGVector |
| Configuration | pydantic-settings | application.properties |

## License

MIT
