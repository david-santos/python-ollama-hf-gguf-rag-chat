"""Vector store service using PGVector."""

import structlog
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from app.config import Settings

logger = structlog.get_logger()


def create_embeddings(settings: Settings) -> OllamaEmbeddings:
    """Create Ollama embeddings model."""
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )


def create_vector_store(settings: Settings) -> PGVector:
    """Create PGVector store with Ollama embeddings.

    Note: Using exact search since HNSW index doesn't support >2000 dimensions
    and we're using 4096-dimension embeddings.
    """
    logger.info(
        "Creating vector store",
        embedding_model=settings.ollama_embedding_model,
        postgres_host=settings.postgres_host,
    )

    embeddings = create_embeddings(settings)

    return PGVector(
        embeddings=embeddings,
        collection_name="document_chunks",
        connection=settings.postgres_dsn,
        use_jsonb=True,
    )
