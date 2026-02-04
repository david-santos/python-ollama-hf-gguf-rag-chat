"""Application lifespan management for startup/shutdown events."""

from pathlib import Path

import structlog

from app.config import Settings
from app.etl.document_loader import DocumentETL
from app.services.vector_store_service import create_vector_store

logger = structlog.get_logger()


def run_startup_etl(settings: Settings) -> None:
    """Run ETL pipeline on application startup.

    Equivalent to Spring AI's ApplicationRunner that loads documents
    into the vector store when the application starts.

    Args:
        settings: Application settings
    """
    document_path = Path(settings.document_path)

    if not document_path.exists():
        logger.warning(
            "Document not found, skipping ETL",
            document_path=str(document_path),
        )
        return

    logger.info("Starting ETL pipeline", document_path=str(document_path))

    # Create vector store connection
    vector_store = create_vector_store(settings)

    # Run ETL pipeline
    etl = DocumentETL(
        document_path=document_path,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunk_count = etl.load_documents()
    logger.info("ETL pipeline completed", chunks_loaded=chunk_count)
