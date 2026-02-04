"""ETL pipeline for document ingestion.

Equivalent to Spring AI's:
- TikaDocumentReader -> PyMuPDFLoader
- TokenTextSplitter -> RecursiveCharacterTextSplitter
- VectorStore.write() -> PGVector.add_documents()
"""

from pathlib import Path

import structlog
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger()


class DocumentETL:
    """ETL pipeline for loading documents into vector store."""

    def __init__(
        self,
        document_path: Path,
        vector_store: PGVector,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the ETL pipeline.

        Args:
            document_path: Path to the document to ingest
            vector_store: PGVector store to write embeddings to
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between chunks for context continuity
        """
        self.document_path = document_path
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_documents(self, batch_size: int = 10) -> int:
        """Load and ingest documents into vector store.

        Args:
            batch_size: Number of chunks to process per batch (for progress logging)

        Returns:
            Number of chunks loaded
        """
        logger.info("Starting document ETL", document=str(self.document_path))

        # Step 1: Read PDF (equivalent to TikaDocumentReader)
        loader = PyMuPDFLoader(str(self.document_path))
        documents = loader.load()
        logger.info("Loaded document", pages=len(documents))

        # Step 2: Split into chunks (equivalent to TokenTextSplitter)
        chunks = self.text_splitter.split_documents(documents)
        total_chunks = len(chunks)
        logger.info("Split into chunks", chunk_count=total_chunks)

        # Step 3: Write to vector store in batches (embeddings computed automatically)
        logger.info(
            "Starting embedding generation",
            total_chunks=total_chunks,
            batch_size=batch_size,
        )

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            logger.info(
                "Processing batch",
                batch=batch_num,
                total_batches=total_batches,
                chunks_in_batch=len(batch),
                progress=f"{min(i + batch_size, total_chunks)}/{total_chunks}",
            )

            self.vector_store.add_documents(batch)

        logger.info("Documents loaded into vector store", chunk_count=total_chunks)

        return total_chunks
