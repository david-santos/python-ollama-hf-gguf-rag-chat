"""FastAPI application entry point.

Equivalent to Spring AI's OllamaHfGgufRagChatApplication.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api.routes import ask
from app.config import Settings
from app.core.lifespan import run_startup_etl
from app.core.logging import setup_logging

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Runs ETL on startup (equivalent to Spring AI's ApplicationRunner).
    """
    settings = Settings()
    setup_logging(debug=settings.debug)

    logger.info("Starting application", app_name=settings.app_name)

    # Run ETL pipeline on startup
    run_startup_etl(settings)

    yield

    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI application
app = FastAPI(
    title="Python Ollama HF GGUF RAG Chat",
    description="RAG chat application using Ollama and PGVector - Python port of Spring AI example",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(ask.router)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}
