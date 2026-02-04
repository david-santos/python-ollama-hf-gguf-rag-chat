"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "python-ollama-hf-gguf-rag-chat"
    debug: bool = False

    # Ollama Chat Model
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "hf.co/Qwen/Qwen3-8B-GGUF"
    ollama_chat_temperature: float = 0.6
    ollama_chat_top_p: float = 0.95
    ollama_chat_top_k: int = 20
    ollama_chat_presence_penalty: float = 1.5
    ollama_chat_think: bool = True  # Enable thinking/reasoning mode (Qwen3 feature)

    # Ollama Embedding Model
    ollama_embedding_model: str = "hf.co/Qwen/Qwen3-Embedding-8B-GGUF"
    ollama_embedding_dimensions: int = 4096

    # PostgreSQL / PGVector
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "postgres"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # ETL
    document_path: str = "resources/2025-nfl-rulebook-final.pdf"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # RAG
    retrieval_k: int = 4

    @property
    def postgres_dsn(self) -> str:
        """PostgreSQL connection string."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
