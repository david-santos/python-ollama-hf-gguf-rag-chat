"""FastAPI dependency injection.

Equivalent to Spring's @Bean configuration in Config.java.
"""

from functools import lru_cache

from langchain_ollama import ChatOllama

from app.config import Settings
from app.services.chat_service import RAGChatService
from app.services.memory_service import ConversationMemoryManager
from app.services.vector_store_service import create_vector_store


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


@lru_cache
def get_llm() -> ChatOllama:
    """Get cached Ollama chat model.

    Equivalent to the ChatClient configuration in Spring AI's Config.java.
    """
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=settings.ollama_chat_temperature,
        top_p=settings.ollama_chat_top_p,
        top_k=settings.ollama_chat_top_k,
        # Note: presence_penalty is passed as repeat_penalty in Ollama
        repeat_penalty=settings.ollama_chat_presence_penalty,
        # Enable thinking/reasoning mode for Qwen3 models
        think=settings.ollama_chat_think,
    )


@lru_cache
def get_vector_store():
    """Get cached vector store instance."""
    settings = get_settings()
    return create_vector_store(settings)


@lru_cache
def get_memory_manager() -> ConversationMemoryManager:
    """Get cached conversation memory manager.

    Equivalent to MessageWindowChatMemory in Spring AI.
    """
    return ConversationMemoryManager(max_messages=20)


def get_chat_service() -> RAGChatService:
    """Get RAG chat service with all dependencies.

    This combines the functionality of:
    - QuestionAnswerAdvisor (RAG)
    - MessageChatMemoryAdvisor (conversation memory)
    - SimpleLoggerAdvisor (logging)
    """
    settings = get_settings()
    return RAGChatService(
        llm=get_llm(),
        vector_store=get_vector_store(),
        memory_manager=get_memory_manager(),
        k_retrieval=settings.retrieval_k,
    )
