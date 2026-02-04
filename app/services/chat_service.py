"""RAG chat service.

Combines the functionality of Spring AI's:
- QuestionAnswerAdvisor (RAG retrieval and augmentation)
- MessageChatMemoryAdvisor (conversation memory)
- SimpleLoggerAdvisor (logging)
"""

import structlog
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_postgres import PGVector

from app.services.memory_service import ConversationMemoryManager

logger = structlog.get_logger()


# RAG prompt template - similar to QuestionAnswerAdvisor's default prompt
RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer questions based on the provided context from the documents. If the context doesn't contain relevant information to answer the question, say so clearly.

Context from documents:
{context}
"""


class RAGChatService:
    """RAG-augmented chat service with conversation memory.

    This service combines:
    1. Vector store retrieval for RAG (QuestionAnswerAdvisor equivalent)
    2. Conversation memory for multi-turn conversations (MessageChatMemoryAdvisor equivalent)
    3. Structured logging (SimpleLoggerAdvisor equivalent)
    """

    def __init__(
        self,
        llm: ChatOllama,
        vector_store: PGVector,
        memory_manager: ConversationMemoryManager,
        k_retrieval: int = 4,
    ):
        """Initialize the RAG chat service.

        Args:
            llm: The Ollama chat model
            vector_store: PGVector store for document retrieval
            memory_manager: Conversation memory manager
            k_retrieval: Number of documents to retrieve for context
        """
        self.llm = llm
        self.vector_store = vector_store
        self.memory_manager = memory_manager
        self.retriever = vector_store.as_retriever(search_kwargs={"k": k_retrieval})

        # Build the prompt template with system prompt, chat history, and user question
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        # Build the chain: prompt -> llm -> parse output to string
        self.chain = self.prompt | self.llm | StrOutputParser()

    def ask(self, question: str, conversation_id: str) -> str:
        """Process a question with RAG and conversation memory.

        Args:
            question: The user's question
            conversation_id: Unique identifier for the conversation

        Returns:
            The generated answer
        """
        logger.info(
            "Processing question",
            question=question[:100],
            conversation_id=conversation_id,
        )

        # Get conversation history
        memory = self.memory_manager.get_memory(conversation_id)
        chat_history = memory.messages

        # Retrieve relevant documents from vector store
        docs = self.retriever.invoke(question)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        logger.debug(
            "Retrieved documents",
            doc_count=len(docs),
            context_length=len(context),
        )

        # Generate response using the chain
        response = self.chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question,
        })

        # Update conversation memory with this exchange
        self.memory_manager.add_exchange(conversation_id, question, response)

        logger.info(
            "Generated response",
            response_length=len(response),
            conversation_id=conversation_id,
        )

        return response
