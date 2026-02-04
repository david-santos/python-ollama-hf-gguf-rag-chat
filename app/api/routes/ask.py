"""Ask endpoint for RAG chat.

Equivalent to Spring AI's AskController.
"""

from fastapi import APIRouter, Depends, Header

from app.dependencies import get_chat_service
from app.models.schemas import Answer, Question
from app.services.chat_service import RAGChatService

router = APIRouter()


@router.post("/ask", response_model=Answer)
def ask(
    question: Question,
    x_conv_id: str = Header(default="defaultConversation", alias="X_CONV_ID"),
    chat_service: RAGChatService = Depends(get_chat_service),
) -> Answer:
    """Ask a question about the ingested documents.

    Uses RAG to retrieve relevant context from the vector store and
    conversation memory for multi-turn conversations.

    The X_CONV_ID header can be used to maintain separate conversation
    contexts for different users or sessions.

    Args:
        question: The question to ask
        x_conv_id: Conversation ID for memory isolation (default: "defaultConversation")
        chat_service: Injected RAG chat service

    Returns:
        The generated answer based on document context
    """
    response = chat_service.ask(question.question, x_conv_id)
    return Answer(answer=response)
