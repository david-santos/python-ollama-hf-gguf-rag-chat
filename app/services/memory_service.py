"""Conversation memory service.

Equivalent to Spring AI's MessageChatMemoryAdvisor with per-conversation memory.
"""

from collections import defaultdict

import structlog
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

logger = structlog.get_logger()


class ConversationMemoryManager:
    """Manages conversation memory per conversation ID.

    Similar to Spring AI's MessageWindowChatMemory, this maintains separate
    conversation histories for different conversation IDs.
    """

    def __init__(self, max_messages: int = 20):
        """Initialize the memory manager.

        Args:
            max_messages: Maximum number of messages to retain per conversation
        """
        self._memories: dict[str, BaseChatMessageHistory] = defaultdict(ChatMessageHistory)
        self.max_messages = max_messages

    def get_memory(self, conversation_id: str) -> BaseChatMessageHistory:
        """Get or create memory for a conversation.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            Chat message history for the conversation
        """
        return self._memories[conversation_id]

    def add_exchange(
        self, conversation_id: str, user_message: str, ai_message: str
    ) -> None:
        """Add a user/AI exchange to conversation memory.

        Args:
            conversation_id: Unique identifier for the conversation
            user_message: The user's question
            ai_message: The AI's response
        """
        memory = self._memories[conversation_id]
        memory.add_user_message(user_message)
        memory.add_ai_message(ai_message)

        # Trim to max messages (keep most recent)
        if len(memory.messages) > self.max_messages:
            memory.messages = memory.messages[-self.max_messages :]

        logger.debug(
            "Updated conversation memory",
            conversation_id=conversation_id,
            message_count=len(memory.messages),
        )

    def clear_memory(self, conversation_id: str) -> None:
        """Clear memory for a conversation.

        Args:
            conversation_id: Unique identifier for the conversation
        """
        if conversation_id in self._memories:
            self._memories[conversation_id].clear()
            logger.info("Cleared conversation memory", conversation_id=conversation_id)

    def get_conversation_ids(self) -> list[str]:
        """Get all active conversation IDs.

        Returns:
            List of conversation IDs with active memory
        """
        return list(self._memories.keys())
