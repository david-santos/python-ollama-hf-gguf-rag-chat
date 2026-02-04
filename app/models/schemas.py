"""Request and response models."""

from pydantic import BaseModel, Field


class Question(BaseModel):
    """Request model for asking a question."""

    question: str = Field(..., min_length=1, description="The question to ask")


class Answer(BaseModel):
    """Response model containing the answer."""

    answer: str = Field(..., description="The generated answer")
