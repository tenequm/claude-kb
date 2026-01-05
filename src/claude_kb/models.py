"""Pydantic models for claude-kb."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_serializer


class KBModel(BaseModel):
    """Base model with null exclusion for clean output."""

    model_config = ConfigDict(extra="ignore")

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        """Exclude None values from serialization."""
        data = handler(self)
        return {k: v for k, v in data.items() if v is not None}


class Message(KBModel):
    """Single message from search or retrieval."""

    id: str
    role: str
    content: str
    timestamp: datetime
    project: str
    conversation_id: str | None = None
    parent_id: str | None = None
    score: float | None = Field(None, description="Relevance score (0.0-1.0)")


class SearchResult(KBModel):
    """Result from search operation."""

    query: str
    collection: str
    count: int
    results: list[Message]


class ConversationSummary(KBModel):
    """Summary of a conversation matching search criteria."""

    conversation_id: str
    project: str
    first_timestamp: datetime
    last_timestamp: datetime
    message_count: int
    preview: str = Field(description="Preview of first matching message content")
    best_score: float = Field(description="Highest relevance score in conversation")


class ConversationSearchResult(KBModel):
    """Result from search operation with conversation grouping."""

    query: str
    collection: str
    count: int
    conversations: list[ConversationSummary]


class GetResult(KBModel):
    """Result from get operation."""

    message: Message
    thread: list[Message] | None = Field(
        None, description="Surrounding messages if context_depth > 0"
    )


class ProjectStats(KBModel):
    """Per-project statistics."""

    project: str
    sessions: int
    messages: int


class StatusResult(KBModel):
    """Database status information."""

    qdrant_url: str
    embedding_model: str
    collections: dict[str, int]
    projects: list[ProjectStats] | None = None


class ErrorResult(KBModel):
    """Error response for MCP."""

    error: str
    suggestion: str | None = None
