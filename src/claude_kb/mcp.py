"""MCP Server for claude-kb."""

from mcp.server.fastmcp import FastMCP

from .models import ErrorResult, GetResult, SearchResult
from .search import SearchService

mcp = FastMCP(
    "kb",
    instructions="""Knowledge base for Claude Code conversation history.

Tools:
- search: Semantic search across conversations (hybrid dense+sparse vectors)
- get: Retrieve message by ID, optionally with thread context

Workflow: search → get(id) for full content or get(id, context_depth=N) for thread

Filtering options for search:
- project: Partial match on project path
- from_date/to_date: ISO format (YYYY-MM-DD)
- role: "user" or "assistant"
- min_score: Relevance threshold (default 0.5)
""",
)

# Lazy-initialized service
_service: SearchService | None = None


def get_service() -> SearchService:
    """Get or create the search service singleton."""
    global _service
    if _service is None:
        _service = SearchService()
    return _service


@mcp.tool()
def search(
    query: str,
    limit: int = 10,
    project: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    role: str | None = None,
    min_score: float = 0.5,
    boost_recent: bool = True,
) -> SearchResult | ErrorResult:
    """Semantic search across Claude Code conversations.

    Uses hybrid search (dense + sparse vectors) when available.
    Recent messages are boosted by default.

    Args:
        query: Search query text
        limit: Maximum number of results (default 10)
        project: Filter by project path (partial match)
        from_date: Filter from date (ISO format: YYYY-MM-DD)
        to_date: Filter to date (ISO format: YYYY-MM-DD)
        role: Filter by role ("user" or "assistant")
        min_score: Minimum relevance score threshold (0.0-1.0, default 0.5)
        boost_recent: Boost recent messages in ranking (default True)

    Returns:
        SearchResult with matching messages, or ErrorResult on failure.
    """
    return get_service().search(
        query=query,
        limit=limit,
        project=project,
        from_date=from_date,
        to_date=to_date,
        role=role,
        min_score=min_score,
        boost_recent=boost_recent,
    )


@mcp.tool()
def get(
    message_id: str,
    context_depth: int = 0,
) -> GetResult | ErrorResult:
    """Retrieve message by ID with optional thread context.

    Args:
        message_id: The message ID to retrieve
        context_depth: If > 0, include ±N surrounding messages from conversation

    Returns:
        GetResult with message and optional thread context.
    """
    return get_service().get(message_id, context_depth=context_depth)


@mcp.resource("schema://kb")
def get_schema() -> str:
    """Search fields and filter documentation."""
    return """# KB Search Schema

## Message Fields
- id: Unique message identifier
- role: "user" or "assistant"
- content: Message text
- timestamp: ISO datetime
- project: Project path where conversation occurred
- conversation_id: Groups messages in same session

## Search Filters
- project: Partial match (e.g., "claude-kb" matches "/Users/x/Projects/claude-kb")
- from_date / to_date: ISO format YYYY-MM-DD
- role: Exact match "user" or "assistant"
- min_score: 0.0-1.0 relevance threshold

## Examples
- Find error discussions: search("error handling", project="my-app")
- Recent assistant responses: search("implementation", role="assistant", from_date="2024-12-01")
- Get message with context: get("message-id", context_depth=3)
"""


@mcp.resource("stats://kb")
def get_stats() -> str:
    """Database statistics."""
    result = get_service().status(include_projects=False)
    if isinstance(result, ErrorResult):
        return f"Error: {result.error}"

    lines = [
        "# KB Statistics",
        f"Qdrant: {result.qdrant_url}",
        f"Embedding: {result.embedding_model}",
        "",
        "## Collections",
    ]
    for name, count in result.collections.items():
        lines.append(f"- {name}: {count:,} messages")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
