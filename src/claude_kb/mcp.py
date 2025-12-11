"""
MCP Server for claude-kb knowledge base.

Provides semantic search across Claude Code conversation history
with hybrid dense+sparse vector search.

Run with:
    uv run kb mcp                    # stdio (default)
    uv run kb mcp --transport http   # Streamable HTTP

Add to Claude Code:
    claude mcp add kb -- uv run kb mcp
"""

from __future__ import annotations

from typing import Literal

import click
from mcp.server.fastmcp import FastMCP
from mcp.types import Icon, ToolAnnotations

from .models import ErrorResult, GetResult, SearchResult
from .search import SearchService

# ==================== Icons (SVG Data URIs) ====================

# Book/database icon for KB server
ICON_KB = Icon(
    src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23666' stroke-width='2'%3E%3Cpath d='M4 19.5A2.5 2.5 0 0 1 6.5 17H20'/%3E%3Cpath d='M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z'/%3E%3C/svg%3E",
    mimeType="image/svg+xml",
)

# Magnifying glass for search
ICON_SEARCH = Icon(
    src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23666' stroke-width='2'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E",
    mimeType="image/svg+xml",
)

# Document icon for get
ICON_GET = Icon(
    src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23666' stroke-width='2'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpath d='M14 2v6h6'/%3E%3Cpath d='M16 13H8'/%3E%3Cpath d='M16 17H8'/%3E%3Cpath d='M10 9H8'/%3E%3C/svg%3E",
    mimeType="image/svg+xml",
)

# Info icon for schema resource
ICON_SCHEMA = Icon(
    src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23666' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M12 16v-4'/%3E%3Cpath d='M12 8h.01'/%3E%3C/svg%3E",
    mimeType="image/svg+xml",
)


# ==================== Tool Annotations ====================

# Search tool: read-only, queries local Qdrant
SEARCH_ANNOTATIONS = ToolAnnotations(
    title="Semantic Search",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,  # Local KB, not external API
)

# Get tool: read-only, retrieves from local Qdrant
GET_ANNOTATIONS = ToolAnnotations(
    title="Get Message",
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


# ==================== Server Factory ====================

SERVER_INSTRUCTIONS = """Knowledge base for Claude Code conversation history.

Tools:
- search: Semantic search across conversations (hybrid dense+sparse vectors)
- get: Retrieve message by ID, optionally with thread context

Workflow: search → get(id) for full content or get(id, context_depth=N) for thread

Filtering options for search:
- project: Partial match on project path
- from_date/to_date: ISO format (YYYY-MM-DD)
- role: "user" or "assistant"
- min_score: Relevance threshold (default 0.5)
"""


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    json_response: bool = True,
    stateless: bool = True,
) -> FastMCP:
    """Create and configure the KB MCP server."""
    return FastMCP(
        "kb",
        instructions=SERVER_INSTRUCTIONS,
        icons=[ICON_KB],
        host=host,
        port=port,
        json_response=json_response,
        stateless_http=stateless,
    )


# Default server instance for stdio transport
mcp = create_server()


# ==================== Service ====================

# Lazy-initialized service
_service: SearchService | None = None


def get_service() -> SearchService:
    """Get or create the search service singleton."""
    global _service
    if _service is None:
        _service = SearchService()
    return _service


# ==================== Tools ====================


@mcp.tool(
    annotations=SEARCH_ANNOTATIONS,
    icons=[ICON_SEARCH],
)
def search(
    query: str,
    limit: int = 10,
    project: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    role: Literal["user", "assistant"] | None = None,
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


@mcp.tool(
    annotations=GET_ANNOTATIONS,
    icons=[ICON_GET],
)
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


# ==================== Resources ====================


@mcp.resource("schema://kb", icons=[ICON_SCHEMA])
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


@mcp.resource("stats://kb", icons=[ICON_KB])
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


# ==================== CLI ====================


def _register_tools(server: FastMCP):
    """Register all tools on a server instance."""
    server.add_tool(search, annotations=SEARCH_ANNOTATIONS, icons=[ICON_SEARCH])
    server.add_tool(get, annotations=GET_ANNOTATIONS, icons=[ICON_GET])


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    help="Transport: stdio (default) or http (Streamable HTTP)",
)
@click.option("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
@click.option("--port", default=8000, type=int, help="HTTP port (default: 8000)")
@click.option(
    "--json-response/--sse-response",
    default=True,
    help="Use JSON responses (default) or SSE streaming",
)
def main(transport: str, host: str, port: int, json_response: bool):
    """
    Run the KB MCP server.

    Examples:

        # stdio transport (for Claude Code)
        uv run kb mcp

        # Streamable HTTP transport
        uv run kb mcp --transport http

        # HTTP with custom port
        uv run kb mcp --transport http --port 3000
    """
    global mcp

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # Create new server instance with HTTP settings
        mcp = create_server(
            host=host,
            port=port,
            json_response=json_response,
            stateless=True,
        )
        # Re-register tools on new instance
        _register_tools(mcp)
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
