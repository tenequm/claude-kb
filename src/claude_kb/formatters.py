"""CLI formatters for string output."""

import json
import logging

from .models import GetResult, SearchResult, StatusResult

logger = logging.getLogger(__name__)

MAX_CONTENT_PREVIEW_LENGTH = 1500


def clean_content(content: str | list | dict) -> str | list | dict:
    """
    Remove signature fields from content to reduce noise and token usage.

    Signatures are cryptographic verification data that:
    - Consume hundreds of base64 characters
    - Provide zero semantic value for search/retrieval
    - Make results harder to read

    Args:
        content: Content string, list, or dict

    Returns:
        Cleaned content with signatures removed
    """
    if isinstance(content, str):
        # Try to parse as JSON if it looks like JSON
        if content.strip().startswith("[") or content.strip().startswith("{"):
            try:
                parsed = json.loads(content)
                cleaned = clean_content(parsed)
                return (
                    json.dumps(cleaned, indent=2) if isinstance(cleaned, dict | list) else content
                )
            except (json.JSONDecodeError, ValueError):
                return content
        return content
    elif isinstance(content, list):
        return [clean_content(item) for item in content]
    elif isinstance(content, dict):
        # Remove 'signature' key recursively
        return {k: clean_content(v) for k, v in content.items() if k != "signature"}
    return content


def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken cl100k_base encoding (GPT-4/Claude compatible).

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens
    """
    import tiktoken

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Failed to count tokens: {e}")
        # Fallback: rough estimate (4 chars per token)
        return len(text) // 4


def format_search(result: SearchResult, show_tokens: bool = False) -> str:
    """
    Format SearchResult for CLI output.

    Args:
        result: SearchResult from search service
        show_tokens: Whether to display token counts

    Returns:
        Formatted string
    """
    if result.count == 0:
        return "No results found."

    lines = []
    total_tokens = 0

    for msg in result.results:
        score_str = f" (score: {msg.score:.2f})" if msg.score is not None else ""
        lines.append(f"=== {msg.id}{score_str} ===")
        lines.append(f"Role: {msg.role}")
        lines.append(f"Time: {msg.timestamp}")
        lines.append(f"Project: {msg.project}")

        content = clean_content(msg.content)
        content_str = str(content) if not isinstance(content, str) else content

        if show_tokens:
            tokens = count_tokens(content_str)
            total_tokens += tokens
            lines.append(f"Tokens: {tokens:,}")

        lines.append("")

        if len(content_str) > MAX_CONTENT_PREVIEW_LENGTH:
            lines.append(content_str[:MAX_CONTENT_PREVIEW_LENGTH] + "...")
        else:
            lines.append(content_str)

        lines.append("---")

    if show_tokens and total_tokens > 0:
        lines.append(f"\n=== Total: {result.count} results, {total_tokens:,} tokens ===")

    return "\n".join(lines)


def format_get(result: GetResult, context_window: int | None = None) -> str:
    """
    Format GetResult for CLI output (single message).

    Args:
        result: GetResult from get service
        context_window: Optional truncation limit for content

    Returns:
        Formatted string
    """
    msg = result.message
    lines = [f"=== {msg.id} ==="]

    lines.append(f"Role: {msg.role}")
    lines.append(f"Time: {msg.timestamp}")
    lines.append(f"Conversation: {msg.conversation_id or 'N/A'}")
    lines.append(f"Project: {msg.project}")
    lines.append("")

    # Clean content
    content = clean_content(msg.content)
    content_str = str(content) if not isinstance(content, str) else content

    # Apply context window truncation
    if context_window and context_window > 0 and len(content_str) > context_window:
        content_str = (
            content_str[:context_window]
            + f"\n\n[... +{len(content_str) - context_window} chars truncated. "
            f"Use --context-window 0 for full content]"
        )

    lines.append(content_str)
    lines.append("")

    # Related messages
    if msg.parent_id:
        lines.append("Related:")
        lines.append(f"  Parent: {msg.parent_id}")

    return "\n".join(lines)


def format_thread(result: GetResult) -> str:
    """
    Format GetResult with thread context for CLI output.

    Args:
        result: GetResult from get service (with thread populated)

    Returns:
        Formatted string
    """
    if not result.thread:
        # Fall back to single message format
        return format_get(result)

    target_id = result.message.id
    messages = result.thread
    depth = len(messages) // 2  # Approximate

    lines = [f"=== Thread Context (±{depth} messages) ===", ""]

    for i, msg in enumerate(messages):
        is_target = msg.id == target_id
        marker = ">>> TARGET <<<" if is_target else f"[{i + 1}/{len(messages)}]"

        lines.append(f"--- {marker} ---")
        lines.append(f"ID: {msg.id}")
        lines.append(f"Role: {msg.role}")
        lines.append(f"Time: {msg.timestamp}")

        if is_target:
            lines.append(f"Project: {msg.project}")

        lines.append("")

        # Clean and display content
        content = clean_content(msg.content)
        content_str = str(content) if not isinstance(content, str) else content

        # For target message, show more content
        max_length = 2000 if is_target else 800
        if len(content_str) > max_length:
            lines.append(
                content_str[:max_length] + f"... [{len(content_str) - max_length} chars more]"
            )
        else:
            lines.append(content_str)

        lines.append("")

    # Summary footer
    lines.append("---")
    lines.append(f"Conversation: {result.message.conversation_id or 'unknown'}")
    lines.append(f"Total messages in thread: {len(messages)}")

    return "\n".join(lines)


def format_status(result: StatusResult) -> str:
    """
    Format StatusResult for CLI output.

    Args:
        result: StatusResult from status service

    Returns:
        Formatted string
    """
    lines = [
        "[bold cyan]=== Claude KB Status ===[/]",
        "",
        f"Qdrant: {result.qdrant_url} [green]✓[/]",
        f"Embedding: {result.embedding_model}",
        "",
        "[bold]Collections:[/]",
    ]

    for name, count in result.collections.items():
        # Format size (rough estimate: ~1KB per point with 768-dim vector)
        size_mb = (count * 1) / 1024
        lines.append(f"  {name:20} {count:6,} points  ({size_mb:.1f} MB)")

    if result.projects:
        lines.append("")
        lines.append("[bold]Projects:[/]")

        # Format as table-like output
        lines.append(f"  {'Project':<30} {'Sessions':>10} {'Messages':>10}")
        lines.append(f"  {'-' * 30} {'-' * 10} {'-' * 10}")

        total_sessions = 0
        total_messages = 0

        for proj in result.projects:
            # Shorten project path for display
            project_name = proj.project.split("/")[-1] if "/" in proj.project else proj.project
            if len(project_name) > 28:
                project_name = project_name[:25] + "..."
            lines.append(f"  {project_name:<30} {proj.sessions:>10,} {proj.messages:>10,}")
            total_sessions += proj.sessions
            total_messages += proj.messages

        lines.append(f"  {'-' * 30} {'-' * 10} {'-' * 10}")
        lines.append(f"  {'TOTAL':<30} {total_sessions:>10,} {total_messages:>10,}")

    return "\n".join(lines)
