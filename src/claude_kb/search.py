"""Search service layer for claude-kb."""

import json
import logging
import math
import re
from datetime import UTC, datetime

from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue, Range

from .config import get_config
from .db import QdrantDB
from .models import (
    ConversationSearchResult,
    ConversationSummary,
    ErrorResult,
    GetResult,
    Message,
    ProjectStats,
    SearchResult,
    StatusResult,
)

logger = logging.getLogger(__name__)

RESTORE_ALLOWED_ROLES = {"user", "assistant"}
RESTORE_SKIP_BLOCK_TYPES = {"tool_use", "tool_result", "thinking"}
SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder\b[^>]*>.*?</system-reminder>", re.IGNORECASE | re.DOTALL
)
SYSTEM_REMINDER_OPEN_RE = re.compile(r"<system-reminder\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
SORT_FALLBACK_TIMESTAMP = datetime.min.replace(tzinfo=UTC)


class SearchService:
    """High-level search operations returning structured results."""

    def __init__(self, db: QdrantDB | None = None):
        """
        Initialize search service.

        Args:
            db: Optional QdrantDB instance. If None, creates one from config.
        """
        if db is None:
            config = get_config()
            db = QdrantDB(config.qdrant_url, config.qdrant_api_key, config.embedding_model)
        self.db = db
        self._collection: str | None = None

    @property
    def collection(self) -> str:
        """Auto-detect best collection (prefer hybrid if exists)."""
        if self._collection is None:
            try:
                collections = [c.name for c in self.db.client.get_collections().collections]
                self._collection = (
                    "conversations_hybrid"
                    if "conversations_hybrid" in collections
                    else "conversations"
                )
            except Exception:
                self._collection = "conversations"
        return self._collection

    def search(
        self,
        query: str,
        limit: int = 10,
        project: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        role: str | None = None,
        conversation: str | None = None,
        min_score: float = 0.5,
        boost_recent: bool = True,
        include_tool_results: bool = False,
        include_thinking: bool = False,
        group_by_conversation: bool = False,
    ) -> SearchResult | ConversationSearchResult | ErrorResult:
        """
        Semantic search with filtering and recency boosting.

        Args:
            query: Search query text
            limit: Maximum number of results
            project: Filter by project path (partial match)
            from_date: Filter by date from (ISO format: YYYY-MM-DD)
            to_date: Filter by date to (ISO format: YYYY-MM-DD)
            role: Filter by message role (user/assistant)
            conversation: Filter by conversation ID (exact match)
            min_score: Minimum relevance score threshold (0.0-1.0)
            boost_recent: Whether to boost recent messages in ranking
            include_tool_results: Include full tool result content (default False)
            include_thinking: Include thinking block content (default False)
            group_by_conversation: Group results by conversation (default False)

        Returns:
            SearchResult with Message objects, ConversationSearchResult if grouped,
            or ErrorResult on failure.
        """
        try:
            # Suppress INFO logs from db module (model loading messages)
            logging.getLogger("claude_kb.db").setLevel(logging.WARNING)

            # Ensure indices exist for efficient filtering
            self.db.ensure_indices(self.collection)

            # Generate dense embeddings
            self.db.embedding_model.load()
            query_vector = self.db.embedding_model.encode(
                [query], batch_size=1, show_progress=False
            )[0]

            # Generate sparse vector for hybrid search if available
            sparse_vector = None
            if self.db.has_sparse_vectors(self.collection):
                self.db.sparse_model.load()
                sparse_embeddings = self.db.sparse_model.encode([query])
                sparse = sparse_embeddings[0]
                sparse_vector = {
                    "indices": sparse.indices.tolist(),
                    "values": sparse.values.tolist(),
                }

            # Build Qdrant filter (now includes server-side date filtering via timestamp_unix)
            query_filter = self._build_filter(project, role, conversation, from_date, to_date)

            # For conversation grouping, fetch more results to ensure good coverage
            if group_by_conversation:
                search_limit = limit * 10  # Fetch more to group effectively
            else:
                search_limit = limit  # No need to over-fetch, date filtering is server-side

            results = self.db.search(
                query_vector=query_vector.tolist(),
                collection=self.collection,
                limit=search_limit,
                query_filter=query_filter,
                score_threshold=min_score,
                sparse_vector=sparse_vector,
            )

            # Fallback: client-side date filtering for old data without timestamp_unix
            if (from_date or to_date) and self._needs_client_side_date_filter(results):
                results = self._filter_by_date(results, from_date, to_date)

            # Recency boosting
            if boost_recent and results:
                results = self._apply_recency_boost(results)

            # Group by conversation if requested
            if group_by_conversation:
                return self._group_by_conversation(query, results, limit)

            # Convert to Pydantic models and limit
            messages = [
                self._to_message(
                    r,
                    include_tool_results=include_tool_results,
                    include_thinking=include_thinking,
                )
                for r in results[:limit]
            ]

            return SearchResult(
                query=query,
                collection=self.collection,
                count=len(messages),
                results=messages,
            )

        except Exception as e:
            logger.exception("Search failed")
            return ErrorResult(error=str(e))

    def get(
        self,
        message_id: str | None = None,
        context_depth: int = 0,
        include_tool_results: bool = False,
        include_thinking: bool = False,
        conversation_id: str | None = None,
        up_to: str | None = None,
        max_messages: int = 100,
    ) -> GetResult | ErrorResult:
        """
        Retrieve message by ID with optional thread context, or restore a conversation.

        Args:
            message_id: Message ID to retrieve (mutually exclusive with conversation_id)
            context_depth: If > 0, include ±N surrounding messages
            include_tool_results: Include full tool result content (default False).
                                  Not supported in restore mode.
            include_thinking: Include thinking block content (default False).
                              Not supported in restore mode.
            conversation_id: Conversation ID to restore (mutually exclusive with message_id)
            up_to: Optional message ID to truncate restored conversation to (inclusive)
            max_messages: Maximum messages to return in restore mode

        Returns:
            GetResult with message and optional thread context.
        """
        try:
            if message_id is not None and conversation_id is not None:
                return ErrorResult(
                    error="message_id and conversation_id are mutually exclusive",
                    suggestion="Use message_id for single-message/thread retrieval OR "
                    "conversation_id for restore mode.",
                )

            if conversation_id:
                if include_tool_results or include_thinking:
                    return ErrorResult(
                        error="include_tool_results/include_thinking are not supported in "
                        "conversation restore mode",
                        suggestion="Remove include_tool_results/include_thinking when using "
                        "conversation_id.",
                    )
                return self.get_conversation(
                    conversation_id=conversation_id,
                    up_to=up_to,
                    max_messages=max_messages,
                )

            if message_id is None:
                return ErrorResult(error="Either message_id or conversation_id must be provided")

            if context_depth > 0:
                # Get thread context
                thread_points = self.db.get_thread_context(
                    self.collection, message_id, context_depth
                )
                if not thread_points:
                    return ErrorResult(error=f"Message not found: {message_id}")

                # Find target message and build thread
                target = None
                thread = []
                for point in thread_points:
                    msg = self._point_to_message(
                        point,
                        include_tool_results=include_tool_results,
                        include_thinking=include_thinking,
                    )
                    if msg.id == message_id or str(point.id) == message_id:
                        target = msg
                    thread.append(msg)

                if target is None:
                    # Fallback: use first message
                    target = thread[0] if thread else None
                    if target is None:
                        return ErrorResult(error=f"Message not found: {message_id}")

                return GetResult(message=target, thread=thread)
            else:
                # Single message
                point = self.db.get_by_id(self.collection, message_id)
                if point is None:
                    return ErrorResult(error=f"Message not found: {message_id}")

                return GetResult(
                    message=self._point_to_message(
                        point,
                        include_tool_results=include_tool_results,
                        include_thinking=include_thinking,
                    )
                )

        except Exception as e:
            logger.exception("Get failed")
            return ErrorResult(error=str(e))

    def get_conversation(
        self,
        conversation_id: str,
        up_to: str | None = None,
        max_messages: int = 100,
    ) -> GetResult | ErrorResult:
        """
        Restore a conversation transcript for injecting into a new session.

        The transcript is aggressively cleaned to keep only human-readable user/assistant text.
        """
        try:
            if max_messages <= 0:
                return ErrorResult(error="max_messages must be greater than 0")

            points = self._scroll_conversation_points(conversation_id)
            if not points:
                return ErrorResult(error=f"Conversation not found: {conversation_id}")

            points.sort(
                key=lambda p: self._parse_timestamp(
                    (p.payload or {}).get("timestamp", ""),
                    fallback=SORT_FALLBACK_TIMESTAMP,
                )
            )

            if up_to:
                up_to_index = next(
                    (i for i, point in enumerate(points) if self._point_matches_id(point, up_to)),
                    None,
                )
                if up_to_index is None:
                    return ErrorResult(
                        error=f"Message {up_to} not found in conversation {conversation_id}"
                    )
                points = points[: up_to_index + 1]

            restored_messages: list[Message] = []
            for point in points:
                payload = point.payload or {}
                role = str(payload.get("role", "")).lower()
                if role not in RESTORE_ALLOWED_ROLES:
                    continue

                cleaned_content = self._extract_restore_text(payload.get("content", ""))
                if not cleaned_content:
                    continue

                point_id = payload.get("message_id") or str(point.id)
                restored_messages.append(
                    Message(
                        id=point_id,
                        role=role,
                        content=f"[{role}] {cleaned_content}",
                        timestamp=self._parse_timestamp(payload.get("timestamp", "")),
                        project=payload.get("project_path", "N/A"),
                        conversation_id=conversation_id,
                        parent_id=payload.get("parent_message_id"),
                        score=None,
                    )
                )

            if not restored_messages:
                return ErrorResult(error=f"No user/assistant messages found in {conversation_id}")

            truncated = False
            if len(restored_messages) > max_messages:
                restored_messages = restored_messages[-max_messages:]
                truncated = True

            for index, message in enumerate(restored_messages[:-1]):
                restored_messages[index].content = f"{message.content}\n\n---"

            first_ts = restored_messages[0].timestamp
            last_ts = restored_messages[-1].timestamp
            summary = (
                f"conversation_id={conversation_id}\n"
                f"message_count={len(restored_messages)}\n"
                f"time_range={first_ts.isoformat()} to {last_ts.isoformat()}"
            )
            if up_to:
                summary += f"\nup_to={up_to}"
            if truncated:
                summary += f"\ntruncated_to_most_recent={max_messages}"

            return GetResult(
                message=Message(
                    id=conversation_id,
                    role="system",
                    content=summary,
                    timestamp=last_ts,
                    project=restored_messages[0].project,
                    conversation_id=conversation_id,
                    parent_id=None,
                    score=None,
                ),
                thread=restored_messages,
            )
        except Exception as e:
            logger.exception("Conversation restore failed")
            return ErrorResult(error=str(e))

    def status(self, include_projects: bool = False) -> StatusResult | ErrorResult:
        """
        Get database status and statistics.

        Args:
            include_projects: Whether to include per-project breakdown

        Returns:
            StatusResult with collection stats and optional project stats.
        """
        try:
            config = get_config()
            stats = self.db.get_stats()

            result = StatusResult(
                qdrant_url=config.qdrant_url,
                embedding_model=config.embedding_model,
                collections=stats,
            )

            if include_projects:
                # Check for conversations collection
                conv_collection = None
                for name in stats:
                    if name.startswith("conversations"):
                        conv_collection = name
                        break

                if conv_collection:
                    project_stats = self.db.get_project_stats(conv_collection)
                    result.projects = [ProjectStats(**p) for p in project_stats]

            return result

        except Exception as e:
            logger.exception("Status failed")
            return ErrorResult(error=str(e))

    # --- Private helpers ---

    def _build_filter(
        self,
        project: str | None,
        role: str | None,
        conversation: str | None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> Filter | None:
        """Build Qdrant filter from parameters including date range."""
        filter_conditions = []

        # Project filter (partial match using MatchText)
        if project:
            filter_conditions.append(
                FieldCondition(key="project_path", match=MatchText(text=project))
            )

        # Conversation ID filter (exact match)
        if conversation:
            filter_conditions.append(
                FieldCondition(key="conversation_id", match=MatchValue(value=conversation))
            )

        # Role filter (exact match)
        if role:
            filter_conditions.append(
                FieldCondition(key="role", match=MatchValue(value=role.lower()))
            )

        # Date range filter using timestamp_unix (server-side)
        date_range = self._build_date_range(from_date, to_date)
        if date_range:
            filter_conditions.append(FieldCondition(key="timestamp_unix", range=date_range))

        # Combine all conditions with AND logic
        if filter_conditions:
            return Filter(must=filter_conditions)

        return None

    def _build_date_range(self, from_date: str | None, to_date: str | None) -> Range | None:
        """Build Range condition for timestamp_unix field.

        Args:
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format

        Returns:
            Range object or None if no date filters specified
        """
        if not from_date and not to_date:
            return None

        gte_unix = None
        lte_unix = None

        if from_date:
            try:
                from_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(
                    hour=0, minute=0, second=0, tzinfo=UTC
                )
                gte_unix = int(from_dt.timestamp())
            except ValueError:
                logger.warning(f"Invalid from_date format: {from_date}, expected YYYY-MM-DD")

        if to_date:
            try:
                to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59, tzinfo=UTC
                )
                lte_unix = int(to_dt.timestamp())
            except ValueError:
                logger.warning(f"Invalid to_date format: {to_date}, expected YYYY-MM-DD")

        if gte_unix is None and lte_unix is None:
            return None

        return Range(gte=gte_unix, lte=lte_unix)

    def _needs_client_side_date_filter(self, results: list) -> bool:
        """Check if results contain old data without timestamp_unix.

        Returns True if any result is missing timestamp_unix field,
        indicating we need client-side filtering as fallback.
        """
        if not results:
            return False

        # Sample first result to check for timestamp_unix
        first_payload = results[0].payload if results else {}
        return "timestamp_unix" not in first_payload

    def _filter_by_date(self, results: list, from_date: str | None, to_date: str | None) -> list:
        """Client-side date filtering (since timestamps are ISO strings)."""
        filtered = []
        from_ts = f"{from_date}T00:00:00" if from_date else None
        to_ts = f"{to_date}T23:59:59" if to_date else None

        for result in results:
            timestamp = result.payload.get("timestamp", "")
            if from_ts and timestamp < from_ts:
                continue
            if to_ts and timestamp > to_ts:
                continue
            filtered.append(result)

        return filtered

    def _apply_recency_boost(self, results: list) -> list:
        """Apply exponential decay boost for recent messages."""
        now = datetime.now(UTC)
        one_week_seconds = 7 * 24 * 60 * 60  # Decay half-life

        def get_boosted_score(result):
            """Calculate boosted score based on message age."""
            timestamp_str = result.payload.get("timestamp", "")
            try:
                # Parse ISO timestamp
                if "+" in timestamp_str or timestamp_str.endswith("Z"):
                    ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    ts = datetime.fromisoformat(timestamp_str).replace(tzinfo=UTC)

                # Calculate age in seconds
                age_seconds = (now - ts).total_seconds()
                age_seconds = max(0, age_seconds)  # No negative ages

                # Exponential decay: boost = 0.2 * exp(-age / scale)
                # Adds up to 0.2 boost for very recent messages, decaying over time
                recency_boost = 0.2 * math.exp(-age_seconds / one_week_seconds)

                return result.score + recency_boost
            except (ValueError, TypeError):
                return result.score

        # Sort by boosted score
        boosted = [(get_boosted_score(r), r) for r in results]
        boosted.sort(key=lambda x: x[0], reverse=True)

        # Create wrapper objects with updated scores
        class BoostedResult:
            """Wrapper to show boosted score while preserving original data."""

            def __init__(self, original, boosted_score):
                self.payload = original.payload
                self.id = original.id
                self.score = boosted_score

        return [BoostedResult(r, score) for score, r in boosted]

    def _group_by_conversation(
        self, query: str, results: list, limit: int
    ) -> ConversationSearchResult:
        """
        Group search results by conversation_id.

        Args:
            query: Original search query
            results: Search results to group
            limit: Maximum number of conversations to return

        Returns:
            ConversationSearchResult with conversation summaries
        """
        from collections import defaultdict

        # Group results by conversation_id
        conversations: dict[str, list] = defaultdict(list)
        for result in results:
            conv_id = result.payload.get("conversation_id")
            if conv_id:
                conversations[conv_id].append(result)

        # Build conversation summaries
        summaries = []
        for conv_id, messages in conversations.items():
            # Sort messages by timestamp
            messages.sort(key=lambda m: m.payload.get("timestamp", ""))

            # Get timestamps
            timestamps = []
            for msg in messages:
                ts_str = msg.payload.get("timestamp", "")
                try:
                    if "+" in ts_str or ts_str.endswith("Z"):
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    else:
                        ts = datetime.fromisoformat(ts_str).replace(tzinfo=UTC)
                    timestamps.append(ts)
                except (ValueError, TypeError):
                    pass

            if not timestamps:
                continue

            # Get best score (highest relevance)
            best_score = max(m.score for m in messages)

            # Get preview from first matching message content
            first_content = messages[0].payload.get("content", "")
            preview = self._extract_preview(first_content, max_length=200)

            # Get project from first message
            project = messages[0].payload.get("project_path", "N/A")

            # Get total message count for this conversation from DB
            # (messages list only contains matches, not full conversation)
            message_count = self._get_conversation_message_count(conv_id)

            summaries.append(
                ConversationSummary(
                    conversation_id=conv_id,
                    project=project,
                    first_timestamp=min(timestamps),
                    last_timestamp=max(timestamps),
                    message_count=message_count,
                    preview=preview,
                    best_score=best_score,
                )
            )

        # Sort by best_score descending
        summaries.sort(key=lambda s: s.best_score, reverse=True)

        # Limit results
        summaries = summaries[:limit]

        return ConversationSearchResult(
            query=query,
            collection=self.collection,
            count=len(summaries),
            conversations=summaries,
        )

    def _extract_preview(self, content: str | list | dict, max_length: int = 200) -> str:
        """Extract a text preview from message content."""
        import json

        # Handle structured content (list of content blocks)
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        return text[:max_length] + ("..." if len(text) > max_length else "")
            # No text block found, return summary
            return f"[{len(content)} content blocks]"

        if isinstance(content, dict):
            # Try to get text from dict
            if "text" in content:
                text = content["text"]
                return str(text)[:max_length] + ("..." if len(str(text)) > max_length else "")
            # Return type indicator if available
            content_type = content.get("type", "unknown")
            return f"[{content_type} content]"

        # Plain string
        if isinstance(content, str):
            # Try to parse as JSON first (only once, don't recurse)
            if content.strip().startswith("[") or content.strip().startswith("{"):
                try:
                    parsed = json.loads(content)
                    # Extract from parsed structure without recursion
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                if text:
                                    return text[:max_length] + (
                                        "..." if len(text) > max_length else ""
                                    )
                        return f"[{len(parsed)} content blocks]"
                    elif isinstance(parsed, dict) and "text" in parsed:
                        text = parsed["text"]
                        return str(text)[:max_length] + (
                            "..." if len(str(text)) > max_length else ""
                        )
                except (json.JSONDecodeError, ValueError):
                    pass
            return content[:max_length] + ("..." if len(content) > max_length else "")

        return str(content)[:max_length]

    def _get_conversation_message_count(self, conversation_id: str) -> int:
        """Get total message count for a conversation."""
        try:
            return len(self._scroll_conversation_points(conversation_id, with_payload=False))
        except Exception:
            return 0  # Return 0 if we can't count

    def _clean_content(
        self,
        content: str | list | dict,
        include_tool_results: bool = True,
        include_thinking: bool = True,
    ) -> str:
        """
        Clean content by removing signatures and optionally filtering heavy content.

        Args:
            content: Raw message content (string, list, or dict)
            include_tool_results: If False, replace tool_result content with summary
            include_thinking: If False, replace thinking block content with summary

        Returns:
            Cleaned content string (JSON for structured content)
        """
        import json

        def clean_item(obj):
            """Recursively clean content items."""
            if isinstance(obj, dict):
                # Remove signatures
                obj = {k: v for k, v in obj.items() if k != "signature"}

                # Handle tool_result blocks
                if obj.get("type") == "tool_result" and not include_tool_results:
                    result_content = obj.get("content", "")
                    content_len = len(str(result_content))
                    return {
                        "type": "tool_result",
                        "tool_use_id": obj.get("tool_use_id"),
                        "content": f"[tool result: {content_len} chars]",
                    }

                # Handle thinking blocks
                if obj.get("type") == "thinking" and not include_thinking:
                    thinking_content = obj.get("thinking", "")
                    content_len = len(str(thinking_content))
                    return {
                        "type": "thinking",
                        "thinking": f"[thinking: {content_len} chars]",
                    }

                # Recurse for other dicts
                return {k: clean_item(v) for k, v in obj.items()}

            elif isinstance(obj, list):
                return [clean_item(item) for item in obj]

            return obj

        # Parse JSON string if needed
        if isinstance(content, str):
            if content.strip().startswith("[") or content.strip().startswith("{"):
                try:
                    parsed = json.loads(content)
                    cleaned = clean_item(parsed)
                    return json.dumps(cleaned, indent=2)
                except (json.JSONDecodeError, ValueError):
                    return content
            return content

        # Handle list/dict directly
        if isinstance(content, list | dict):
            cleaned = clean_item(content)
            return json.dumps(cleaned, indent=2)

        return str(content)

    def _extract_restore_text(self, content: str | list | dict) -> str:
        """Extract only human-readable text for restore mode."""
        parsed: str | list | dict = content
        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    parsed = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    return self._strip_system_reminders(content)
            else:
                return self._strip_system_reminders(content)

        text_blocks = self._extract_restore_text_blocks(parsed)
        cleaned = [self._strip_system_reminders(block) for block in text_blocks]
        return "\n\n".join(block for block in cleaned if block)

    def _extract_restore_text_blocks(self, content: str | list | dict) -> list[str]:
        """Extract text blocks while skipping tool/thinking structures."""
        if isinstance(content, str):
            return [content]

        if isinstance(content, list):
            blocks: list[str] = []
            for item in content:
                blocks.extend(self._extract_restore_text_blocks(item))
            return blocks

        if isinstance(content, dict):
            block_type = str(content.get("type", "")).lower()
            if block_type in RESTORE_SKIP_BLOCK_TYPES:
                return []
            if block_type == "text":
                text = content.get("text")
                return [str(text)] if text else []
            if "content" in content:
                nested_content = content.get("content")
                if isinstance(nested_content, str | list | dict):
                    return self._extract_restore_text_blocks(nested_content)
            text = content.get("text")
            if isinstance(text, str):
                return [text]
            return []

        return [str(content)]

    def _strip_system_reminders(self, text: str) -> str:
        """Remove system-reminder tags and normalize spacing."""
        text = SYSTEM_REMINDER_RE.sub("", text)
        text = SYSTEM_REMINDER_OPEN_RE.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _scroll_conversation_points(
        self,
        conversation_id: str,
        *,
        with_payload: bool = True,
    ) -> list:
        """Scroll all points for a conversation ID."""
        points = []
        offset = None
        while True:
            batch, offset = self.db.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="conversation_id",
                            match=MatchValue(value=conversation_id),
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=with_payload,
            )
            if not batch:
                break
            points.extend(batch)
            if offset is None:
                break
        return points

    def _point_matches_id(self, point, message_id: str) -> bool:
        """Check if point ID or payload message_id matches a target ID."""
        payload = point.payload or {}
        return str(point.id) == message_id or payload.get("message_id") == message_id

    def _parse_timestamp(self, timestamp_str: str, fallback: datetime | None = None) -> datetime:
        """Parse ISO timestamp with sane fallback."""
        try:
            parsed = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed
        except (ValueError, TypeError, AttributeError):
            return fallback if fallback is not None else datetime.now(UTC)

    def _to_message(
        self,
        result,
        include_tool_results: bool = True,
        include_thinking: bool = True,
    ) -> Message:
        """Convert search result to Message model."""
        payload = result.payload
        point_id = payload.get("message_id") or payload.get("id") or str(result.id)

        return Message(
            id=point_id,
            role=payload.get("role", "unknown"),
            content=self._clean_content(
                payload.get("content", ""),
                include_tool_results=include_tool_results,
                include_thinking=include_thinking,
            ),
            timestamp=self._parse_timestamp(payload.get("timestamp", "")),
            project=payload.get("project_path", "N/A"),
            conversation_id=payload.get("conversation_id"),
            parent_id=payload.get("parent_message_id"),
            score=result.score,
        )

    def _point_to_message(
        self,
        point,
        include_tool_results: bool = True,
        include_thinking: bool = True,
    ) -> Message:
        """Convert Qdrant point to Message model."""
        payload = point.payload or {}
        point_id = payload.get("message_id") or str(point.id)

        return Message(
            id=point_id,
            role=payload.get("role", "unknown"),
            content=self._clean_content(
                payload.get("content", ""),
                include_tool_results=include_tool_results,
                include_thinking=include_thinking,
            ),
            timestamp=self._parse_timestamp(payload.get("timestamp", "")),
            project=payload.get("project_path", "N/A"),
            conversation_id=payload.get("conversation_id"),
            parent_id=payload.get("parent_message_id"),
            score=None,  # No score for direct retrieval
        )
