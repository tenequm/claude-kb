"""Search service layer for claude-kb."""

import logging
import math
from datetime import UTC, datetime

from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue

from .config import get_config
from .db import QdrantDB
from .models import ErrorResult, GetResult, Message, ProjectStats, SearchResult, StatusResult

logger = logging.getLogger(__name__)


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
    ) -> SearchResult | ErrorResult:
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

        Returns:
            SearchResult with Message objects, or ErrorResult on failure.
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

            # Build Qdrant filter
            query_filter = self._build_filter(project, role, conversation)

            # Search (fetch extra only for date filtering which happens client-side)
            search_limit = limit * 3 if (from_date or to_date) else limit
            results = self.db.search(
                query_vector=query_vector.tolist(),
                collection=self.collection,
                limit=search_limit,
                query_filter=query_filter,
                score_threshold=min_score,
                sparse_vector=sparse_vector,
            )

            # Client-side date filtering
            if from_date or to_date:
                results = self._filter_by_date(results, from_date, to_date)

            # Recency boosting
            if boost_recent and results:
                results = self._apply_recency_boost(results)

            # Convert to Pydantic models and limit
            messages = [self._to_message(r) for r in results[:limit]]

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
        message_id: str,
        context_depth: int = 0,
    ) -> GetResult | ErrorResult:
        """
        Retrieve message by ID with optional thread context.

        Args:
            message_id: Message ID to retrieve
            context_depth: If > 0, include ±N surrounding messages

        Returns:
            GetResult with message and optional thread context.
        """
        try:
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
                    msg = self._point_to_message(point)
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

                return GetResult(message=self._point_to_message(point))

        except Exception as e:
            logger.exception("Get failed")
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
    ) -> Filter | None:
        """Build Qdrant filter from parameters."""
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

        # Combine all conditions with AND logic
        if filter_conditions:
            return Filter(must=filter_conditions)

        return None

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

    def _clean_content(self, content: str | list | dict) -> str:
        """
        Clean content by removing only signatures (useless base64 noise).

        Preserves all semantic content including tool calls, thinking, etc.
        for search result explainability.
        """
        import json

        def remove_signatures(obj):
            """Recursively remove 'signature' keys from nested structures."""
            if isinstance(obj, dict):
                return {k: remove_signatures(v) for k, v in obj.items() if k != "signature"}
            elif isinstance(obj, list):
                return [remove_signatures(item) for item in obj]
            return obj

        # Parse JSON string if needed
        if isinstance(content, str):
            if content.strip().startswith("[") or content.strip().startswith("{"):
                try:
                    parsed = json.loads(content)
                    cleaned = remove_signatures(parsed)
                    return json.dumps(cleaned, indent=2)
                except (json.JSONDecodeError, ValueError):
                    return content
            return content

        # Handle list/dict directly
        if isinstance(content, list | dict):
            cleaned = remove_signatures(content)
            return json.dumps(cleaned, indent=2)

        return str(content)

    def _to_message(self, result) -> Message:
        """Convert search result to Message model."""
        payload = result.payload
        point_id = payload.get("message_id") or payload.get("id") or str(result.id)

        # Parse timestamp
        timestamp_str = payload.get("timestamp", "")
        try:
            if "+" in timestamp_str or timestamp_str.endswith("Z"):
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                timestamp = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            timestamp = datetime.now(UTC)

        return Message(
            id=point_id,
            role=payload.get("role", "unknown"),
            content=self._clean_content(payload.get("content", "")),
            timestamp=timestamp,
            project=payload.get("project_path", "N/A"),
            conversation_id=payload.get("conversation_id"),
            parent_id=payload.get("parent_message_id"),
            score=result.score,
        )

    def _point_to_message(self, point) -> Message:
        """Convert Qdrant point to Message model."""
        payload = point.payload or {}
        point_id = payload.get("message_id") or str(point.id)

        # Parse timestamp
        timestamp_str = payload.get("timestamp", "")
        try:
            if "+" in timestamp_str or timestamp_str.endswith("Z"):
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                timestamp = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            timestamp = datetime.now(UTC)

        return Message(
            id=point_id,
            role=payload.get("role", "unknown"),
            content=self._clean_content(payload.get("content", "")),
            timestamp=timestamp,
            project=payload.get("project_path", "N/A"),
            conversation_id=payload.get("conversation_id"),
            parent_id=payload.get("parent_message_id"),
            score=None,  # No score for direct retrieval
        )
