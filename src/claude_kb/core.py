"""Core backend: Qdrant client with sentence-transformers on MPS."""

import hashlib
import logging

import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Collection names
COLLECTION_CONVERSATIONS = "conversations"
COLLECTION_ENTITIES = "entities"
COLLECTION_DOCUMENTS = "documents"

# Content truncation
MAX_CONTENT_PREVIEW_LENGTH = 500  # Characters to show in search results

# Size estimation
ESTIMATED_KB_PER_POINT = 1  # Rough estimate for 768-dim vector + metadata


# =============================================================================
# EMBEDDING MODEL
# =============================================================================


class EmbeddingModel:
    """Sentence-transformers wrapper with MPS support for fast embeddings."""

    def __init__(self, model_name: str, device: str | None = None):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name
            device: Device to use (None = auto-detect: CUDA > MPS > CPU)
        """
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._device = device

    def load(self) -> None:
        """Load the model (lazy loading on first use)."""
        if self._model is not None:
            return

        # Suppress sentence-transformers logging during model load
        import logging as std_logging

        st_logger = std_logging.getLogger("sentence_transformers")
        original_level = st_logger.level
        st_logger.setLevel(std_logging.WARNING)

        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(
            self.model_name, trust_remote_code=True, device=self._device
        )

        # Restore logging level
        st_logger.setLevel(original_level)

        device_str = str(self._model.device).lower()
        if "mps" in device_str:
            logger.info("✓ Using Apple Silicon GPU (MPS)")
        elif "cuda" in device_str:
            logger.info("✓ Using NVIDIA GPU (CUDA)")
        else:
            logger.info(f"Using CPU: {device_str}")

    def encode(
        self, texts: list[str], batch_size: int = 100, show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings (len(texts), embedding_dim)
        """
        if self._model is None:
            self.load()

        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


# =============================================================================
# QDRANT CLIENT
# =============================================================================


class QdrantDB:
    """Qdrant vector database wrapper with sentence-transformers."""

    def __init__(
        self,
        url: str,
        api_key: str | None,
        embedding_model: str,
    ):
        """
        Initialize Qdrant client with sentence-transformers.

        Args:
            url: Qdrant server URL (e.g., http://localhost:6333)
            api_key: Optional API key for Qdrant Cloud
            embedding_model: Model name (from config)
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.url = url
        self.embedding_model_name = embedding_model
        self.embedding_model = EmbeddingModel(embedding_model)
        logger.info(f"Using embedding model: {embedding_model}")

    def init_collections(self) -> None:
        """
        Verify Qdrant connection.

        Note: When using FastEmbed, collections are auto-created on first .add()
        with the correct vector configuration for the embedding model.
        """
        # Just verify we can connect to Qdrant
        try:
            collections = self.client.get_collections()
            logger.info(
                f"✓ Connected to Qdrant ({len(collections.collections)} existing collections)"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def get_by_id(self, collection: str, point_id: str) -> PointStruct | None:
        """
        Retrieve single point by ID.

        Args:
            collection: Collection name
            point_id: Point ID

        Returns:
            Point object or None
        """
        results = self.client.retrieve(collection_name=collection, ids=[point_id])
        return results[0] if results else None

    def get_stats(self) -> dict[str, int]:
        """
        Get collection statistics.

        Returns:
            Dict mapping collection name to point count
        """
        try:
            collections = self.client.get_collections().collections
            stats = {}
            for c in collections:
                info = self.client.get_collection(c.name)
                stats[c.name] = info.points_count
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def get_project_stats(
        self, collection: str = COLLECTION_CONVERSATIONS
    ) -> list[dict[str, int | str]]:
        """
        Get per-project statistics (sessions and messages).

        Returns:
            List of dicts: [{"project": path, "sessions": N, "messages": M}, ...]
        """
        try:
            # Scroll through all points and group by project
            project_data = {}
            offset = None

            while True:
                results = self.client.scroll(
                    collection_name=collection,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                )

                points, offset = results

                if not points:
                    break

                for point in points:
                    payload = point.payload
                    project = payload.get("project_path", "Unknown")
                    conv_id = payload.get("conversation_id")

                    if project not in project_data:
                        project_data[project] = {
                            "conversations": set(),
                            "messages": 0,
                        }

                    project_data[project]["conversations"].add(conv_id)
                    project_data[project]["messages"] += 1

                if offset is None:
                    break

            # Convert to list and sort by message count
            result = []
            for project, data in project_data.items():
                result.append(
                    {
                        "project": project,
                        "sessions": len(data["conversations"]),
                        "messages": data["messages"],
                    }
                )

            # Sort by message count descending
            result.sort(key=lambda x: x["messages"], reverse=True)
            return result

        except Exception as e:
            logger.error(f"Failed to get project stats: {e}")
            return []


class AsyncQdrantDB:
    """Async Qdrant vector database wrapper with sentence-transformers."""

    def __init__(
        self,
        url: str,
        api_key: str | None,
        embedding_model: str,
    ):
        """
        Initialize Async Qdrant client with sentence-transformers.

        Args:
            url: Qdrant server URL (e.g., http://localhost:6333)
            api_key: Optional API key for Qdrant Cloud
            embedding_model: Model name (from config)
        """
        self.client = AsyncQdrantClient(url=url, api_key=api_key)
        self.url = url
        self.embedding_model_name = embedding_model
        self.embedding_model = EmbeddingModel(embedding_model)
        logger.info(f"Using async embedding model: {embedding_model}")

    async def init_collections(self) -> None:
        """
        Verify Qdrant connection.

        Note: When using FastEmbed, collections are auto-created on first .add()
        with the correct vector configuration for the embedding model.
        """
        # Just verify we can connect to Qdrant
        try:
            collections = await self.client.get_collections()
            logger.info(
                f"✓ Connected to Qdrant ({len(collections.collections)} existing collections)"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def close(self) -> None:
        """Close the async client connection."""
        await self.client.close()


# =============================================================================
# FORMATTING
# =============================================================================


def format_search_results(results, collection: str) -> str:
    """
    Format search results as structured text for LLM parsing.

    Output format:
    === id (score: 0.95) ===
    Field: value
    Field: value

    Content preview...
    ---

    Args:
        results: List of QueryResponse objects from client.query()
        collection: Collection name

    Returns:
        Formatted string
    """
    if not results:
        return "No results found."

    output = []

    for result in results:
        # Handle both QueryResponse (from client.query) and ScoredPoint (from client.search)
        if hasattr(result, "metadata"):
            payload = result.metadata
            score = result.score
            point_id = result.id
        else:
            payload = result.payload
            score = result.score
            point_id = payload.get("message_id") or payload.get("id") or str(result.id)

        lines = [f"=== {point_id} (score: {score:.2f}) ==="]

        if collection == COLLECTION_CONVERSATIONS:
            lines.append(f"Role: {payload.get('role', 'unknown')}")
            lines.append(f"Time: {payload.get('timestamp', 'N/A')}")
            lines.append(f"Project: {payload.get('project_path', 'N/A')}")
            lines.append("")
            content = payload.get("content", "")
            # Truncate long content
            if len(content) > MAX_CONTENT_PREVIEW_LENGTH:
                lines.append(content[:MAX_CONTENT_PREVIEW_LENGTH] + "...")
            else:
                lines.append(content)

        elif collection == COLLECTION_ENTITIES:
            lines.append(f"Type: {payload.get('type', 'unknown')}")
            lines.append(f"Name: {payload.get('name', 'N/A')}")
            description = payload.get("description", "")
            if description:
                lines.append("")
                lines.append(
                    description[:MAX_CONTENT_PREVIEW_LENGTH] + "..."
                    if len(description) > MAX_CONTENT_PREVIEW_LENGTH
                    else description
                )

        elif collection == COLLECTION_DOCUMENTS:
            lines.append(f"Title: {payload.get('title', 'N/A')}")
            lines.append(f"Source: {payload.get('source', 'N/A')}")
            tags = payload.get("tags", [])
            if tags:
                lines.append(f"Tags: {', '.join(tags)}")
            lines.append("")
            content = payload.get("content", "")
            lines.append(
                content[:MAX_CONTENT_PREVIEW_LENGTH] + "..."
                if len(content) > MAX_CONTENT_PREVIEW_LENGTH
                else content
            )

        else:
            # Generic format
            lines.append(str(payload))

        lines.append("---")
        output.append("\n".join(lines))

    return "\n".join(output)


def format_get_result(point, collection: str) -> str:
    """
    Format single item for get command.

    Args:
        point: Point object
        collection: Collection name

    Returns:
        Formatted string
    """
    if not point:
        return "Not found."

    payload = point.payload
    point_id = payload.get("message_id") or payload.get("id") or str(point.id)

    lines = [f"=== {point_id} ==="]

    if collection == "conversations":
        lines.append("Type: conversation/message")
        lines.append(f"Role: {payload.get('role', 'unknown')}")
        lines.append(f"Time: {payload.get('timestamp', 'N/A')}")
        lines.append(f"Conversation: {payload.get('conversation_id', 'N/A')}")
        lines.append(f"Project: {payload.get('project_path', 'N/A')}")
        lines.append("")
        lines.append(payload.get("content", ""))
        lines.append("")

        # Related messages
        parent_id = payload.get("parent_message_id")
        if parent_id:
            lines.append("Related:")
            lines.append(f"  Parent: {parent_id}")

    elif collection == "entities":
        lines.append(f"Type: {payload.get('type', 'unknown')}")
        lines.append(f"Name: {payload.get('name', 'N/A')}")
        lines.append(f"Created: {payload.get('created_at', 'N/A')}")
        lines.append(f"Updated: {payload.get('updated_at', 'N/A')}")
        lines.append("")
        lines.append(payload.get("description", ""))

        # Relationships
        relationships = payload.get("relationships", {})
        if relationships:
            lines.append("")
            lines.append("Relationships:")
            for rel_type, targets in relationships.items():
                lines.append(f"  {rel_type}: {', '.join(targets)}")

    elif collection == "documents":
        lines.append("Type: document")
        lines.append(f"Title: {payload.get('title', 'N/A')}")
        lines.append(f"Source: {payload.get('source', 'N/A')}")
        lines.append(f"Document ID: {payload.get('document_id', 'N/A')}")
        lines.append(
            f"Chunk: {payload.get('chunk_index', 0)} of {payload.get('metadata', {}).get('total_chunks', '?')}"
        )
        tags = payload.get("tags", [])
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
        lines.append("")
        lines.append(payload.get("content", ""))

    return "\n".join(lines)


# =============================================================================
# UTILITIES
# =============================================================================


def generate_id(content: str, type_prefix: str) -> str:
    """
    Generate content-hash ID for deduplication.

    Args:
        content: Content to hash
        type_prefix: Prefix (e.g., 'msg', 'doc', 'entity')

    Returns:
        ID string: {type_prefix}_{hash12}
    """
    hash_input = f"{type_prefix}:{content}"
    hash_hex = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    return f"{type_prefix}_{hash_hex}"
