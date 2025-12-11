"""Qdrant database client and embedding models."""

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.models import Record
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# Collection names
COLLECTION_CONVERSATIONS = "conversations"


@dataclass
class SparseEmbedding:
    """Sparse embedding result with indices and values arrays.

    Compatible with FastEmbed output format for seamless migration.
    """

    indices: np.ndarray
    values: np.ndarray


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

        assert self._model is not None  # Guaranteed by load()
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class SparseEmbeddingModel:
    """SPLADE sparse embedding model via sentence-transformers with MPS/CUDA support.

    Uses SparseEncoder which automatically detects and uses the best available
    device (MPS on Apple Silicon, CUDA on NVIDIA, or CPU).
    """

    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        """
        Initialize sparse embedding model.

        Args:
            model_name: HuggingFace model name for SPLADE
        """
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        """Load the model (lazy loading on first use)."""
        if self._model is not None:
            return

        from sentence_transformers import SparseEncoder

        logger.info(f"Loading sparse embedding model: {self.model_name}")
        self._model = SparseEncoder(self.model_name)

        device_str = str(self._model.device).lower()
        if "mps" in device_str:
            logger.info("✓ Using Apple Silicon GPU (MPS) for sparse embeddings")
        elif "cuda" in device_str:
            logger.info("✓ Using NVIDIA GPU (CUDA) for sparse embeddings")
        else:
            logger.info(f"Using CPU for sparse embeddings: {device_str}")

    def encode(self, texts: list[str]) -> list[SparseEmbedding]:
        """
        Generate sparse embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            List of SparseEmbedding objects with indices and values arrays
        """
        if self._model is None:
            self.load()

        assert self._model is not None  # Guaranteed by load()

        # Get dense output - sparse COO tensors don't work on MPS
        embeddings = self._model.encode(
            texts,
            convert_to_sparse_tensor=False,
            show_progress_bar=False,
        )

        # Extract sparse representations
        results = []
        for vec in embeddings:
            # Convert to numpy array
            arr: np.ndarray
            if hasattr(vec, "cpu"):
                # PyTorch tensor - move to CPU and convert to numpy
                arr = vec.cpu().numpy()  # type: ignore[union-attr]
            elif isinstance(vec, np.ndarray):
                arr = vec
            else:
                arr = np.array(vec)

            # Get non-zero indices and values
            nonzero_indices = np.nonzero(arr)[0]
            results.append(
                SparseEmbedding(
                    indices=nonzero_indices.astype(np.int32),
                    values=arr[nonzero_indices].astype(np.float32),
                )
            )
        return results


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
        self.sparse_model = SparseEmbeddingModel()
        logger.debug(f"Using embedding model: {embedding_model}")

    def init_collections(self) -> None:
        """
        Verify Qdrant connection.

        Note: When using FastEmbed, collections are auto-created on first .add()
        with the correct vector configuration for the embedding model.
        """
        try:
            collections = self.client.get_collections()
            logger.info(
                f"✓ Connected to Qdrant ({len(collections.collections)} existing collections)"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def has_sparse_vectors(self, collection: str) -> bool:
        """
        Check if collection has sparse vectors configured.

        Args:
            collection: Collection name

        Returns:
            True if collection has sparse vectors
        """
        try:
            info = self.client.get_collection(collection)
            vectors = info.config.params.vectors

            # Check for named vectors with 'sparse' key
            if isinstance(vectors, dict):
                return "sparse" in vectors

            return False
        except Exception:
            return False

    def get_by_id(self, collection: str, point_id: str) -> Record | None:
        """
        Retrieve single point by ID.

        Args:
            collection: Collection name
            point_id: Point ID

        Returns:
            Record object or None
        """
        results = self.client.retrieve(collection_name=collection, ids=[point_id])
        return results[0] if results else None

    def search(
        self,
        query_vector: list[float],
        collection: str,
        limit: int = 10,
        query_filter: Filter | None = None,
        score_threshold: float | None = None,
        sparse_vector: dict | None = None,
    ) -> list:
        """
        Search collection with optional hybrid search (dense + sparse) and RRF fusion.

        Args:
            query_vector: Dense query embedding vector
            collection: Collection name
            limit: Maximum number of results
            query_filter: Optional Qdrant Filter for metadata filtering
            score_threshold: Minimum score threshold (0.0-1.0)
            sparse_vector: Optional sparse vector dict with 'indices' and 'values'

        Returns:
            List of search results (ScoredPoint objects)
        """
        from qdrant_client import models

        # Detect vector configuration (named vs default)
        collection_info = self.client.get_collection(collection)
        vectors_config = collection_info.config.params.vectors
        sparse_config = collection_info.config.params.sparse_vectors

        # Check if collection has sparse vectors configured
        has_sparse = sparse_config is not None and "sparse" in (sparse_config or {})

        # Check if vectors is a dict (named vectors) or VectorParams (single default vector)
        if isinstance(vectors_config, dict):
            has_named_dense = "dense" in vectors_config
        else:
            has_named_dense = False

        # Use hybrid search if collection supports it and sparse vector provided
        if has_sparse and sparse_vector and has_named_dense:
            # Hybrid search with prefetch + RRF fusion
            logger.debug("Using hybrid search (dense + sparse) with RRF fusion")

            prefetch_limit = limit * 3  # Fetch more candidates for fusion

            result = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vector["indices"],
                            values=sparse_vector["values"],
                        ),
                        using="sparse",
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )
        else:
            # Dense-only search (original or migrated collection without sparse)
            logger.debug("Using dense-only search")

            result = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                using="dense" if has_named_dense else None,
            )

        return result.points

    def get_thread_context(self, collection: str, message_id: str, depth: int = 2) -> list[Record]:
        """
        Get message with surrounding context from conversation thread.

        Strategy:
        1. Get target message to find conversation_id and timestamp
        2. Scroll through conversation to find all messages
        3. Sort by timestamp
        4. Return target message ± depth messages

        Args:
            collection: Collection name (usually 'conversations')
            message_id: Target message ID
            depth: Number of messages before/after to include

        Returns:
            List of Record objects in chronological order
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Get target message
        target = self.get_by_id(collection, message_id)
        if not target:
            return []

        if not target.payload:
            return [target]

        conversation_id = target.payload.get("conversation_id")
        if not conversation_id:
            return [target]

        # Get all messages from this conversation
        messages = []
        offset = None

        while True:
            results = self.client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="conversation_id", match=MatchValue(value=conversation_id)
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
            )

            points, offset = results
            if not points:
                break

            messages.extend(points)

            if offset is None:
                break

        # Sort by timestamp
        messages.sort(key=lambda p: (p.payload or {}).get("timestamp", ""))

        # Find target message index
        target_idx = None
        for i, msg in enumerate(messages):
            if msg.id == message_id or (msg.payload or {}).get("message_id") == message_id:
                target_idx = i
                break

        if target_idx is None:
            return [target]

        # Get context window
        start = max(0, target_idx - depth)
        end = min(len(messages), target_idx + depth + 1)

        return messages[start:end]

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

    def get_project_stats(self, collection: str = COLLECTION_CONVERSATIONS) -> list[dict]:
        """
        Get per-project statistics (sessions and messages).

        Returns:
            List of dicts: [{"project": path, "sessions": N, "messages": M}, ...]
        """
        try:
            # Scroll through all points and group by project
            project_data: dict[str, dict] = {}
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
                    if not payload:
                        continue
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

    def ensure_indices(self, collection: str = COLLECTION_CONVERSATIONS) -> None:
        """
        Create payload indices for efficient filtering.

        Indices are created lazily on first search. Safe to call multiple times.
        """
        from qdrant_client.models import PayloadSchemaType

        indices = [
            ("project_path", PayloadSchemaType.KEYWORD),
            ("timestamp", PayloadSchemaType.DATETIME),
            ("role", PayloadSchemaType.KEYWORD),
            ("conversation_id", PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_schema in indices:
            try:
                self.client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema=field_schema,
                )
                logger.debug(f"Created index on {field_name}")
            except Exception:
                # Index already exists or other error - safe to ignore
                pass


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
        self.sparse_model = SparseEmbeddingModel()
        logger.debug(f"Using async embedding model: {embedding_model}")

    async def init_collections(self) -> None:
        """
        Verify Qdrant connection.

        Note: When using FastEmbed, collections are auto-created on first .add()
        with the correct vector configuration for the embedding model.
        """
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
