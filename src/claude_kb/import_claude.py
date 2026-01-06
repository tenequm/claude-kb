"""Import Claude Code conversation history."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import models
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# DISCOVERY
# =============================================================================


def discover_claude_projects(claude_dir: str = "~/.claude/projects") -> list[Path]:
    """
    Find all Claude Code project directories.

    Args:
        claude_dir: Path to Claude Code projects directory

    Returns:
        List of project directory paths
    """
    base = Path(claude_dir).expanduser()
    if not base.exists():
        logger.warning(f"Claude Code directory not found: {base}")
        return []

    projects = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")]
    return projects


def discover_sessions(project_dir: Path) -> list[Path]:
    """
    Find all session JSONL files in a project directory.

    Args:
        project_dir: Project directory path

    Returns:
        List of session file paths
    """
    if not project_dir.exists():
        return []

    # Session files are UUIDs with .jsonl extension
    # Skip agent-*.jsonl files (those are agent transcripts)
    session_files = [
        f
        for f in project_dir.glob("*.jsonl")
        if not f.name.startswith("agent-") and not f.name.startswith(".")
    ]
    return session_files


def get_project_path_from_encoded(encoded_name: str) -> str:
    """
    Convert encoded project directory name back to original path.

    Claude Code encodes paths like:
    /Users/username/Projects/my-project → -Users-username-Projects-my-project

    Args:
        encoded_name: Encoded directory name

    Returns:
        Original path
    """
    # Remove leading dash and replace dashes with slashes
    if encoded_name.startswith("-"):
        path = encoded_name[1:].replace("-", "/")
        return f"/{path}"
    return encoded_name


def find_project_dir(project_path: str, claude_dir: str = "~/.claude/projects") -> Path | None:
    """
    Find encoded project directory for given path.

    Args:
        project_path: Original project path (e.g., /Users/username/Projects/my-project)
        claude_dir: Claude Code projects directory

    Returns:
        Project directory path or None
    """
    base = Path(claude_dir).expanduser()
    if not base.exists():
        return None

    # Encode the path
    # /Users/username/Projects/my-project → -Users-username-Projects-my-project
    encoded = project_path.replace("/", "-")

    project_dir = base / encoded
    if project_dir.exists():
        return project_dir

    # Try without leading dash (shouldn't be needed but kept for safety)
    if encoded.startswith("-"):
        project_dir = base / encoded[1:]
        if project_dir.exists():
            return project_dir

    return None


# =============================================================================
# PARSING
# =============================================================================


def parse_session_file(session_file: Path, include_meta: bool = False) -> list[dict]:
    """
    Parse Claude Code JSONL session file.

    Args:
        session_file: Path to session JSONL file
        include_meta: Include meta messages (system prompts, etc.)

    Returns:
        List of message dicts with our schema:
        {
            "message_id": str (Claude's uuid),
            "conversation_id": str (Claude's sessionId),
            "role": "user" | "assistant",
            "content": str,
            "timestamp": datetime,
            "project_path": str,
            "cwd": str,
            "parent_message_id": Optional[str],
            "is_meta": bool,
            "metadata": {}
        }
    """
    messages = []

    try:
        with open(session_file) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)

                    # Skip meta messages unless requested
                    if data.get("isMeta") and not include_meta:
                        continue

                    # Extract message
                    msg = data.get("message", {})
                    if not msg:
                        continue

                    content = msg.get("content")
                    if not content:
                        continue

                    # Convert content to string if needed
                    if isinstance(content, list):
                        # content might be array of content blocks
                        # For now, stringify it
                        content = json.dumps(content)
                    elif not isinstance(content, str):
                        content = str(content)

                    # Parse timestamp
                    timestamp_str = data.get("timestamp", "")
                    try:
                        # Claude Code uses ISO format with Z
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        timestamp = datetime.now()
                        logger.warning(f"Invalid timestamp in {session_file.name}:{line_num}")

                    messages.append(
                        {
                            "message_id": data.get("uuid", f"unknown_{line_num}"),
                            "conversation_id": data.get("sessionId", "unknown"),
                            "role": msg.get("role", "unknown"),
                            "content": content,
                            "timestamp": timestamp,
                            "project_path": data.get("cwd", ""),
                            "cwd": data.get("cwd", ""),
                            "parent_message_id": data.get("parentUuid"),
                            "is_meta": data.get("isMeta", False),
                            "metadata": {
                                "version": data.get("version", ""),
                                "git_branch": data.get("gitBranch", ""),
                                "user_type": data.get("userType", ""),
                            },
                        }
                    )

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {session_file.name}:{line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num} in {session_file.name}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Failed to read {session_file}: {e}")
        return []

    return messages


# =============================================================================
# IMPORT
# =============================================================================


async def get_existing_message_ids_async(async_db, collection: str) -> set[str]:
    """
    Get all existing message IDs from Qdrant collection (async version).

    Args:
        async_db: AsyncQdrantDB instance
        collection: Collection name

    Returns:
        Set of message IDs (point IDs)
    """
    try:
        # Check if collection exists
        collections = await async_db.client.get_collections()
        if not any(c.name == collection for c in collections.collections):
            return set()

        # Scroll through all points to get IDs
        existing_ids = set()
        offset = None

        while True:
            # Scroll returns (points, next_offset)
            points, offset = await async_db.client.scroll(
                collection_name=collection,
                limit=1000,
                offset=offset,
                with_payload=False,  # Don't need payload, just IDs
                with_vectors=False,  # Don't need vectors
            )

            if not points:
                break

            # Extract IDs
            for point in points:
                existing_ids.add(str(point.id))

            # If no next offset, we're done
            if offset is None:
                break

        return existing_ids

    except Exception as e:
        logger.warning(f"Failed to get existing IDs: {e}")
        return set()


async def _upload_batch_async(
    client, collection: str, batch_messages: list[dict], embedding_model, sparse_model=None
) -> int:
    """
    Upload a batch of messages to Qdrant asynchronously.

    Args:
        client: AsyncQdrantClient instance
        collection: Collection name
        batch_messages: List of message dicts to upload
        embedding_model: EmbeddingModel instance (dense)
        sparse_model: Optional SparseEmbeddingModel for hybrid search

    Returns:
        Number of messages uploaded
    """
    # Prepare batch data
    batch_documents = []
    batch_ids = []
    batch_payloads = []

    for msg in batch_messages:
        # Document for embedding
        batch_documents.append(msg["content"])
        batch_ids.append(msg["message_id"])

        # Payload
        msg_timestamp = msg["timestamp"]
        if isinstance(msg_timestamp, datetime):
            timestamp_str = msg_timestamp.isoformat()
            timestamp_unix = int(msg_timestamp.timestamp())
        else:
            timestamp_str = str(msg_timestamp)
            # Try to parse and convert to unix timestamp
            try:
                parsed_ts = datetime.fromisoformat(str(msg_timestamp).replace("Z", "+00:00"))
                timestamp_unix = int(parsed_ts.timestamp())
            except (ValueError, TypeError):
                timestamp_unix = int(datetime.now(UTC).timestamp())

        batch_payloads.append(
            {
                "schema_version": 2,
                "message_id": msg["message_id"],
                "conversation_id": msg["conversation_id"],
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": timestamp_str,
                "timestamp_unix": timestamp_unix,
                "project_path": msg["project_path"],
                "cwd": msg["cwd"],
                "parent_message_id": msg["parent_message_id"],
                "is_meta": msg["is_meta"],
                "metadata": msg["metadata"],
            }
        )

    # Generate dense embeddings using sentence-transformers (MPS-accelerated)
    embeddings = embedding_model.encode(batch_documents, batch_size=100, show_progress=False)

    # Check if collection supports hybrid search (named vectors)
    try:
        info = await client.get_collection(collection)
        vectors_config = info.config.params.vectors
        sparse_config = info.config.params.sparse_vectors
        has_sparse = sparse_config is not None and "sparse" in (sparse_config or {})
        has_named_dense = isinstance(vectors_config, dict) and "dense" in vectors_config
        use_hybrid = has_sparse and has_named_dense and sparse_model is not None
    except Exception:
        use_hybrid = False

    # Upload batch with pre-computed embeddings
    if use_hybrid and sparse_model is not None:
        # Generate sparse embeddings for hybrid search
        sparse_embeddings = sparse_model.encode(
            [str(doc)[:8000] for doc in batch_documents]  # Truncate for SPLADE
        )

        points = [
            models.PointStruct(
                id=batch_ids[i],
                vector={
                    "dense": embeddings[i].tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_embeddings[i].indices.tolist(),
                        values=sparse_embeddings[i].values.tolist(),
                    ),
                },
                payload=batch_payloads[i],
            )
            for i in range(len(batch_ids))
        ]
    else:
        # Dense-only (legacy collection)
        points = [
            models.PointStruct(
                id=batch_ids[i], vector=embeddings[i].tolist(), payload=batch_payloads[i]
            )
            for i in range(len(batch_ids))
        ]

    await client.upsert(collection_name=collection, points=points)

    return len(batch_messages)


async def _get_best_collection(client) -> str:
    """Auto-detect best collection (prefer hybrid if exists)."""
    try:
        collections = await client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if "conversations_hybrid" in collection_names:
            return "conversations_hybrid"
        return "conversations"
    except Exception:
        return "conversations"


async def import_conversations_async(
    async_db,  # AsyncQdrantDB instance
    project_path: str | None = None,
    session_file: Path | None = None,
    include_meta: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Import Claude Code conversations into Qdrant (ASYNC optimized version).

    PERFORMANCE OPTIMIZATIONS:
    - Removed pre-scan bottleneck (was 60% of time for large collections)
    - Parallel file parsing with ThreadPoolExecutor
    - Async pipelined uploads with larger batches (500 vs 100)
    - Uses Python 3.13's improved asyncio.as_completed()

    Args:
        async_db: AsyncQdrantDB instance (with built-in FastEmbed)
        project_path: Specific project path to import (auto-detect if None)
        session_file: Specific session file to import (takes priority)
        include_meta: Include meta messages
        dry_run: Don't actually import, just show stats

    Returns:
        Stats dict: {"conversations": N, "messages": M, "skipped": K}
    """
    stats = {"conversations": 0, "messages": 0, "skipped": 0}

    # 1. Discover session files
    session_files = []

    if session_file:
        session_files = [session_file]
    elif project_path:
        project_dir = find_project_dir(project_path)
        if project_dir:
            session_files = discover_sessions(project_dir)
            logger.info(f"Found project: {project_dir.name}")
        else:
            logger.error(f"Project not found for path: {project_path}")
            return stats
    else:
        # Import all projects (default behavior)
        all_projects = discover_claude_projects()

        if not all_projects:
            logger.error("No Claude Code projects found in ~/.claude/projects")
            logger.info("Hint: Use --project to specify a specific project path")
            return stats

        console.print(f"\n[cyan]📂 Discovered {len(all_projects)} Claude Code project(s)[/cyan]")

        for project_dir in all_projects:
            project_sessions = discover_sessions(project_dir)
            if project_sessions:
                console.print(
                    f"  • {project_dir.name}: {len(project_sessions)} sessions", highlight=False
                )
            session_files.extend(project_sessions)

    if not session_files:
        logger.warning("No session files found")
        return stats

    console.print(f"\n[cyan]🔍 Parsing {len(session_files)} session files...[/cyan]")

    # 2. OPTIMIZATION: Parallel file parsing (GIL doesn't block I/O)
    all_messages = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Parsing...", total=len(session_files))

        # Parse files in parallel (I/O-bound, GIL doesn't matter)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(parse_session_file, sf, include_meta): sf for sf in session_files
            }

            for future in futures:
                messages = future.result()
                if messages:
                    all_messages.extend(messages)
                    console.print(
                        f"  [green]✓[/green] {futures[future].name}: {len(messages)} messages"
                    )
                progress.update(task, advance=1)

    stats["conversations"] = len(session_files)
    stats["messages"] = len(all_messages)

    if not all_messages:
        console.print("\n[yellow]No messages found[/yellow]\n")
        return stats

    # 3. Auto-detect best collection (prefer hybrid)
    collection = await _get_best_collection(async_db.client)
    if collection == "conversations_hybrid":
        console.print("[cyan]Using hybrid search collection (conversations_hybrid)[/cyan]")

    # 4. Get existing messages for incremental import
    console.print()
    with console.status("[yellow]Checking existing messages in Qdrant...[/yellow]", spinner="dots"):
        existing_ids = await get_existing_message_ids_async(async_db, collection)

    if existing_ids:
        console.print(f"[dim]   Found {len(existing_ids):,} existing messages (will skip)[/dim]")

    # 5. Filter to only new messages
    new_messages = [msg for msg in all_messages if msg["message_id"] not in existing_ids]
    stats["skipped"] = len(all_messages) - len(new_messages)

    if dry_run:
        logger.info(
            f"Dry run: Would import {len(new_messages)} new messages (skip {stats['skipped']} existing)"
        )
        return stats

    # Show summary before import
    console.print()
    console.print(f"[cyan]📊 Found {len(all_messages):,} total messages[/cyan]")
    console.print(f"  • [green]{len(new_messages):,} new[/green] (to import)")
    console.print(f"  • [dim]{stats['skipped']:,} already imported[/dim] (skipped)")

    if not new_messages:
        console.print("\n[green]✓ All messages already imported (no new content)[/green]\n")
        return stats

    # Ensure collection exists (auto-create if needed)
    try:
        await async_db.client.get_collection(collection)
    except Exception:
        # Collection doesn't exist, create it (dense-only for backwards compatibility)
        console.print(f"[yellow]Creating {collection} collection...[/yellow]")
        from qdrant_client.models import Distance, VectorParams

        await async_db.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        console.print("[green]✓ Collection created[/green]")

    console.print(f"\n[cyan]📥 Importing {len(new_messages):,} new messages...[/cyan]")

    # 5. Async upload with batching
    # Batch size: 100 messages - good balance of progress updates vs API overhead
    batch_size = 100
    batches = []
    for i in range(0, len(new_messages), batch_size):
        batches.append(new_messages[i : i + batch_size])

    start_time = time.time()
    messages_uploaded = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("•"),
            TextColumn("{task.fields[rate]}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            progress_task = progress.add_task(
                "[cyan]Embedding & uploading...",
                total=len(new_messages),
                rate="0 msg/s",
            )

            # Upload batches sequentially with smooth progress updates
            # (Embedding is CPU-bound with GIL, so async concurrency doesn't help)
            for batch in batches:
                count = await _upload_batch_async(
                    async_db.client,
                    collection,
                    batch,
                    async_db.embedding_model,
                    async_db.sparse_model,
                )
                messages_uploaded += count

                # Update progress after each batch completes
                elapsed = time.time() - start_time
                rate = messages_uploaded / elapsed if elapsed > 0 else 0

                progress.update(
                    progress_task,
                    completed=messages_uploaded,
                    rate=f"{rate:.1f} msg/s",
                )

        elapsed = time.time() - start_time
        rate = messages_uploaded / elapsed if elapsed > 0 else 0

        # Summary table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("[green]✓ Import Complete[/green]", "")
        table.add_row("New messages imported:", f"{messages_uploaded:,}")
        table.add_row("Already existed (skipped):", f"{stats['skipped']:,}")
        table.add_row("Sessions processed:", f"{len(session_files)}")
        table.add_row("Total time:", f"{elapsed:.1f}s ({rate:.1f} msg/sec)")

        console.print()
        console.print(Panel(table, border_style="green", padding=(0, 2)))

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        logger.info("Note: Partial data may have been uploaded. Re-run to continue.")
        raise

    return stats
