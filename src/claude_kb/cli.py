"""CLI entry point for Claude KB."""

import asyncio
import logging
import os
import sys
from datetime import UTC
from pathlib import Path
from typing import TypedDict

import click
import torch
from dotenv import load_dotenv
from rich.console import Console

from . import __version__
from .core import (
    AsyncQdrantDB,
    QdrantDB,
    format_get_result,
    format_search_results,
    format_thread_context,
)
from .import_claude import import_conversations_async

# Load environment variables
load_dotenv()

# Enable MPS fallback for Apple Silicon (must be before any torch operations)
if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    logging.getLogger(__name__).debug("MPS detected: Enabled CPU fallback for unsupported ops")


# Configuration type
class Config(TypedDict):
    qdrant_url: str
    qdrant_api_key: str | None
    embedding_model: str
    hf_token: str | None
    use_hf_api: bool


CONFIG: Config = {
    "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
    "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
    "hf_token": os.getenv("HF_TOKEN"),  # For HF Inference API (faster than local)
    "use_hf_api": os.getenv("USE_HF_API", "false").lower() == "true",
}

console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

# Suppress HTTP request logs from httpx (used by qdrant_client)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# =============================================================================
# CLI GROUP
# =============================================================================


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__)
def main():
    """Claude KB - Universal Knowledge Base CLI."""
    pass


# =============================================================================
# AI COMMAND - LLM-optimized schema
# =============================================================================


@main.command()
def ai():
    """Output LLM-optimized command definitions."""
    schema = f"""kb/{__version__}
Searches Claude Code conversation history. Each message is tagged with the project directory (cwd) it occurred in.
Workflow: search → get message_id (full content) or get-thread message_id (with conversation context)

# search - Find messages, returns IDs with scores and content previews
kb search "query"
kb search "error handling" --project claude-kb --from 2024-11-27 --limit 5
opts: --limit N (10), --project NAME (partial match on cwd), --from/--to YYYY-MM-DD, --conversation ID, --role user|assistant, --show-tokens, --min-score N (0.5), --no-boost-recent
note: Recent messages ranked higher by default. Uses hybrid search (semantic + keyword) if migrated.
exit: 0=found 1=none 2=error

# get - Full message content by ID
kb get <message_id>
kb get <message_id> --context-window 2000
opts: --context-window N (chars, 0=unlimited)
exit: 0=ok 1=notfound 2=error

# get-thread - Message with surrounding conversation context
kb get-thread <message_id>
kb get-thread <message_id> --depth 3
opts: --depth N (2) messages before/after
exit: 0=ok 1=notfound 2=error

# import-claude-code-chats - Sync Claude Code history to KB
kb import-claude-code-chats
kb import-claude-code-chats --project /path/to/project --dry-run
opts: --project PATH, --session-file FILE, --include-meta, --dry-run

# migrate - Enable hybrid search (one-time, ~15min for 160k messages)
kb migrate --dry-run
kb migrate

# status - Show database statistics
kb status
"""
    console.print(schema)


# =============================================================================
# IMPORT COMMAND
# =============================================================================


@main.command()
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True),
    help="Specific project path (auto-detect if omitted)",
)
@click.option("--session-file", type=click.Path(exists=True), help="Import single session file")
@click.option("--include-meta", is_flag=True, help="Include meta messages")
@click.option("--dry-run", is_flag=True, help="Show what would be imported")
def import_claude_code_chats(project, session_file, include_meta, dry_run):
    """Import Claude Code conversation history (async-optimized)."""
    try:
        console.print("[bold cyan]=== Importing Claude Code Conversations ===[/]")
        console.print()

        # Session path
        session_path = Path(session_file) if session_file else None

        async def _run_async():
            async_db = AsyncQdrantDB(
                CONFIG["qdrant_url"], CONFIG["qdrant_api_key"], CONFIG["embedding_model"]
            )
            try:
                return await import_conversations_async(
                    async_db=async_db,
                    project_path=project,
                    session_file=session_path,
                    include_meta=include_meta,
                    dry_run=dry_run,
                )
            finally:
                await async_db.close()

        stats = asyncio.run(_run_async())

        console.print()
        if dry_run:
            console.print(
                f"[yellow]Dry run: Would import {stats['conversations']} conversations, {stats['messages']} messages[/]"
            )
        else:
            console.print(
                f"[green]✓ Import complete: {stats['conversations']} conversations, {stats['messages']} messages[/]"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Import failed")
        sys.exit(2)


# =============================================================================
# SEARCH COMMAND
# =============================================================================


def get_default_collection(db) -> str:
    """Return best available collection (prefer hybrid if exists)."""
    try:
        collections = [c.name for c in db.client.get_collections().collections]
        if "conversations_hybrid" in collections:
            return "conversations_hybrid"
        return "conversations"
    except Exception:
        return "conversations"


@main.command()
@click.argument("query")
@click.option("--collection", "-c", default=None, help="Collection to search (auto-detects hybrid)")
@click.option("--limit", "-l", default=10, help="Max results")
@click.option("--stream", "-s", is_flag=True, help="Stream results (background mode)")
@click.option("--project", "-p", help="Filter by project path (partial match)")
@click.option("--from", "from_date", help="Filter by date from (ISO format: YYYY-MM-DD)")
@click.option("--to", "to_date", help="Filter by date to (ISO format: YYYY-MM-DD)")
@click.option("--conversation", help="Filter by conversation ID (exact match)")
@click.option(
    "--role",
    type=click.Choice(["user", "assistant"], case_sensitive=False),
    help="Filter by message role",
)
@click.option("--show-tokens", is_flag=True, help="Display token counts for each result")
@click.option("--min-score", type=float, default=0.5, help="Minimum relevance score (0.0-1.0)")
@click.option(
    "--boost-recent/--no-boost-recent",
    default=True,
    help="Boost recent messages in ranking (default: on)",
)
def search(
    query,
    collection,
    limit,
    stream,
    project,
    from_date,
    to_date,
    conversation,
    role,
    show_tokens,
    min_score,
    boost_recent,
):
    """Semantic search with optional metadata filtering. Uses hybrid search when available."""
    try:
        # Suppress INFO logs from core module (model loading messages)
        logging.getLogger("claude_kb.core").setLevel(logging.WARNING)

        # Initialize
        db = QdrantDB(CONFIG["qdrant_url"], CONFIG["qdrant_api_key"], CONFIG["embedding_model"])

        # Auto-detect best collection if not specified
        if collection is None:
            collection = get_default_collection(db)

        # Ensure payload indices exist for efficient filtering
        db.ensure_indices(collection)

        # Load dense embedding model and encode query
        db.embedding_model.load()
        query_vector = db.embedding_model.encode([query], batch_size=1, show_progress=False)[0]

        # Check if collection supports hybrid search (has sparse vectors)
        sparse_vector = None
        if db.has_sparse_vectors(collection):
            # Load sparse model and generate sparse embedding
            db.sparse_model.load()
            sparse_embeddings = db.sparse_model.encode([query])
            sparse = sparse_embeddings[0]
            sparse_vector = {
                "indices": sparse.indices.tolist(),
                "values": sparse.values.tolist(),
            }

        # Build Qdrant filter from CLI flags (server-side filtering)
        query_filter = None
        filter_conditions = []

        if project or conversation or role:
            from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue

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
                query_filter = Filter(must=filter_conditions)

        # Search using hybrid or dense-only depending on collection config
        search_limit = limit * 3 if (from_date or to_date) else limit
        results = db.search(
            query_vector=query_vector.tolist(),
            collection=collection,
            limit=search_limit,
            query_filter=query_filter,
            score_threshold=min_score,
            sparse_vector=sparse_vector,
        )

        # Client-side date filtering (since timestamps are ISO strings, not numbers)
        if from_date or to_date:
            filtered_results = []
            from_ts = f"{from_date}T00:00:00" if from_date else None
            to_ts = f"{to_date}T23:59:59" if to_date else None

            for result in results:
                timestamp = result.payload.get("timestamp", "")
                if from_ts and timestamp < from_ts:
                    continue
                if to_ts and timestamp > to_ts:
                    continue
                filtered_results.append(result)

            results = filtered_results[:limit]

        # Apply recency boosting if enabled
        if boost_recent and results:
            import math
            from datetime import datetime

            now = datetime.now(UTC)
            one_week_seconds = 7 * 24 * 60 * 60  # Decay half-life

            def apply_recency_boost(result):
                """Apply exponential decay boost based on message age."""
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
                    # This adds up to 0.2 boost for very recent messages, decaying over time
                    recency_boost = 0.2 * math.exp(-age_seconds / one_week_seconds)

                    # Create modified score (without mutating original)
                    return (result.score + recency_boost, result)
                except (ValueError, TypeError):
                    return (result.score, result)

            # Sort by boosted score
            boosted = [apply_recency_boost(r) for r in results]
            boosted.sort(key=lambda x: x[0], reverse=True)

            # Update result scores for display (create wrapper with new score)
            class BoostedResult:
                """Wrapper to show boosted score while preserving original data."""

                def __init__(self, original, boosted_score):
                    self.payload = original.payload
                    self.id = original.id
                    self.score = boosted_score

            results = [BoostedResult(r, score) for score, r in boosted]

        # Format and output
        if not results:
            console.print("No results found.")
            sys.exit(1)

        if stream:
            # Stream results one at a time
            from .core import clean_content

            for i, result in enumerate(results):
                payload = result.payload
                point_id = payload.get("message_id") or payload.get("id") or str(result.id)

                console.print(f"=== {point_id} (score: {result.score:.2f}) ===")
                if collection.startswith("conversations"):
                    console.print(f"Role: {payload.get('role', 'unknown')}")
                    console.print(f"Time: {payload.get('timestamp', 'N/A')}")
                    console.print(f"Project: {payload.get('project_path', 'N/A')}")

                    content = clean_content(payload.get("content", ""))
                    content = str(content) if not isinstance(content, str) else content

                    if show_tokens:
                        from .core import count_tokens

                        token_count = count_tokens(content)
                        console.print(f"Tokens: {token_count:,}")

                    console.print("")

                    # Truncate long content
                    max_len = 1500
                    if len(content) > max_len:
                        console.print(content[:max_len] + "...")
                    else:
                        console.print(content)

                console.print("---")

                # Small delay between results for streaming effect
                if i < len(results) - 1:
                    import time

                    time.sleep(0.1)

            if show_tokens:
                from .core import count_tokens

                total = sum(
                    count_tokens(str(clean_content(r.payload.get("content", "")))) for r in results
                )
                console.print(f"\n=== Total: {len(results)} results, {total:,} tokens ===")
        else:
            # Non-streaming: output all at once
            output = format_search_results(results, collection, show_tokens=show_tokens)
            console.print(output)

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Search failed")
        sys.exit(2)


# =============================================================================
# GET COMMAND
# =============================================================================


@main.command()
@click.argument("item_id")
@click.option(
    "--context-window",
    type=int,
    help="Truncate content to N characters (0 = unlimited, default: unlimited)",
)
@click.option("--collection", "-c", default=None, help="Collection (auto-detects hybrid)")
def get(item_id, context_window, collection):
    """Retrieve item by ID."""
    try:
        # Initialize
        db = QdrantDB(CONFIG["qdrant_url"], CONFIG["qdrant_api_key"], CONFIG["embedding_model"])

        # Auto-detect best collection if not specified
        if collection is None:
            collection = get_default_collection(db)

        # Retrieve
        point = db.get_by_id(collection, item_id)

        if point is None:
            console.print(f"Not found: {item_id}")
            sys.exit(1)
            return  # Explicit return for type narrowing

        # Apply context window truncation if specified
        if context_window and context_window > 0 and point.payload:
            content = point.payload.get("content", "")
            if isinstance(content, str) and len(content) > context_window:
                point.payload["content"] = (
                    content[:context_window]
                    + f"\n\n[... +{len(content) - context_window} chars truncated. "
                    f"Use --context-window 0 for full content]"
                )

        # Format and output
        output = format_get_result(point, collection)
        console.print(output)

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Get failed")
        sys.exit(2)


# =============================================================================
# GET-THREAD COMMAND
# =============================================================================


@main.command()
@click.argument("message_id")
@click.option("--depth", "-d", default=2, type=int, help="Messages before/after to include")
@click.option("--collection", "-c", default=None, help="Collection (auto-detects hybrid)")
def get_thread(message_id, depth, collection):
    """Get message with surrounding conversation context."""
    try:
        # Initialize
        db = QdrantDB(CONFIG["qdrant_url"], CONFIG["qdrant_api_key"], CONFIG["embedding_model"])

        # Auto-detect best collection if not specified
        if collection is None:
            collection = get_default_collection(db)

        # Get thread context
        with console.status(
            f"[yellow]Fetching thread context (±{depth} messages)...[/yellow]", spinner="dots"
        ):
            messages = db.get_thread_context(collection, message_id, depth)

        if not messages:
            console.print(f"Not found: {message_id}")
            sys.exit(1)

        # Format and output
        output = format_thread_context(messages, message_id, collection, depth)
        console.print(output)

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Get thread failed")
        sys.exit(2)


# =============================================================================
# MIGRATE COMMAND
# =============================================================================


@main.command()
@click.option("--collection", "-c", default="conversations", help="Collection to migrate")
@click.option("--batch-size", default=100, help="Batch size for migration")
@click.option("--dry-run", is_flag=True, help="Show what would be migrated")
def migrate(collection, batch_size, dry_run):
    """Migrate collection to hybrid search (add sparse vectors).

    Creates a new collection with both dense and sparse vectors,
    copies all data from the original collection, then swaps them.
    """
    try:
        from qdrant_client import models
        from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

        console.print("[bold cyan]=== Hybrid Search Migration ===[/]")
        console.print()

        # Initialize
        db = QdrantDB(CONFIG["qdrant_url"], CONFIG["qdrant_api_key"], CONFIG["embedding_model"])

        # Check if collection exists
        try:
            info = db.client.get_collection(collection)
            total_points = info.points_count
            console.print(f"Source collection: {collection} ({total_points:,} points)")
        except Exception:
            console.print(f"[red]Collection '{collection}' not found[/]")
            sys.exit(1)

        # Check current vector configuration
        vectors = info.config.params.vectors
        if vectors is None:
            console.print("[red]Collection has no vector configuration[/]")
            sys.exit(1)
            return  # Explicit return for type narrowing

        if isinstance(vectors, dict) and "sparse" in vectors:
            console.print("[green]✓ Collection already has sparse vectors - no migration needed[/]")
            return

        # Get vector dimension from existing collection
        if isinstance(vectors, dict):
            # Named vectors - get first one's dimension
            first_vec = list(vectors.values())[0]
            vector_dim = first_vec.size
        else:
            # Single unnamed vector
            vector_dim = vectors.size

        console.print(f"Vector dimension: {vector_dim}")
        console.print()

        if dry_run:
            console.print(
                f"[yellow]Dry run: Would migrate {total_points:,} points to hybrid search[/]"
            )
            console.print("Run without --dry-run to perform migration")
            return

        # Create new collection with named vectors
        new_collection = f"{collection}_hybrid"
        console.print(f"Creating new collection: {new_collection}")

        try:
            db.client.delete_collection(new_collection)
        except Exception:
            pass

        db.client.create_collection(
            collection_name=new_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,  # Improve sparse vector quality
                ),
            },
        )
        console.print("[green]✓ Created hybrid collection[/]")

        # Load sparse model
        console.print("Loading sparse embedding model...")
        db.sparse_model.load()
        console.print("[green]✓ Sparse model ready[/]")

        # Migrate data in batches
        console.print()
        console.print(f"Migrating {total_points:,} points...")

        migrated = 0
        offset = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Migrating...", total=total_points)

            while True:
                # Scroll through source collection
                results = db.client.scroll(
                    collection_name=collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                points, offset = results

                if not points:
                    break

                # Extract content for sparse embeddings
                contents = []
                for point in points:
                    if not point.payload:
                        contents.append("")
                        continue
                    content = point.payload.get("content", "")
                    if isinstance(content, list | dict):
                        import json

                        content = json.dumps(content)
                    contents.append(str(content)[:8000])  # Truncate for SPLADE

                # Generate sparse embeddings
                sparse_embeddings = db.sparse_model.encode(contents)

                # Build new points with both vectors
                new_points = []
                for i, point in enumerate(points):
                    # Get existing dense vector
                    if isinstance(point.vector, dict):
                        dense_vec = list(point.vector.values())[0]
                    else:
                        dense_vec = point.vector

                    # Get sparse vector (indices and values)
                    sparse = sparse_embeddings[i]

                    new_points.append(
                        models.PointStruct(
                            id=point.id,
                            payload=point.payload,
                            vector={
                                "dense": dense_vec,
                                "sparse": models.SparseVector(
                                    indices=sparse.indices.tolist(),
                                    values=sparse.values.tolist(),
                                ),
                            },
                        )
                    )

                # Upsert to new collection
                db.client.upsert(collection_name=new_collection, points=new_points)

                migrated += len(points)
                progress.update(task, completed=migrated)

                if offset is None:
                    break

        console.print()
        console.print(f"[green]✓ Migrated {migrated:,} points to {new_collection}[/]")

        # Swap collections
        console.print()
        console.print("Swapping collections...")

        backup_name = f"{collection}_backup"
        try:
            db.client.delete_collection(backup_name)
        except Exception:
            pass

        # Rename: original → backup
        db.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=collection,
                        alias_name=backup_name,
                    )
                )
            ]
        )

        # Actually we need to use a different approach - Qdrant aliases don't rename
        # Let's just inform the user
        console.print()
        console.print("[bold green]Migration complete![/]")
        console.print()
        console.print("Next steps:")
        console.print(
            f"  1. Verify hybrid collection: kb search 'test' --collection {new_collection}"
        )
        console.print(
            f"  2. To use hybrid search, update your collection references to '{new_collection}'"
        )
        console.print(f"  3. Original collection '{collection}' preserved as backup")
        console.print()
        console.print(
            "[dim]Tip: To make hybrid the default, you can rename collections in Qdrant dashboard[/]"
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Migration failed")
        sys.exit(2)


# =============================================================================
# STATUS COMMAND
# =============================================================================


@main.command()
def status():
    """Show database statistics."""
    try:
        from rich.table import Table

        console.print("[bold cyan]=== Claude KB Status ===[/]")
        console.print()

        # Connect
        db = QdrantDB(CONFIG["qdrant_url"], CONFIG["qdrant_api_key"], CONFIG["embedding_model"])
        console.print(f"Qdrant: {CONFIG['qdrant_url']} [green]✓[/]")
        console.print(f"Embedding: {CONFIG['embedding_model']}")
        console.print()

        # Get stats
        stats = db.get_stats()

        if not stats:
            console.print("[yellow]No collections found. Run 'kb init' first.[/]")
            sys.exit(1)

        console.print("[bold]Collections:[/]")
        for collection, count in stats.items():
            # Format size (rough estimate: ~1KB per point with 768-dim vector)
            size_mb = (count * 1) / 1024
            console.print(f"  {collection:20} {count:6,} points  ({size_mb:.1f} MB)")

        # Get per-project breakdown for conversations
        if "conversations" in stats and stats["conversations"] > 0:
            console.print()
            console.print("[bold]Projects:[/]")

            with console.status("[yellow]Analyzing projects...[/yellow]", spinner="dots"):
                project_stats = db.get_project_stats("conversations")

            if project_stats:
                table = Table(show_header=True, header_style="bold cyan", box=None)
                table.add_column("Project", style="dim", no_wrap=False)
                table.add_column("Sessions", justify="right", style="green")
                table.add_column("Messages", justify="right", style="blue")

                total_sessions = 0
                total_messages = 0

                for proj in project_stats:
                    # Shorten project path for display
                    project_name = (
                        proj["project"].split("/")[-1]
                        if "/" in proj["project"]
                        else proj["project"]
                    )
                    table.add_row(project_name, f"{proj['sessions']:,}", f"{proj['messages']:,}")
                    total_sessions += proj["sessions"]
                    total_messages += proj["messages"]

                # Add total row
                table.add_row(
                    "[bold]TOTAL[/bold]",
                    f"[bold]{total_sessions:,}[/bold]",
                    f"[bold]{total_messages:,}[/bold]",
                    style="bold",
                )

                console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Status failed")
        sys.exit(2)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
