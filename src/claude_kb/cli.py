"""CLI entry point for Claude KB."""

import asyncio
import logging
import os
import sys
from importlib.metadata import version
from pathlib import Path

import click
import torch

# Load environment variables
from dotenv import load_dotenv
from rich.console import Console

from .config import get_config
from .db import AsyncQdrantDB, QdrantDB
from .formatters import format_get, format_search, format_status, format_thread
from .models import ErrorResult, GetResult, SearchResult, StatusResult
from .search import SearchService

load_dotenv()

# Enable MPS fallback for Apple Silicon (must be before any torch operations)
if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    logging.getLogger(__name__).debug("MPS detected: Enabled CPU fallback for unsupported ops")

__version__ = version("claude-kb")
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

# Suppress HTTP request logs
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

# mcp - Run as MCP server for Claude Code integration
kb mcp
Add to Claude Code: claude mcp add kb -- uv run kb mcp
Exposes: search, get tools
"""
    console.print(schema)


# =============================================================================
# SEARCH COMMAND
# =============================================================================


@main.command()
@click.argument("query")
@click.option("--limit", "-l", default=10, help="Max results")
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
    limit,
    project,
    from_date,
    to_date,
    conversation,
    role,
    show_tokens,
    min_score,
    boost_recent,
):
    """Semantic search with optional metadata filtering."""
    try:
        service = SearchService()
        result = service.search(
            query=query,
            limit=limit,
            project=project,
            from_date=from_date,
            to_date=to_date,
            role=role,
            conversation=conversation,
            min_score=min_score,
            boost_recent=boost_recent,
        )

        if isinstance(result, ErrorResult):
            console.print(f"[red]Error: {result.error}[/]")
            sys.exit(2)

        assert isinstance(result, SearchResult)
        console.print(format_search(result, show_tokens=show_tokens))

        if result.count == 0:
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Search failed")
        sys.exit(2)


# =============================================================================
# GET COMMAND
# =============================================================================


@main.command()
@click.argument("message_id")
@click.option(
    "--context-window",
    type=int,
    help="Truncate content to N characters (0 = unlimited, default: unlimited)",
)
def get(message_id, context_window):
    """Retrieve message by ID."""
    try:
        service = SearchService()
        result = service.get(message_id)

        if isinstance(result, ErrorResult):
            console.print(f"[red]{result.error}[/]")
            sys.exit(1)

        assert isinstance(result, GetResult)
        console.print(format_get(result, context_window=context_window))

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Get failed")
        sys.exit(2)


# =============================================================================
# GET-THREAD COMMAND
# =============================================================================


@main.command("get-thread")
@click.argument("message_id")
@click.option("--depth", "-d", default=2, type=int, help="Messages before/after to include")
def get_thread(message_id, depth):
    """Get message with surrounding conversation context."""
    try:
        service = SearchService()

        with console.status(
            f"[yellow]Fetching thread context (±{depth} messages)...[/yellow]", spinner="dots"
        ):
            result = service.get(message_id, context_depth=depth)

        if isinstance(result, ErrorResult):
            console.print(f"[red]{result.error}[/]")
            sys.exit(1)

        assert isinstance(result, GetResult)
        console.print(format_thread(result))

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Get thread failed")
        sys.exit(2)


# =============================================================================
# STATUS COMMAND
# =============================================================================


@main.command()
def status():
    """Show database statistics."""
    try:
        console.print("[bold cyan]=== Claude KB Status ===[/]")
        console.print()

        service = SearchService()

        with console.status("[yellow]Analyzing projects...[/yellow]", spinner="dots"):
            result = service.status(include_projects=True)

        if isinstance(result, ErrorResult):
            console.print(f"[red]Error: {result.error}[/]")
            sys.exit(2)

        assert isinstance(result, StatusResult)
        console.print(format_status(result))

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Status failed")
        sys.exit(2)


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
        from .import_claude import import_conversations_async

        console.print("[bold cyan]=== Importing Claude Code Conversations ===[/]")
        console.print()

        # Session path
        session_path = Path(session_file) if session_file else None
        config = get_config()

        async def _run_async():
            async_db = AsyncQdrantDB(
                config.qdrant_url, config.qdrant_api_key, config.embedding_model
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

        config = get_config()
        db = QdrantDB(config.qdrant_url, config.qdrant_api_key, config.embedding_model)

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

        if isinstance(vectors, dict) and "sparse" in vectors:
            console.print("[green]✓ Collection already has sparse vectors - no migration needed[/]")
            return

        # Get vector dimension from existing collection
        from qdrant_client.models import VectorParams

        if isinstance(vectors, dict):
            first_vec = list(vectors.values())[0]
            vector_dim = first_vec.size
        elif isinstance(vectors, VectorParams):
            vector_dim = vectors.size
        else:
            console.print("[red]Unknown vector configuration type[/]")
            sys.exit(1)

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
                    modifier=models.Modifier.IDF,
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
                    contents.append(str(content)[:8000])

                # Generate sparse embeddings
                sparse_embeddings = db.sparse_model.encode(contents)

                # Build new points with both vectors
                new_points = []
                for i, point in enumerate(points):
                    if isinstance(point.vector, dict):
                        dense_vec = list(point.vector.values())[0]
                    else:
                        dense_vec = point.vector

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

                db.client.upsert(collection_name=new_collection, points=new_points)

                migrated += len(points)
                progress.update(task, completed=migrated)

                if offset is None:
                    break

        console.print()
        console.print(f"[green]✓ Migrated {migrated:,} points to {new_collection}[/]")

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

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Migration failed")
        sys.exit(2)


# =============================================================================
# BACKFILL-TIMESTAMPS COMMAND
# =============================================================================


@main.command("backfill-timestamps")
@click.option(
    "--collection", "-c", default=None, help="Collection to update (auto-detect if omitted)"
)
@click.option("--batch-size", default=500, help="Batch size for updates")
@click.option("--dry-run", is_flag=True, help="Show what would be updated")
def backfill_timestamps(collection: str | None, batch_size: int, dry_run: bool):
    """Backfill timestamp_unix field for existing messages.

    This enables server-side date filtering for messages imported
    before the timestamp_unix field was added.

    Examples:

    \b
        # Preview changes
        kb backfill-timestamps --dry-run

    \b
        # Update all messages
        kb backfill-timestamps

    \b
        # Update specific collection
        kb backfill-timestamps --collection conversations_hybrid
    """
    from datetime import UTC, datetime

    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    console.print("[bold cyan]=== Backfill timestamp_unix ===[/]")
    console.print()

    config = get_config()
    db = QdrantDB(config.qdrant_url, config.qdrant_api_key, config.embedding_model)

    # Auto-detect collection
    if collection is None:
        try:
            collections = [c.name for c in db.client.get_collections().collections]
            collection = (
                "conversations_hybrid" if "conversations_hybrid" in collections else "conversations"
            )
        except Exception:
            collection = "conversations"

    # Check if collection exists
    try:
        info = db.client.get_collection(collection)
        total_points = info.points_count
        console.print(f"Collection: {collection} ({total_points:,} points)")
    except Exception:
        console.print(f"[red]Collection '{collection}' not found[/]")
        sys.exit(1)

    if dry_run:
        console.print(f"[yellow]Dry run: Would scan {total_points:,} points[/]")

    # Ensure the new index exists
    console.print("Ensuring timestamp_unix index exists...")
    db.ensure_indices(collection)
    console.print("[green]✓ Index ready[/]")
    console.print()

    # Process in batches
    console.print("Scanning for points missing timestamp_unix...")
    console.print()

    updated = 0
    skipped = 0
    offset = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=total_points)

        while True:
            # Scroll through points
            results = db.client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # Don't need vectors
            )

            points, offset = results

            if not points:
                break

            # Find points missing timestamp_unix
            for point in points:
                payload = point.payload or {}

                if "timestamp_unix" in payload:
                    skipped += 1
                    progress.update(task, completed=updated + skipped)
                    continue

                # Parse timestamp and compute unix timestamp
                timestamp_str = payload.get("timestamp", "")
                try:
                    if "+" in timestamp_str or timestamp_str.endswith("Z"):
                        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    else:
                        ts = datetime.fromisoformat(timestamp_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=UTC)

                    timestamp_unix = int(ts.timestamp())

                    if not dry_run:
                        # Update payload with timestamp_unix
                        db.client.set_payload(
                            collection_name=collection,
                            payload={
                                "timestamp_unix": timestamp_unix,
                                "schema_version": 2,
                            },
                            points=[point.id],
                        )

                    updated += 1

                except (ValueError, TypeError, AttributeError) as e:
                    logging.warning(f"Failed to parse timestamp for point {point.id}: {e}")
                    skipped += 1

                progress.update(task, completed=updated + skipped)

            if offset is None:
                break

    console.print()
    if dry_run:
        console.print("[bold yellow]Dry run complete[/]")
        console.print(f"  Would update: {updated:,} points")
        console.print(f"  Already have timestamp_unix: {skipped:,} points")
    else:
        console.print("[bold green]✓ Backfill complete![/]")
        console.print(f"  Updated: {updated:,} points")
        console.print(f"  Skipped (already had timestamp_unix): {skipped:,} points")


# =============================================================================
# MCP COMMAND
# =============================================================================


@main.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    help="Transport: stdio (default) or http (Streamable HTTP)",
)
@click.option("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
@click.option("--port", default=8000, type=int, help="HTTP port (default: 8000)")
def mcp(transport: str, host: str, port: int):
    """Run as MCP server for Claude Code integration.

    Examples:

    \b
        # stdio transport (default, for Claude Code)
        claude mcp add kb -- uv run kb mcp

    \b
        # Streamable HTTP transport
        uv run kb mcp --transport http

    \b
        # HTTP with custom port
        uv run kb mcp --transport http --port 3000
    """
    from .mcp import _register_tools, create_server

    if transport == "stdio":
        from .mcp import mcp as mcp_server

        mcp_server.run(transport="stdio")
    else:
        server = create_server(host=host, port=port, json_response=True, stateless=True)
        _register_tools(server)
        server.run(transport="streamable-http")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
