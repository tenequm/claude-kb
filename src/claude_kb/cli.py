"""CLI entry point for Claude KB."""

import asyncio
import logging
import os
import sys
from pathlib import Path

import click
import torch
from dotenv import load_dotenv
from qdrant_client import models
from rich.console import Console

from . import __version__
from .core import AsyncQdrantDB, QdrantDB, format_get_result, format_search_results
from .import_claude import import_conversations_async

# Load environment variables
load_dotenv()

# Enable MPS fallback for Apple Silicon (must be before any torch operations)
if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    logging.getLogger(__name__).debug("MPS detected: Enabled CPU fallback for unsupported ops")

# Configuration
CONFIG = {
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
    schema = """kb/0.1.0

# search - Hybrid semantic+keyword search (85% usage)
kb search "your query here"
kb search "query" --collection conversations --limit 20
kb search "large query" --stream  # MUST use run_in_background=true in Bash tool
out: === id (score: 0.95) ===\\nRole: X\\nTime: Y\\nProject: Z\\n\\nContent...\\n---
exit: 0=found 1=none 2=error
note: When using --stream, Claude Code must invoke Bash tool with run_in_background=true, then poll with BashOutput

# get - Retrieve item by ID
kb get msg_abc123
kb get msg_abc123 --context-window 2000
out: === id ===\\nType: X\\nField: value\\n\\nContent...\\n---
exit: 0=ok 1=notfound 2=error

# import-claude-code-chats - Import Claude Code conversations
kb import-claude-code-chats
kb import-claude-code-chats --project /path --dry-run
out: ✓ Imported N conversations, M messages (duration)
exit: 0=ok 1=invalid_path 2=error

# status - Database statistics
kb status
out: Collections:\\n  conversations    N points\\n  entities         N points
exit: 0=ok 2=error

# ai - This command (LLM-optimized schema)
kb ai
out: <this-format>
exit: 0
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


@main.command()
@click.argument("query")
@click.option("--collection", "-c", default="conversations", help="Collection to search")
@click.option("--limit", "-l", default=10, help="Max results")
@click.option("--stream", "-s", is_flag=True, help="Stream results (background mode)")
@click.option("--filter", "-f", multiple=True, help="Metadata filters (key:value)")
def search(query, collection, limit, stream, filter):
    """Hybrid semantic+keyword search."""
    try:
        # TODO: Implement streaming mode for background execution
        if stream:
            raise NotImplementedError(
                "Streaming mode not yet implemented. Remove --stream flag to use blocking mode."
            )

        # Suppress INFO logs from core module (model loading messages)
        logging.getLogger("claude_kb.core").setLevel(logging.WARNING)

        # Initialize
        db = QdrantDB(CONFIG["qdrant_url"], CONFIG["qdrant_api_key"], CONFIG["embedding_model"])

        # Load embedding model and encode query
        db.embedding_model.load()
        query_vector = db.embedding_model.encode([query], batch_size=1, show_progress=False)[0]

        # Detect vector configuration (named vs default)
        collection_info = db.client.get_collection(collection)
        vectors = collection_info.config.params.vectors

        # Check if vectors is a dict (named vectors) or VectorParams (single default vector)
        if isinstance(vectors, dict):
            vector_names = list(vectors.keys())
        else:
            # Single unnamed vector - use default behavior (don't specify vector_name)
            vector_names = []

        # Search using pre-computed query embedding
        search_params = {
            "collection_name": collection,
            "query_vector": query_vector.tolist(),
            "limit": limit,
        }

        # If collection has named vectors, specify which one to use
        if vector_names:
            search_params["vector_name"] = vector_names[0]

        results = db.client.search(**search_params)

        # Format and output
        if not results:
            console.print("No results found.")
            sys.exit(1)

        output = format_search_results(results, collection)
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
    "--context-window", type=int, help="Truncate content to N tokens (not yet implemented)"
)
@click.option("--collection", "-c", default="conversations", help="Collection to search in")
def get(item_id, context_window, collection):
    """Retrieve item by ID."""
    try:
        # Initialize
        db = QdrantDB(CONFIG["qdrant_url"], CONFIG["qdrant_api_key"], CONFIG["embedding_model"])

        # Retrieve
        point = db.get_by_id(collection, item_id)

        if not point:
            console.print(f"Not found: {item_id}")
            sys.exit(1)

        # TODO: Implement token-aware truncation for context window
        if context_window:
            raise NotImplementedError(
                "Context window truncation not yet implemented. Remove --context-window flag."
            )

        # Format and output
        output = format_get_result(point, collection)
        console.print(output)

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logging.exception("Get failed")
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
