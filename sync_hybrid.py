#!/usr/bin/env python3
"""Sync missing messages from conversations to conversations_hybrid with sparse vectors."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qdrant_client import models
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from claude_kb.db import AsyncQdrantDB, QdrantDB

console = Console()

# Config
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


async def sync_collections():
    """Find and migrate missing messages to hybrid collection."""

    console.print("[bold cyan]=== Syncing Missing Messages to Hybrid Collection ===[/]")
    console.print()

    # Initialize clients
    db = QdrantDB(QDRANT_URL, None, EMBEDDING_MODEL)
    async_db = AsyncQdrantDB(QDRANT_URL, None, EMBEDDING_MODEL)

    # Load sparse model
    console.print("Loading sparse embedding model...")
    db.sparse_model.load()
    console.print("[green]✓ Sparse model ready[/]")
    console.print()

    # Get all IDs from both collections
    console.print("Fetching message IDs from both collections...")

    conversations_ids = set()
    hybrid_ids = set()

    # Scroll conversations
    offset = None
    while True:
        results = db.client.scroll(
            collection_name="conversations",
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        points, offset = results
        if not points:
            break
        conversations_ids.update(p.id for p in points)
        if offset is None:
            break

    console.print(f"  conversations: {len(conversations_ids):,} IDs")

    # Scroll hybrid
    offset = None
    while True:
        results = db.client.scroll(
            collection_name="conversations_hybrid",
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        points, offset = results
        if not points:
            break
        hybrid_ids.update(p.id for p in points)
        if offset is None:
            break

    console.print(f"  conversations_hybrid: {len(hybrid_ids):,} IDs")
    console.print()

    # Find missing IDs
    missing_ids = conversations_ids - hybrid_ids

    if not missing_ids:
        console.print("[green]✓ No missing messages - collections are in sync![/]")
        return

    console.print(f"[yellow]Found {len(missing_ids):,} missing messages[/]")
    console.print()
    console.print(f"[cyan]Migrating {len(missing_ids):,} messages to hybrid collection...[/]")
    console.print()

    # Fetch and migrate in batches
    missing_list = list(missing_ids)
    batch_size = 100
    migrated = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Migrating...", total=len(missing_list))

        for i in range(0, len(missing_list), batch_size):
            batch_ids = missing_list[i : i + batch_size]

            # Retrieve points from conversations (with vectors and payload)
            points = db.client.retrieve(
                collection_name="conversations",
                ids=batch_ids,
                with_payload=True,
                with_vectors=True,
            )

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
            for j, point in enumerate(points):
                # Get existing dense vector
                dense_vec = point.vector

                # Get sparse vector
                sparse = sparse_embeddings[j]

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

            # Upsert to hybrid collection
            await async_db.client.upsert(
                collection_name="conversations_hybrid",
                points=new_points,
            )

            migrated += len(points)
            progress.update(task, completed=migrated)

    console.print()
    console.print(f"[bold green]✓ Successfully migrated {migrated:,} messages![/]")
    console.print()

    # Verify
    hybrid_info = db.client.get_collection("conversations_hybrid")
    console.print(f"conversations_hybrid now has: {hybrid_info.points_count:,} points")

    await async_db.close()


if __name__ == "__main__":
    asyncio.run(sync_collections())
