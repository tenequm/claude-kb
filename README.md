# Claude KB

Universal knowledge base with Qdrant for Claude Code integration.

Provides semantic search across:
- Claude Code conversation history
- Personal knowledge entities
- Documents and research notes

## Installation

### Run directly (no install)
```bash
uvx claude-kb@latest status
```

### Install as a tool
```bash
uv tool install claude-kb
kb status

# Update to latest version
uv tool upgrade claude-kb
```

### Development
```bash
git clone https://github.com/tenequm/claude-kb.git
cd claude-kb
uv sync --extra dev
```

## Features

- **Hybrid search**: Dense (semantic) + sparse (keyword) vectors with RRF fusion
- **Claude Code import**: Automatically import your conversation history
- **LLM-optimized CLI**: `kb ai` command provides token-efficient schema for AI agents
- **FastEmbed/ONNX**: Fast local embeddings with bge-base-en-v1.5 (768 dim, ~1s search time)
- **Self-hosted**: Run locally with Docker Compose

## Quick Start

```bash
# Start Qdrant
docker compose up -d

# Initialize collections
kb init

# Import your Claude Code conversations
kb import claude-code-chats

# Search!
kb search "qdrant vector databases"
```

## Usage

### Search conversations
```bash
kb search "your query"
kb search "query" --collection conversations --limit 20
```

### Get specific message
```bash
kb get msg_abc123
```

### Check status
```bash
kb status
```

### LLM-optimized schema (for AI agents)
```bash
kb ai
```

This outputs a token-efficient format that Claude Code and other LLMs can parse to understand how to use the CLI. See [docs/AI_COMMAND_SPEC.md](docs/AI_COMMAND_SPEC.md) for details.

## Architecture

- **Simplified structure**: cli.py, core.py, import_claude.py (No manual embedding code!)
- **Qdrant collections**: conversations, entities, documents
- **Embedding**: QdrantClient built-in FastEmbed with BAAI/bge-base-en-v1.5 (768 dim, ONNX-optimized)
- **Search time**: ~1 second total (0.7s model load + 0.3s search)
- **Output format**: Structured plaintext (NOT JSON) optimized for LLM parsing

## Configuration

Create `.env` file (see `.env.example`):
```bash
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5  # FastEmbed model (768 dims, ~1s search)

# Alternative models:
# EMBEDDING_MODEL=BAAI/bge-small-en-v1.5  # Faster (384 dims, ~0.5s)
# EMBEDDING_MODEL=BAAI/bge-large-en-v1.5  # Higher quality (1024 dims, ~2s)
```

## Development

```bash
# Format + lint
ruff format . && ruff check . --fix

# Test (manual for now)
uv run kb --help
```

## Roadmap

- [ ] Streaming search (background mode)
- [ ] Entity management (`kb add entity`)
- [ ] Document import (`kb add document`)
- [ ] Relationship traversal (`kb related`)
- [ ] Full hybrid search (sparse vectors)
- [ ] Token-aware context window truncation

## License

MIT
