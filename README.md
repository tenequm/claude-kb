# Claude KB

Semantic search across your Claude Code conversation history, powered by Qdrant.

## Setup

### 1. Install

```bash
uv tool install claude-kb
```

### 2. Start Qdrant

```bash
docker compose up -d
```

### 3. Initialize and import

```bash
kb init
kb import claude-code-chats
```

### 4. Add MCP server to Claude Code

```bash
claude mcp add -s user kb -- kb mcp
```

That's it. Claude Code now has access to `kb_search` and `kb_get` tools for searching your conversation history.

### Update

```bash
uv tool upgrade claude-kb
```

To check current version:

```bash
kb --version
```

## CLI Usage

```bash
kb search "your query"                    # semantic search
kb search "query" --limit 20             # with options
kb get <message-id>                       # retrieve specific message
kb get-thread <message-id>                # message with context
kb status                                 # check collections and stats
kb ai                                     # LLM-optimized schema
```

## Configuration

Set `QDRANT_URL` if Qdrant is not on localhost:

```bash
export QDRANT_URL=http://your-host:6333
```

Or create a `.env` file:

```bash
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

## Search Tips

- **Score 0.9+**: exact topic match, **0.7-0.9**: related, **0.5-0.7**: partial, **<0.5**: filtered
- Lower `min_score` to 0.3 for broader exploration
- Use `project` parameter to filter by project (partial match), not the query
- By default, thinking and tool_result content shows as `[thinking: N chars]` placeholders - use `include_thinking=True` / `include_tool_results=True` for full content

## Development

```bash
git clone https://github.com/tenequm/claude-kb.git
cd claude-kb
uv sync --extra dev
just check  # lint + format
```

## License

MIT
