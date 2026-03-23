# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.4] - 2026-03-23

### Changed
- Simplified README with single install path: `uv tool install claude-kb` + `claude mcp add -s user kb -- kb mcp`
- Removed alternative install methods (uvx, uv run) to reduce setup confusion

## [0.7.3] - 2026-03-18

### Changed
- Grouped search preview length increased from 200 to 2000 chars for better relevance evaluation

### Fixed
- Conversations with no text preview are filtered from compact grouped search results (zero-signal noise removal)

## [0.7.1] - 2026-03-18

### Fixed
- Grouped search previews now iterate messages to find actual text instead of showing useless `[N content blocks]` for tool-only messages
- `_extract_preview()` returns `None` instead of bracket-string fallbacks, eliminating false previews
- `ConversationSummary.preview` is now optional - dropped from compact output when no text preview is available

## [0.7.0] - 2026-03-18

### Added
- Compact response mode for MCP tools (~29% token reduction, up to 86% on assistant messages)
- `_shorten_project()` helper for consistent project path shortening
- `_is_thinking_only()` helper to detect and filter placeholder-only search results
- 8 new unit tests for compact mode behavior

### Changed
- MCP `kb_search` and `kb_get` now strip non-essential metadata: query echo, collection name, parent_id, path prefixes, excess score precision
- Single text content blocks are unwrapped from JSON array wrapper in MCP responses
- `tool_use` blocks reduced to `{type, name}` when `include_tool_results=False`
- Thinking-only messages are filtered from MCP search results (with 1.5x over-fetch to compensate)
- `SearchResult` and `ConversationSearchResult` envelope fields (`query`, `collection`) are now optional
- CLI output is unchanged (all optimizations are behind `compact=True`, which only MCP uses)

## [0.6.0] - 2026-02-19

### Added
- Conversation restore mode in `kb_get` using `conversation_id` with optional `up_to` and `max_messages`
- `conversation_id` filter in MCP `kb_search` tool arguments
- Test coverage for `SearchService` restore/cleaning helpers and edge-case validation

### Changed
- `kb_get` now treats `message_id` and `conversation_id` as mutually exclusive
- Restore mode now fails fast if `include_tool_results`/`include_thinking` are passed (instead of silently ignoring)
- Conversation restore ordering now sorts by parsed timestamps instead of raw timestamp strings
- Unified conversation scroll logic for restore and message counting

## [0.5.0] - 2026-01-06

### Added
- Server-side date filtering via `timestamp_unix` across search and MCP flows
- `timestamp_unix` support in import/sync pipelines for efficient time-range filtering

### Changed
- Date filters now run in Qdrant payload filtering instead of client-side fallback for new data
- Documentation updated to clarify server-side date filter behavior

## [0.4.0] - 2026-01-05

### Added
- Output content filtering controls (`include_tool_results`, `include_thinking`)
- Conversation-grouped search mode (`group_by_conversation`) with conversation summaries

### Changed
- Search and MCP output now defaults to lightweight content with optional expanded blocks
- Search docs and schemas updated for conversation-level exploration workflow

## [0.3.1] - 2025-12-16

### Changed
- Renamed MCP tools to include `kb_` prefix: `search` → `kb_search`, `get` → `kb_get`

## [0.3.0] - 2025-12-11

### Added
- HTTP transport support for MCP server via `kb mcp --transport http`
- Tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`)
- Server and tool icons (SVG data URIs) for MCP clients
- `Literal["user", "assistant"]` type for role parameter with schema validation
- Server factory function `create_server()` for configurable MCP instantiation
- Click CLI with `--transport`, `--host`, `--port` options for MCP command

### Changed
- MCP server now uses latest protocol version (2025-11-25)
- Improved type safety throughout CLI and database modules

### Fixed
- Type narrowing issues in CLI commands with proper assertions
- Tensor conversion type safety in sparse embedding encoder

## [0.2.0] - 2025-12-09

### Added
- MCP server capability via `kb mcp` command for Claude Code integration
- FastMCP integration with 2 tools (`search`, `get`) and 2 resources (`schema://kb`, `stats://kb`)
- Hybrid search combining dense semantic vectors with sparse BM25-style keyword matching
- Recency boosting with exponential decay for search results
- `kb migrate` command to upgrade existing collections to hybrid search
- Pydantic models for structured outputs shared between CLI and MCP
- Service layer (`SearchService`) for business logic reuse
- Centralized configuration module (`config.py`)
- SparseEncoder for MPS-accelerated sparse embeddings on Apple Silicon

### Changed
- Restructured codebase: split monolithic `core.py` into `db.py`, `search.py`, `formatters.py`, `models.py`, `config.py`
- Slimmed `cli.py` by delegating to service layer
- Version now retrieved via `importlib.metadata` instead of `__init__.py`
- Replaced FastEmbed with SparseEncoder for better MPS compatibility
- Collection detection now uses `startswith` for conversations collection matching

### Removed
- `__init__.py` (no longer needed with modern Python packaging)
- FastEmbed dependency (replaced with sentence-transformers SparseEncoder)

### Fixed
- Collection check now properly detects `conversations_hybrid` collection

### Migration
To enable hybrid search on existing data, run:
```bash
kb migrate --dry-run  # Preview changes
kb migrate            # Run migration (time varies by collection size)
```
This creates a new `conversations_hybrid` collection with both dense and sparse vectors. The original `conversations` collection is preserved as backup.

## [0.1.1] - 2025-11-19

### Fixed
- Added `claude-kb` CLI entry point so `uvx claude-kb` works directly

## [0.1.0] - 2025-11-19

### Added
- Initial release
- Core knowledge base functionality with Qdrant vector database
- Semantic search with sentence-transformers embeddings
- CLI commands: `search`, `get`, `import`, `status`, `ai`
- Claude Code conversation import from `~/.claude/projects/`
- Metadata filtering: `--project`, `--from`, `--to`, `--conversation`, `--role`
- Token counting with `--show-tokens`
- Streaming mode with `--stream`
- Rich terminal output with syntax highlighting
- Apple Silicon MPS support for embeddings
- Docker Compose setup for local Qdrant
- Pre-commit hooks with secret detection

[Unreleased]: https://github.com/tenequm/claude-kb/compare/v0.7.4...HEAD
[0.7.4]: https://github.com/tenequm/claude-kb/compare/v0.7.3...v0.7.4
[0.7.3]: https://github.com/tenequm/claude-kb/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/tenequm/claude-kb/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/tenequm/claude-kb/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/tenequm/claude-kb/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/tenequm/claude-kb/compare/v0.3.1...v0.6.0
[0.3.1]: https://github.com/tenequm/claude-kb/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/tenequm/claude-kb/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/tenequm/claude-kb/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/tenequm/claude-kb/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/tenequm/claude-kb/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/tenequm/claude-kb/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/tenequm/claude-kb/releases/tag/v0.1.0
