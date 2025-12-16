# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/tenequm/claude-kb/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/tenequm/claude-kb/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/tenequm/claude-kb/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tenequm/claude-kb/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/tenequm/claude-kb/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/tenequm/claude-kb/releases/tag/v0.1.0
