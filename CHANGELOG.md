# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/tenequm/claude-kb/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/tenequm/claude-kb/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/tenequm/claude-kb/releases/tag/v0.1.0
