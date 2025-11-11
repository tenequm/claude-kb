# AI Command Format Specification v0.1

## Purpose

Provide LLM/AI agents with token-efficient, example-driven CLI documentation that enables accurate tool usage without parsing verbose `--help` text.

## Problem

Traditional CLI documentation (`--help`, man pages) is optimized for human readers:
- Verbose descriptions with full sentences
- Abstract syntax notation (`[OPTIONS]`, `<REQUIRED>`)
- Scattered information across multiple sections
- **~100 tokens per command** when parsed by LLMs

This creates problems for AI agents:
- High token usage for every tool invocation
- Ambiguous parsing of syntax notation
- Unclear output format (what will the command print?)
- Unclear error semantics (when to retry vs give up?)

## Solution

A standardized, example-driven format that shows:
1. **Real invocations** (copy-paste ready)
2. **Output structure** (delimiters, format)
3. **Exit codes** (semantic meanings)
4. **Usage frequency** (guide LLM prioritization)

## Convention

Any CLI tool should provide:
```bash
<tool> ai
```

This outputs a standardized, machine-optimized format that LLMs can parse in **~15 tokens per command** (85% reduction).

## Format Structure

```
<tool>/<version>

# <command> - <description> [(<usage-frequency>)]
<example1>
<example2> [--flag]  # inline comment for clarification
out: <output-format-description-with-literal-delimiters>
exit: <code>=<meaning> <code>=<meaning>

# <next-command> ...
```

### Elements

1. **Header**: `tool/version`
   - Tracks format stability
   - LLMs can cache based on version
   - Example: `kb/0.1.0`

2. **Command block** for each command:
   - **Command line**: `# cmd - description [(...)]`
     - Brief one-line purpose
     - Optional usage frequency hint: `(85% usage)`, `(rare)`
   - **Examples**: 2-4 real invocations showing common patterns
     - NO abstract syntax (`[OPTIONS]`)
     - Actual commands users/LLMs would type
     - Inline comments for flags that need context
   - **Output format**: `out: ...`
     - Show literal delimiters (use `\n` for newlines)
     - Show structure markers (`===`, `---`, etc.)
     - Give LLMs exact parsing targets
   - **Exit codes**: `exit: N=meaning N=meaning`
     - Semantic error categories
     - Guide retry logic (0=success, 1=user_error, 2=system_error)

### Design Principles

- **Example-driven**: Show actual usage, not abstract syntax
- **Token-efficient**: Target ~15 tokens/command vs ~100 for --help
- **Output-aware**: LLMs see exact structure to parse
- **Error-aware**: Exit codes guide retry/escalation logic
- **Human-readable**: Still works as fallback documentation
- **Version-stable**: Breaking changes require version bump

## Example: kb CLI

```
kb/0.1.0

# search - Hybrid semantic+keyword search (85% usage)
kb search "your query here"
kb search "query" --collection conversations --limit 20
kb search "large query" --stream  # background mode, poll with BashOutput
out: === id (score: 0.95) ===\nRole: X\nTime: Y\nProject: Z\n\nContent...\n---
exit: 0=found 1=none 2=error

# get - Retrieve item by ID
kb get msg_abc123
kb get msg_abc123 --context-window 2000
out: === id ===\nType: X\nField: value\n\nContent...\n---
exit: 0=ok 1=notfound 2=error

# import - Import Claude Code conversations
kb import claude-code-chats
kb import claude-code-chats --project /path --dry-run
out: ✓ Imported N conversations, M messages (duration: Xm Ys)
exit: 0=ok 1=invalid_path 2=error

# init - Initialize Qdrant collections
kb init
out: ✓ Connected to Qdrant\n✓ Created conversations, entities, documents
exit: 0=ok 2=qdrant_error

# status - Database statistics
kb status
out: Collections:\n  conversations    N points\n  entities         N points
exit: 0=ok 2=error

# ai - This command (LLM-optimized schema)
kb ai
out: <this-format>
exit: 0
```

## Token Efficiency Comparison

### Traditional --help output:
```
search - Perform a hybrid semantic and keyword search

Usage: kb search [OPTIONS] QUERY

  Searches the knowledge base using both semantic embeddings and
  keyword matching with BM25. Results are ranked using Reciprocal
  Rank Fusion (RRF) to combine both search strategies.

Arguments:
  QUERY  The search query text [required]

Options:
  -c, --collection TEXT  Collection to search [default: conversations]
  -l, --limit INTEGER    Maximum number of results [default: 10]
  -s, --stream          Enable streaming mode for large result sets
  -f, --filter TEXT     Metadata filters in key:value format
  --help                Show this message and exit.
```
**Token count**: ~120 tokens

### AI format output:
```
# search - Hybrid semantic+keyword search (85% usage)
kb search "your query here"
kb search "query" --collection conversations --limit 20
kb search "large query" --stream
out: === id (score: 0.95) ===\nRole: X\nContent...\n---
exit: 0=found 1=none 2=error
```
**Token count**: ~18 tokens

**Savings: 85%** per command

For a CLI with 10 commands:
- Traditional: ~1,000 tokens
- AI format: ~150 tokens
- **Total savings: 85%**

## Benefits for LLMs

1. **Faster context loading**: 6-7x fewer tokens means faster startup
2. **Clearer parsing**: No ambiguity in syntax notation
3. **Better error handling**: Exit codes guide retry logic
4. **Accurate expectations**: Output format shown literally
5. **Prioritization hints**: Usage frequency guides which commands to try first

## Benefits for Tool Authors

1. **Single source of truth**: One format for both humans and AI
2. **Version control**: Track breaking changes explicitly
3. **Lower support burden**: Fewer LLM misunderstandings
4. **Future-proof**: Easy to extend with new metadata

## Adoption Path

### For Tool Authors

1. Add a new command to your CLI:
   ```python
   @cli.command()
   def ai():
       """Output LLM-optimized command definitions."""
       print("""
   your-tool/1.0.0

   # cmd1 - Description
   your-tool cmd1 arg
   out: format
   exit: 0=ok
   """)
   ```

2. Document your commands in this format
3. Test with an LLM to verify parseability
4. Increment version on breaking changes

### For LLM System Designers

1. Check if tool provides `<tool> ai` command
2. Parse the output to build tool understanding
3. Cache based on `tool/version`
4. Fallback to `--help` if ai command doesn't exist

## Future Extensions

Potential additions for v0.2:

- **Streaming hints**: `stream: yes` for long-running commands
- **Chaining patterns**: Common multi-command workflows
- **Context requirements**: File/dir prerequisites
- **Permissions**: Commands requiring elevated privileges
- **Async markers**: Background/daemon commands
- **Deprecation warnings**: For commands being phased out

Example:
```
# deploy - Deploy application (rare, requires auth)
myapp deploy --env production
stream: yes  # long-running, poll status
requires: authenticated, admin_role
exit: 0=deployed 1=invalid_config 2=auth_fail 3=deploy_fail
```

## Comparison to Alternatives

### vs MCP (Model Context Protocol)
- **MCP**: Full protocol for tool integration (JSON-RPC, schemas, resources)
- **AI format**: Lightweight convention for existing CLIs
- **Use MCP when**: Building new AI-native tools
- **Use AI format when**: Exposing existing CLIs to LLMs

### vs JSON Schema / OpenAPI
- **JSON Schema**: Abstract type definitions
- **AI format**: Concrete examples
- **Advantage**: LLMs excel at pattern matching from examples

### vs Traditional --help
- **--help**: Human-optimized, verbose, abstract syntax
- **AI format**: LLM-optimized, concise, concrete examples
- **Both can coexist**: `--help` for humans, `ai` for LLMs

## Real-World Impact

When Claude Code interacts with `kb`:

**Without AI format:**
1. Call `kb --help` (~500 tokens)
2. Parse abstract syntax
3. Guess output format
4. Hope exit codes are standard

**With AI format:**
1. Call `kb ai` once (~80 tokens)
2. Cache result for session
3. Know exact invocation patterns
4. Parse output accurately

**Result**: Faster, more accurate, more token-efficient tool usage.

## License

This specification is public domain. Use it freely to improve AI-CLI interaction.

## Contributing

This is v0.1 - a starting point for standardization. Feedback welcome:
- What works well for your use case?
- What's missing?
- What should change for v0.2?

Open an issue or PR at: https://github.com/anthropics/claude-code/issues
(Or wherever this spec is eventually hosted)

---

**Author**: Created for the `kb` project (Claude Code knowledge base)
**Date**: November 2025
**Version**: 0.1 (Draft)
