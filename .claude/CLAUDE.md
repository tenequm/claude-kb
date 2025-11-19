# claude-kb Project Instructions

## Commit Format

Always use Conventional Commits format:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code change that neither fixes a bug nor adds a feature
- `chore:` maintenance tasks (deps, build, etc.)
- `test:` adding or updating tests

Examples:
```
feat: add streaming search mode
fix: handle empty query gracefully
docs: update installation instructions
chore: bump version to 0.2.0
```

## Release Process

After each release:

1. **Update CHANGELOG.md**
   - Move items from `[Unreleased]` to new version section
   - Add release date
   - Update comparison links at bottom of file

2. **Create GitHub Release**
   - Use `gh release create vX.Y.Z --title "vX.Y.Z" --notes-file CHANGELOG.md`
   - Or create via GitHub UI with changelog content
   - Tag should match version in pyproject.toml

3. **Publish to PyPI**
   ```bash
   uv build
   uv publish --token <PYPI_TOKEN>
   ```

## Version Bumping

Use uv to bump versions:
```bash
uv version --bump patch  # 0.1.0 -> 0.1.1
uv version --bump minor  # 0.1.0 -> 0.2.0
uv version --bump major  # 0.1.0 -> 1.0.0
```

## Development

- Run `uv sync --extra dev` for development dependencies
- Run `just check` for linting and formatting
- Run `kb ai` before using kb commands to see AI-optimized help
