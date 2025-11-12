# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Claude KB seriously. If you discover a security vulnerability, please follow these steps:

### Reporting Process

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. Email security reports to: misha@kolesnik.io or open a private security advisory on GitHub
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Suggested fix (if available)

### What to Expect

- **Initial Response**: Within 48 hours
- **Status Updates**: Every 5-7 days until resolved
- **Resolution Timeline**: Security fixes will be prioritized and released as soon as possible

### Disclosure Policy

- We will work with you to understand and resolve the issue promptly
- We request that you do not publicly disclose the vulnerability until we've had a chance to address it
- We will credit you in the fix announcement (unless you prefer to remain anonymous)

## Security Best Practices

### Environment Configuration

**NEVER commit sensitive files:**
- `.env` - Contains API keys and tokens
- `qdrant_storage/` - Contains your conversation data
- `.claude/settings.local.json` - Contains local settings

These files are already excluded in `.gitignore`, but always verify before pushing:
```bash
git status
```

### API Keys and Tokens

- Store all credentials in `.env` file (use `.env.example` as template)
- Never hardcode API keys in source code
- Use environment variables: `QDRANT_API_KEY`, `HF_TOKEN`
- Rotate keys if accidentally exposed

### Local Deployment

- Run Qdrant locally with Docker Compose (default: `http://localhost:6333`)
- For production deployments with Qdrant Cloud, always use API keys
- Never expose Qdrant port (6333) to the public internet without authentication

### Data Privacy

Claude KB imports and stores your Claude Code conversation history:
- All data is stored locally in `qdrant_storage/` by default
- This directory is gitignored and should never be committed
- When backing up, ensure conversation data is encrypted
- Be aware of what conversations you're importing if using shared Qdrant instances

## Security Features

- ✅ No external API calls without explicit configuration
- ✅ Local embeddings with sentence-transformers (no data sent to third parties)
- ✅ Environment variable-based configuration
- ✅ Comprehensive .gitignore for sensitive files
- ✅ Pre-commit hooks to prevent accidental secret commits

## Vulnerability Scope

**In Scope:**
- Authentication bypass
- Data leakage
- Command injection
- Arbitrary code execution
- Dependency vulnerabilities

**Out of Scope:**
- Social engineering attacks
- Physical security
- DOS/DDOS attacks on local services
- Issues in third-party dependencies (report to upstream)

## Dependencies

We regularly update dependencies to address known vulnerabilities:
- Run `uv sync` to update to latest compatible versions
- Check for security advisories: `pip-audit` or GitHub Dependabot

## Contact

- Security issues: misha@kolesnik.io
- General questions: GitHub Issues
- Project maintainer: [@tenequm](https://github.com/tenequm)
