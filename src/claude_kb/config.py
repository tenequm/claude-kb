"""Configuration management for claude-kb."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Application configuration."""

    qdrant_url: str
    qdrant_api_key: str | None
    embedding_model: str


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
    )


_config: Config | None = None


def get_config() -> Config:
    """Get cached configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
