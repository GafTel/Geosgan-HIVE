from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings read from environment variables with prefix HIVE_ (or a .env file)."""

    model_config = SettingsConfigDict(
        env_prefix="HIVE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    model: str = "openai/gpt-4o-mini"
    api_key: str | None = None
    max_steps: int = 8
    max_tool_steps: int = 5
    temperature: float = 0.2
    config_path: Path = Path("config/agents.yaml")
    usage_log: Path | None = None
    default_timeout_seconds: float = 60.0
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 30.0


@lru_cache
def load_settings() -> Settings:
    return Settings()
