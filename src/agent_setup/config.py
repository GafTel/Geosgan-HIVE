from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración leída de variables de entorno con prefijo AGENT_ (o de un archivo .env)."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    max_steps: int = 8
    max_tool_steps: int = 5
    temperature: float = 0.2


@lru_cache
def load_settings() -> Settings:
    return Settings()
