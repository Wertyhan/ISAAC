"""API Configuration."""

from functools import lru_cache

from pydantic import Field

from isaac_core.config import BaseConfig


class Settings(BaseConfig):
    """API settings - inherits from BaseConfig with additional API-specific fields."""

    default_search_k: int = Field(default=10, ge=1, le=100)
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
