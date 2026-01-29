"""Core Configuration"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """PostgreSQL and vector store configuration."""
    
    postgres_connection_string: str = Field(
        default="postgresql+psycopg2://admin:secret@localhost:5433/isaac_vec_db",
    )
    collection_name: str = Field(default="isaac_vectors")
    embedding_model: str = Field(default="models/gemini-embedding-001")
    
    @field_validator("postgres_connection_string")
    @classmethod
    def validate_connection_string(cls, v: str) -> str:
        valid_prefixes = ("postgresql://", "postgresql+psycopg2://", "postgresql+psycopg://")
        if not v.startswith(valid_prefixes):
            raise ValueError(
                f"Invalid PostgreSQL connection string. Must start with one of: {valid_prefixes}"
            )
        return v


class GeminiConfig(BaseSettings):
    """Gemini API configuration."""
    
    gemini_api_key: SecretStr = Field(description="Google Gemini API key")
    
    generation_model: str = Field(default="gemini-2.0-flash")
    vision_model: str = Field(default="gemini-2.0-flash")
    
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=4096, ge=100, le=8192)
    
    vision_max_retries: int = Field(default=3, ge=0, le=10)
    vision_retry_delay: float = Field(default=2.0, ge=0.5)


class PathsConfig(BaseSettings):
    """File system paths configuration."""
    
    raw_data_file: Path = Field(default=Path("data/raw/isaac_raw_data.json"))
    images_dir: Path = Field(default=Path("data/images"))
    
    @field_validator("images_dir")
    @classmethod
    def ensure_images_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


class BaseConfig(DatabaseConfig, GeminiConfig, PathsConfig):
    """
    Complete base configuration combining all config components.
    
    Modules can inherit from this or compose specific configs as needed.
    Environment variables take precedence over defaults.
    """
    
    top_k_chunks: int = Field(default=5, ge=1, le=20)
    chat_history_limit: int = Field(default=6, ge=0, le=20)
    
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    
    request_timeout: float = Field(default=30.0, ge=5.0)
    download_max_retries: int = Field(default=3, ge=0, le=10)
    
    debug: bool = Field(default=False)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
    }


_base_config: Optional[BaseConfig] = None


@lru_cache
def get_base_config() -> BaseConfig:
    """Get or create base configuration singleton."""
    global _base_config
    if _base_config is None:
        _base_config = BaseConfig()
    return _base_config
