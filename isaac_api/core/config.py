from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    #Application settings from environment variables.

    postgres_connection_string: str = Field(
        default="postgresql+psycopg2://admin:secret@localhost:5433/isaac_vec_db",
        description="PostgreSQL connection string with pgvector extension",
    )

    gemini_api_key: SecretStr = Field(
        description="Google Gemini API key for embeddings",
    )

    collection_name: str = Field(
        default="isaac_vectors",
        description="Name of the vector store collection",
    )
    embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Embedding model identifier",
    )

    default_search_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default number of documents to retrieve (k parameter)",
    )

    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
    }

    @field_validator("postgres_connection_string")
    @classmethod
    def validate_connection_string(cls, v: str) -> str:
        if not v.startswith(("postgresql://", "postgresql+psycopg2://", "postgresql+psycopg://")):
            raise ValueError(
                "Invalid PostgreSQL connection string. "
                "Must start with 'postgresql://' or 'postgresql+psycopg2://'"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
