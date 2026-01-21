from pathlib import Path
from typing import Optional
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


# Configuration
class Config(BaseSettings):
    
    gemini_api_key: SecretStr
    postgres_connection_string: str = Field(
        default="postgresql+psycopg2://user:pass@localhost:5432/isaac_db"
    )
    
    # Paths
    raw_data_file: Path = Field(default=Path("data/raw/isaac_raw_data.json"))
    images_dir: Path = Field(default=Path("data/images"))
    
    # Vector Store
    collection_name: str = "isaac_vectors"
    embedding_model: str = "models/text-embedding-004"
    
    # Gemini Vision
    vision_model: str = "gemini-2.0-flash"
    vision_max_retries: int = Field(default=3, ge=0, le=10)
    vision_retry_delay: float = Field(default=2.0, ge=0.5)
    
    # Text Processing
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    
    # Network
    request_timeout: float = Field(default=30.0, ge=5.0)
    download_max_retries: int = Field(default=3, ge=0, le=10)
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    @field_validator("raw_data_file")
    @classmethod
    def validate_raw_data_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Raw data file not found: {v}")
        return v
    
    @field_validator("images_dir")
    @classmethod
    def ensure_images_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


# Models
class ImageMeta:
    
    def __init__(
        self,
        original_url: str,
        local_path: Path,
        file_hash: str,
        project_name: str,
    ):
        self.original_url = original_url
        self.local_path = local_path
        self.file_hash = file_hash
        self.project_name = project_name
    
    @property
    def image_id(self) -> str:
        return f"IMG_{self.file_hash[:12]}"
    
    def __repr__(self) -> str:
        return f"ImageMeta(id={self.image_id}, path={self.local_path.name})"


class ProcessedProject:
    
    def __init__(
        self,
        project_name: str,
        category: str,
        processed_content: str,
        images: list["ImageMeta"],
        source_path: str,
    ):
        self.project_name = project_name
        self.category = category
        self.processed_content = processed_content
        self.images = images
        self.source_path = source_path
