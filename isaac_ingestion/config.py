"""Ingestion Configuration."""

from pathlib import Path

from pydantic import Field, field_validator

from isaac_core.config import BaseConfig


class Config(BaseConfig):
    """Ingestion pipeline configuration - inherits from BaseConfig."""
    
    @field_validator("raw_data_file")
    @classmethod
    def validate_raw_data_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Raw data file not found: {v}")
        return v


class ImageMeta:
    __slots__ = ("original_url", "local_path", "file_hash", "project_name")
    
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
    __slots__ = ("project_name", "category", "processed_content", "images", "source_path")
    
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
