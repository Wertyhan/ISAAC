"""Ingestion Configuration."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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


def generate_doc_id(project_name: str, source_path: str) -> str:
    """Generate stable document ID from project name and source path."""
    content = f"{project_name}:{source_path}"
    hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"DOC_{hash_val}"


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate chunk ID from document ID and chunk index."""
    return f"CHK_{doc_id[4:]}_{chunk_index}"


class ImageMeta:
    """Image metadata with full traceability."""
    __slots__ = ("original_url", "local_path", "file_hash", "project_name", "doc_id", "caption")
    
    def __init__(
        self,
        original_url: str,
        local_path: Path,
        file_hash: str,
        project_name: str,
        doc_id: Optional[str] = None,
        caption: Optional[str] = None,
    ):
        self.original_url = original_url
        self.local_path = local_path
        self.file_hash = file_hash
        self.project_name = project_name
        self.doc_id = doc_id
        self.caption = caption
    
    @property
    def image_id(self) -> str:
        return f"IMG_{self.file_hash[:12]}"
    
    def __repr__(self) -> str:
        return f"ImageMeta(id={self.image_id}, doc={self.doc_id}, path={self.local_path.name})"


class ProcessedProject:
    """Processed project with full metadata for traceability."""
    __slots__ = (
        "doc_id", "project_name", "title", "category", 
        "processed_content", "images", "source_path",
        "source_uri", "created_at", "description"
    )
    
    def __init__(
        self,
        doc_id: str,
        project_name: str,
        title: str,
        category: str,
        processed_content: str,
        images: list["ImageMeta"],
        source_path: str,
        source_uri: str,
        created_at: datetime,
        description: Optional[str] = None,
    ):
        self.doc_id = doc_id
        self.project_name = project_name
        self.title = title
        self.category = category
        self.processed_content = processed_content
        self.images = images
        self.source_path = source_path
        self.source_uri = source_uri
        self.created_at = created_at
        self.description = description
