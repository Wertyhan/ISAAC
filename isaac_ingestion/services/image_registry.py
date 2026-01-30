"""Image Registry - Persistent storage of image metadata for retrieval."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImageRecord(BaseModel):
    """Complete image record with full traceability."""
    
    image_id: str = Field(..., description="Unique image identifier (IMG_xxx)")
    doc_id: str = Field(..., description="Parent document ID")
    project_name: str = Field(..., description="Project/source name")
    
    # Source information
    source_uri: str = Field(..., description="Original image URL")
    filepath: str = Field(..., description="Local file path")
    
    # Content metadata
    caption: Optional[str] = Field(default=None, description="Original caption if available")
    description: str = Field(default="", description="AI-generated description")
    alt_text: Optional[str] = Field(default=None, description="Alt text for accessibility")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Technical metadata
    file_hash: str = Field(..., description="Content hash for deduplication")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    mime_type: Optional[str] = Field(default=None, description="MIME type")
    
    model_config = {"extra": "ignore"}


class ImageRegistry:
    """
    Persistent registry for image metadata.
    
    Enables:
    - Image retrieval by ID, doc_id, or project
    - Caption-based search preparation
    - Full traceability for citations
    """
    
    def __init__(self, registry_path: Path):
        self._registry_path = registry_path
        self._images: Dict[str, ImageRecord] = {}
        self._load()
    
    def _load(self) -> None:
        """Load registry from disk."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for record in data.get("images", []):
                        img = ImageRecord(**record)
                        self._images[img.image_id] = img
                logger.info(f"Loaded {len(self._images)} images from registry")
            except Exception as e:
                logger.warning(f"Failed to load image registry: {e}")
                self._images = {}
    
    def _save(self) -> None:
        """Persist registry to disk."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_images": len(self._images),
            "images": [img.model_dump(mode="json") for img in self._images.values()]
        }
        
        with open(self._registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.debug(f"Saved {len(self._images)} images to registry")
    
    def register(
        self,
        image_id: str,
        doc_id: str,
        project_name: str,
        source_uri: str,
        filepath: str,
        file_hash: str,
        description: str = "",
        caption: Optional[str] = None,
        file_size: Optional[int] = None,
        mime_type: Optional[str] = None,
    ) -> ImageRecord:
        """Register a new image or update existing."""
        record = ImageRecord(
            image_id=image_id,
            doc_id=doc_id,
            project_name=project_name,
            source_uri=source_uri,
            filepath=filepath,
            file_hash=file_hash,
            description=description,
            caption=caption,
            file_size=file_size,
            mime_type=mime_type,
        )
        
        self._images[image_id] = record
        logger.debug(f"Registered image: {image_id} for doc {doc_id}")
        return record
    
    def get(self, image_id: str) -> Optional[ImageRecord]:
        """Get image by ID."""
        return self._images.get(image_id)
    
    def get_by_doc(self, doc_id: str) -> List[ImageRecord]:
        """Get all images for a document."""
        return [img for img in self._images.values() if img.doc_id == doc_id]
    
    def get_by_project(self, project_name: str) -> List[ImageRecord]:
        """Get all images for a project."""
        project_lower = project_name.lower()
        return [
            img for img in self._images.values() 
            if img.project_name.lower() == project_lower
        ]
    
    def search_by_description(self, query: str, limit: int = 10) -> List[ImageRecord]:
        """Simple keyword search in descriptions (for basic retrieval)."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored = []
        for img in self._images.values():
            desc_lower = img.description.lower()
            # Simple word overlap scoring
            desc_words = set(desc_lower.split())
            overlap = len(query_words & desc_words)
            if overlap > 0:
                scored.append((overlap, img))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [img for _, img in scored[:limit]]
    
    def list_all(self) -> List[ImageRecord]:
        """List all registered images."""
        return list(self._images.values())
    
    def commit(self) -> None:
        """Save registry to disk."""
        self._save()
    
    def stats(self) -> Dict[str, Any]:
        """Return registry statistics."""
        projects = {img.project_name for img in self._images.values()}
        docs = {img.doc_id for img in self._images.values()}
        
        return {
            "total_images": len(self._images),
            "unique_projects": len(projects),
            "unique_documents": len(docs),
        }
    
    def __len__(self) -> int:
        return len(self._images)
    
    def __contains__(self, image_id: str) -> bool:
        return image_id in self._images


# Factory
_registry_instance: Optional[ImageRegistry] = None


def get_image_registry(registry_path: Optional[Path] = None) -> ImageRegistry:
    """Get or create image registry singleton."""
    global _registry_instance
    
    if _registry_instance is None:
        if registry_path is None:
            registry_path = Path("data/image_registry.json")
        _registry_instance = ImageRegistry(registry_path)
    
    return _registry_instance


def reset_image_registry() -> None:
    """Reset the registry singleton (for testing)."""
    global _registry_instance
    _registry_instance = None
