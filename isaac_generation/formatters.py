"""Context Formatters - Implementations of IContextFormatter for processing retrieval results."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from isaac_generation.config import GenerationConfig, get_config
from isaac_generation.interfaces import (
    IContextFormatter,
    FormattedContext,
    ResolvedImage,
)
from isaac_core.constants import MIN_RELEVANCE_SCORE

logger = logging.getLogger(__name__)


class ContextFormatter(IContextFormatter):
    """Default context formatter for processing retrieval results."""
    
    SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self._config = config or get_config()
        self._images_dir = self._config.images_dir
    
    def format(self, retrieval_result: Dict[str, Any]) -> FormattedContext:
        """Format retrieval results into structured context."""
        chunks = retrieval_result.get("chunks", [])
        image_refs = retrieval_result.get("images", [])
        
        return FormattedContext(
            text=self._format_chunks(chunks),
            sources=self._extract_sources(chunks),
            images=self._resolve_images(image_refs),
        )
    
    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into context text."""
        if not chunks:
            return "No relevant context found."
        
        parts = []
        for i, chunk in enumerate(chunks[:self._config.top_k_chunks], 1):
            content = chunk.get("content", "").strip()
            if not content:
                continue
            
            header = self._build_header(chunk.get("metadata", {}), i)
            parts.append(f"--- {header} ---\n{content}")
        
        return "\n\n".join(parts) if parts else "No relevant context found."
    
    def _build_header(self, metadata: Dict[str, Any], index: int) -> str:
        """Build header string from metadata."""
        source_name = (
            metadata.get("h1") or
            metadata.get("project_name") or
            metadata.get("source") or
            f"Source {index}"
        )
        
        section = metadata.get("h2", "")
        if section:
            return f"[{source_name}] - {section}"
        return f"[{source_name}]"
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract unique source names from chunks."""
        sources = []
        seen = set()
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            source = (
                metadata.get("h1") or
                metadata.get("project_name") or
                metadata.get("source") or
                f"Source {i}"
            )
            
            if source not in seen:
                sources.append(source)
                seen.add(source)
        
        return sources
    
    def _resolve_images(self, image_refs: List[Dict[str, Any]]) -> List[ResolvedImage]:
        """Resolve image references to actual file paths."""
        resolved = []
        seen_ids = set()
        
        for ref in image_refs:
            image_id = ref.get("image_id", "")
            if not image_id or image_id in seen_ids:
                continue
            
            path = self._find_image_file(image_id)
            if path:
                resolved.append(ResolvedImage(
                    image_id=image_id,
                    path=path,
                    description=ref.get("description", "Architecture diagram"),
                    source_chunk_index=ref.get("source_chunk_index", 0),
                ))
                seen_ids.add(image_id)
        
        return resolved
    
    def _find_image_file(self, image_id: str) -> Optional[Path]:
        """Find image file by ID."""
        hash_part = image_id.replace("IMG_", "")
        
        for file in self._images_dir.iterdir():
            if file.is_file() and hash_part in file.stem:
                logger.debug(f"Resolved {image_id} -> {file}")
                return file
        
        logger.warning(f"Could not resolve image: {image_id}")
        return None
    
    def find_images_by_project(self, project_name: str) -> List[ResolvedImage]:
        """Find all images for a specific project."""
        resolved = []
        project_lower = project_name.lower().replace(" ", "_")
        
        for file in self._images_dir.iterdir():
            if not file.is_file():
                continue
            
            if file.stem.lower().startswith(project_lower):
                parts = file.stem.split("_")
                hash_part = parts[-1] if len(parts) >= 2 else file.stem
                image_id = f"IMG_{hash_part}"
                
                resolved.append(ResolvedImage(
                    image_id=image_id,
                    path=file,
                    description=f"Architecture diagram for {project_name}",
                    source_chunk_index=0,
                ))
        
        logger.debug(f"Found {len(resolved)} images for project: {project_name}")
        return resolved


class EnhancedContextFormatter(ContextFormatter):
    """Enhanced formatter with image context injection and auto-discovery."""
    
    def format(self, retrieval_result: Dict[str, Any]) -> FormattedContext:
        """Format with image descriptions injected into context."""
        chunks = retrieval_result.get("chunks", [])
        image_refs = retrieval_result.get("images", [])
        
        # Check relevance based on chunk scores
        max_score = self._get_max_score(chunks)
        is_relevant = max_score >= MIN_RELEVANCE_SCORE
        
        logger.info(f"Context relevance check: max_score={max_score:.4f}, threshold={MIN_RELEVANCE_SCORE}, is_relevant={is_relevant}")
        
        # If not relevant, return minimal context
        if not is_relevant:
            return FormattedContext(
                text="No relevant context found in knowledge base.",
                sources=[],
                images=[],
                is_relevant=False,
                max_score=max_score,
            )
        
        context_text = self._format_chunks(chunks)
        sources = self._extract_sources(chunks)
        images = self._resolve_images(image_refs)
        
        if not images:
            images = self._discover_images_from_chunks(chunks)
        
        if images:
            context_text = self._inject_image_context(context_text, images)
        
        return FormattedContext(
            text=context_text,
            sources=sources,
            images=images,
            is_relevant=True,
            max_score=max_score,
        )
    
    def _get_max_score(self, chunks: List[Dict[str, Any]]) -> float:
        """Get maximum relevance score from chunks."""
        if not chunks:
            return 0.0
        scores = [chunk.get("score", 0.0) for chunk in chunks]
        return max(scores) if scores else 0.0
    
    def _discover_images_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[ResolvedImage]:
        """Auto-discover images based on project names in chunk metadata."""
        images = []
        seen_projects = set()
        
        for chunk in chunks[:self._config.top_k_chunks]:
            metadata = chunk.get("metadata", {})
            
            project_name = (
                metadata.get("project_name") or
                metadata.get("h1", "").lower().replace(" ", "_") or
                ""
            )
            
            if not project_name or project_name in seen_projects:
                continue
            
            seen_projects.add(project_name)
            project_images = self.find_images_by_project(project_name)
            
            if project_images:
                images.append(project_images[0])
        
        return images
    
    def _inject_image_context(self, text: str, images: List[ResolvedImage]) -> str:
        """Inject image descriptions into context."""
        image_section = "\n\n--- [Architecture Diagrams Available] ---\n"
        
        for i, img in enumerate(images, 1):
            image_section += f"Figure {i} ({img.image_id}): {img.description}\n"
        
        image_section += (
            "\nNote: These diagrams will be shown to the user. "
            "Reference them as 'Figure N' when relevant."
        )
        
        return text + image_section
