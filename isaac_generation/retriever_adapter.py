"""Retriever Adapter - Adapter pattern for integrating with isaac_api retriever service."""

import logging
from typing import Dict, Any, Optional

from isaac_generation.interfaces import IRetrieverAdapter
from isaac_api.services.retriever import RetrieverService, get_retriever_service

logger = logging.getLogger(__name__)


class RetrieverAdapter(IRetrieverAdapter):
    """Adapter wrapping RetrieverService for generation use."""
    
    _SEARCH_ENHANCEMENT_KEYWORDS = ("architecture", "system design")
    
    def __init__(self, retriever_service: Optional[RetrieverService] = None):
        self._retriever = retriever_service or get_retriever_service()
    
    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search for relevant documents."""
        logger.debug(f"Searching: '{query[:50]}...' k={k}")
        result = self._retriever.search(query=query, k=k)
        return result.model_dump()
    
    def search_by_image(self, image_description: str, k: int = 5) -> Dict[str, Any]:
        """Search using image description as query."""
        enhanced_query = self._enhance_image_query(image_description)
        logger.debug(f"Image-based search: '{enhanced_query[:50]}...'")
        return self.search(enhanced_query, k)
    
    def _enhance_image_query(self, description: str) -> str:
        """Enhance image description for better retrieval."""
        description_lower = description.lower()
        additions = [kw for kw in self._SEARCH_ENHANCEMENT_KEYWORDS if kw not in description_lower]
        
        if additions:
            return f"{description} Related: {' '.join(additions)}"
        return description


def get_retriever_adapter() -> RetrieverAdapter:
    """Create or get retriever adapter singleton."""
    return RetrieverAdapter()
