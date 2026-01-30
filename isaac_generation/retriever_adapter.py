"""Retriever Adapter - Adapter pattern for integrating with isaac_api retriever service."""

import logging
from typing import Dict, Any, Optional

from isaac_generation.interfaces import IRetrieverAdapter
from isaac_api.services.retriever import RetrieverService, get_retriever_service

logger = logging.getLogger(__name__)


class RetrieverAdapter(IRetrieverAdapter):
    """Adapter wrapping RetrieverService for generation use."""
    
    _SEARCH_ENHANCEMENT_KEYWORDS = ("architecture", "system design", "distributed system")
    
    # Common query patterns that benefit from expansion
    _QUERY_PATTERNS = {
        # Food/delivery apps
        "food delivery": "real-time processing message queue notification microservices scaling",
        "glovo": "real-time processing order management notification microservices scaling",
        "doordash": "real-time processing order management notification microservices scaling",
        "uber eats": "real-time processing order management notification microservices scaling",
        # E-commerce
        "ecommerce": "product catalog payment order management shopping cart database scaling",
        "marketplace": "product catalog payment order management shopping cart database scaling",
        # Social
        "twitter": "fan-out timeline feed caching notification scaling",
        "social network": "fan-out timeline feed caching notification scaling graph",
        # Real-time
        "real-time": "message queue websocket kafka notification event-driven async",
        "realtime": "message queue websocket kafka notification event-driven async",
        # High scale
        "million": "scaling horizontal load balancer caching sharding CDN replication",
        "scale": "scaling horizontal load balancer caching sharding CDN replication",
        # URL shortener
        "url shortener": "hash base62 key-value redirect analytics pastebin",
        "bit.ly": "hash base62 key-value redirect analytics pastebin",
        # Web crawler
        "web crawler": "distributed queue URL frontier indexing BFS politeness",
        "crawler": "distributed queue URL frontier indexing BFS politeness",
        # Cache
        "cache": "Redis Memcached LRU TTL memory cache-aside write-through",
        # Database
        "database": "SQL NoSQL replication sharding master-slave consistency ACID",
        # CAP theorem
        "cap theorem": "consistency availability partition tolerance distributed",
    }
    
    def __init__(self, retriever_service: Optional[RetrieverService] = None):
        self._retriever = retriever_service or get_retriever_service()
    
    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search for relevant documents with query enhancement."""
        enhanced_query = self._enhance_query(query)
        logger.debug(f"Searching: '{enhanced_query[:80]}...' k={k}")
        result = self._retriever.search(query=enhanced_query, k=k)
        return result.model_dump()
    
    def search_by_image(self, image_description: str, k: int = 5) -> Dict[str, Any]:
        """Search using image description as query."""
        enhanced_query = self._enhance_image_query(image_description)
        logger.debug(f"Image-based search: '{enhanced_query[:50]}...'")
        return self.search(enhanced_query, k)
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query with relevant architecture terms for better retrieval."""
        query_lower = query.lower()
        enhancements = []
        
        # Check for matching patterns
        for pattern, expansion in self._QUERY_PATTERNS.items():
            if pattern in query_lower:
                enhancements.append(expansion)
        
        # Add general architecture terms if not present
        if not any(kw in query_lower for kw in self._SEARCH_ENHANCEMENT_KEYWORDS):
            enhancements.append("system architecture design pattern")
        
        if enhancements:
            expansion = " ".join(enhancements[:3])  # Limit to 3 expansions
            return f"{query} {expansion}"
        
        return query
    
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
