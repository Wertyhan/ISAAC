"""Generation Service - Main orchestrator following Facade pattern."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, AsyncIterator

from langchain_core.messages import BaseMessage

from isaac_generation.config import GenerationConfig, get_config
from isaac_generation.interfaces import (
    FormattedContext,
    GenerationInput,
    ResolvedImage,
)
from isaac_generation.formatters import EnhancedContextFormatter
from isaac_generation.image_analyzer import GeminiImageAnalyzer, get_image_analyzer
from isaac_generation.retriever_adapter import RetrieverAdapter, get_retriever_adapter
from isaac_generation.generator import ResponseGenerator, get_generator
from isaac_core.constants import IMAGE_ANALYSIS_FAILURE_MARKER

logger = logging.getLogger(__name__)


@dataclass
class StreamingResponse:
    """Container for streaming response with metadata."""
    token_stream: AsyncIterator[str]
    context: FormattedContext
    user_image_analysis: Optional[str] = None


class GenerationService:
    """Main generation service orchestrating the RAG pipeline."""
    
    _IMAGE_KEYWORDS = frozenset([
        "image", "picture", "diagram", "architecture", "this",
        "what", "analyze", "explain", "describe", "show",
        "зображення", "картинка", "що", "опиши", "поясни",
    ])
    
    _FALLBACK_SIMILARITY_QUERY = "system architecture diagram components data flow"
    
    # Keywords indicating user wants HELP/RECOMMENDATIONS (show similar diagrams)
    _HELP_REQUEST_KEYWORDS = frozenset([
        "help", "create", "build", "make", "design", "implement", "develop",
        "improve", "optimize", "scale", "enhance", "recommend", "suggest",
        "how to", "how do", "how can", "what should", "best practice",
        "similar", "like this", "based on", "inspiration", "reference",
        "compare", "comparison", "alternative", "better", "advice",
    ])
    
    # Keywords indicating user just wants ANALYSIS/DESCRIPTION (no recommendations)
    _ANALYSIS_ONLY_KEYWORDS = frozenset([
        "what is this", "what's this", "what does this show", "describe",
        "explain this", "tell me about this", "what's on", "what is on",
        "analyze this", "breakdown", "overview", "summary",
    ])
    
    # Keywords that indicate the query is about system architecture/software
    _ARCHITECTURE_KEYWORDS = frozenset([
        "architecture", "system", "design", "microservice", "database", "api",
        "scalability", "cache", "queue", "load balancer", "backend", "frontend",
        "distributed", "cloud", "server", "service", "pattern", "infrastructure",
        "deployment", "kubernetes", "docker", "aws", "azure", "gcp", "redis",
        "kafka", "rabbitmq", "postgresql", "mongodb", "elasticsearch", "nginx",
        "rest", "graphql", "grpc", "websocket", "cdn", "dns", "ssl", "http",
        "latency", "throughput", "availability", "reliability", "fault tolerance",
        "replication", "sharding", "partition", "consistency", "transaction",
        "authentication", "authorization", "oauth", "jwt", "security",
        "monitoring", "logging", "metrics", "alerting", "observability",
        "ci/cd", "devops", "agile", "scrum", "sprint", "mvp",
        "app", "application", "platform", "software", "tech", "technology",
        "delivery", "order", "payment", "notification", "real-time", "realtime",
        "taxi", "ride", "food", "e-commerce", "ecommerce", "marketplace",
        "twitter", "instagram", "facebook", "netflix", "spotify", "airbnb",
        "booking", "amazon", "uber", "bolt", "like", "similar",
    ])
    
    # Keywords that indicate completely off-topic queries (NOT about tech/software)
    _OFFTOPIC_KEYWORDS = frozenset([
        "volkswagen", "passat", "bmw", "mercedes", "audi", "toyota", "honda",
        "ford", "chevrolet", "nissan", "mazda", "hyundai", "kia", "engine",
        "horsepower", "transmission", "sedan", "suv", "truck", "motorcycle",
        "car model", "vehicle specs",
        "recipe", "cook", "bake", "ingredient", "cuisine", "chef", "kitchen",
        "football score", "basketball game", "soccer match", "tennis", "golf", "swimming",
        "actor biography", "actress", "singer", "band concert",
        "weather forecast", "horoscope", "lottery numbers",
    ])
    
    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        retriever: Optional[RetrieverAdapter] = None,
        formatter: Optional[EnhancedContextFormatter] = None,
        image_analyzer: Optional[GeminiImageAnalyzer] = None,
        generator: Optional[ResponseGenerator] = None,
    ):
        self._config = config or get_config()
        self._retriever = retriever or get_retriever_adapter()
        self._formatter = formatter or EnhancedContextFormatter(self._config)
        self._image_analyzer = image_analyzer or get_image_analyzer()
        self._generator = generator or get_generator()
        
        logger.info("GenerationService initialized")
    
    async def process_query(
        self,
        query: str,
        chat_history: Optional[List[BaseMessage]] = None,
        user_image_path: Optional[Path] = None,
        top_k: Optional[int] = None,
    ) -> StreamingResponse:
        """Process a user query and return streaming response."""
        chat_history = chat_history or []
        effective_top_k = top_k or self._config.top_k_chunks
        
        # Check if query is off-topic before processing
        if self._is_offtopic_query(query):
            logger.info(f"Off-topic query detected: '{query[:50]}...'")
            empty_context = FormattedContext(
                text="No relevant context found in knowledge base.",
                sources=[],
                images=[],
                is_relevant=False,
                max_score=0.0,
            )
            gen_input = GenerationInput(
                query=query,
                context=empty_context,
                chat_history=chat_history,
            )
            return StreamingResponse(
                token_stream=self._generator.generate_stream(gen_input),
                context=empty_context,
                user_image_analysis=None,
            )
        
        user_image_analysis = await self._analyze_user_image(query, user_image_path)
        
        # Check if this is an "analysis only" request (user just wants description, no recommendations)
        is_analysis_only = self._is_analysis_only_request(query, user_image_path)
        
        if is_analysis_only and user_image_analysis:
            # User just wants to understand the diagram - no need for KB search
            logger.info("Analysis-only request detected - skipping knowledge base search")
            minimal_context = FormattedContext(
                text="User provided an architecture diagram for analysis.",
                sources=[],
                images=[],
                is_relevant=True,
                max_score=1.0,
            )
            gen_input = GenerationInput(
                query=query,
                context=minimal_context,
                chat_history=chat_history,
                user_image_path=user_image_path,
                user_image_description=user_image_analysis,
            )
            return StreamingResponse(
                token_stream=self._generator.generate_stream(gen_input),
                context=minimal_context,
                user_image_analysis=user_image_analysis,
            )
        
        # User wants help/recommendations - search knowledge base for similar architectures
        search_query = self._build_search_query(query, user_image_analysis)
        
        retrieval_result = self._retriever.search(
            query=search_query,
            k=effective_top_k,
        )
        context = self._formatter.format(retrieval_result)
        
        logger.info(f"Retrieved {len(context.sources)} sources, {len(context.images)} images, is_relevant={context.is_relevant}, max_score={context.max_score:.4f}")
        
        gen_input = GenerationInput(
            query=query,
            context=context,
            chat_history=chat_history,
            user_image_path=user_image_path,
            user_image_description=user_image_analysis,
        )
        
        return StreamingResponse(
            token_stream=self._generator.generate_stream(gen_input),
            context=context,
            user_image_analysis=user_image_analysis,
        )
    
    def _is_analysis_only_request(self, query: str, image_path: Optional[Path]) -> bool:
        """
        Determine if user just wants image analysis (no KB recommendations).
        
        Returns True if:
        - User provided an image AND
        - Query is short/generic OR contains analysis-only keywords AND
        - Query does NOT contain help/recommendation keywords
        """
        if not image_path:
            return False
        
        query_lower = query.lower().strip()
        
        # Check for explicit help/recommendation keywords
        wants_help = any(kw in query_lower for kw in self._HELP_REQUEST_KEYWORDS)
        if wants_help:
            return False
        
        # Check for analysis-only keywords
        wants_analysis = any(kw in query_lower for kw in self._ANALYSIS_ONLY_KEYWORDS)
        if wants_analysis:
            return True
        
        # Very short queries with image = probably just wants description
        # e.g., "what is this?", "explain", "describe"
        if len(query_lower.split()) <= 5:
            return True
        
        return False
    
    def _is_offtopic_query(self, query: str) -> bool:
        """Check if query is completely off-topic (not about architecture/software)."""
        query_lower = query.lower()
        
        # Check for explicit off-topic keywords
        has_offtopic = any(kw in query_lower for kw in self._OFFTOPIC_KEYWORDS)
        
        # Check for architecture-related keywords
        has_architecture = any(kw in query_lower for kw in self._ARCHITECTURE_KEYWORDS)
        
        # If query has off-topic keywords and NO architecture keywords - it's off-topic
        if has_offtopic and not has_architecture:
            return True
        
        return False
    
    async def _analyze_user_image(
        self,
        query: str,
        image_path: Optional[Path],
    ) -> Optional[str]:
        """Analyze user image if provided and relevant to query."""
        if not image_path or not image_path.exists():
            return None
        
        if not self._is_image_related_query(query):
            return None
        
        logger.info(f"Analyzing user image: {image_path.name}")
        
        try:
            analysis = await self._image_analyzer.analyze_async(image_path)
            if IMAGE_ANALYSIS_FAILURE_MARKER in analysis:
                logger.warning("Image analysis failed, proceeding without it")
                return None
            return analysis
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return None
    
    def _is_image_related_query(self, query: str) -> bool:
        """Check if query is related to image analysis."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self._IMAGE_KEYWORDS)
    
    def _build_search_query(self, query: str, image_analysis: Optional[str]) -> str:
        """Build search query, enhancing with image analysis if available."""
        if not image_analysis:
            return query
        if query.strip():
            return f"{query}\n\nRelated context: {image_analysis[:500]}"
        return image_analysis
    
    async def process_image_similarity(
        self,
        image_path: Path,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> StreamingResponse:
        """Find similar architectures based on user-provided image."""
        chat_history = chat_history or []
        
        similarity_desc, full_analysis = await self._get_image_descriptions(image_path)
        
        retrieval_result = self._retriever.search_by_image(
            image_description=similarity_desc,
            k=self._config.top_k_chunks,
        )
        context = self._formatter.format(retrieval_result)
        
        query = (
            "Based on the architecture diagram I provided, explain what kind of system "
            "this is and compare it with similar architectures from your knowledge base. "
            "Show me the most relevant reference diagrams."
        )
        
        gen_input = GenerationInput(
            query=query,
            context=context,
            chat_history=chat_history,
            user_image_path=image_path,
            user_image_description=full_analysis,
        )
        
        return StreamingResponse(
            token_stream=self._generator.generate_stream(gen_input),
            context=context,
            user_image_analysis=full_analysis,
        )
    
    async def _get_image_descriptions(self, image_path: Path) -> tuple[str, Optional[str]]:
        """Get similarity description and full analysis for image."""
        try:
            similarity_desc = await self._image_analyzer.get_similarity_description_async(image_path)
            full_analysis = await self._image_analyzer.analyze_async(image_path)
            
            if IMAGE_ANALYSIS_FAILURE_MARKER in similarity_desc:
                return self._FALLBACK_SIMILARITY_QUERY, None
            if IMAGE_ANALYSIS_FAILURE_MARKER in full_analysis:
                return similarity_desc, None
            
            return similarity_desc, full_analysis
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return self._FALLBACK_SIMILARITY_QUERY, None
    
    def get_resolved_images(self, context: FormattedContext) -> List[ResolvedImage]:
        """Get list of resolved images from context."""
        return [img for img in context.images if img.exists]


# Singleton management
_service_instance: Optional[GenerationService] = None


def get_generation_service() -> GenerationService:
    """Get or create generation service singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = GenerationService()
    return _service_instance
