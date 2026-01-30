"""Image Analyzer - Vision model integration for analyzing user-provided images."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from google import genai

from isaac_generation.config import GenerationConfig, get_config
from isaac_generation.interfaces import IImageAnalyzer

logger = logging.getLogger(__name__)

ARCHITECTURE_ANALYSIS_PROMPT = """Analyze this system architecture diagram and provide:

1. **System Type**: What kind of system is this? (e.g., social media, e-commerce, messaging)
2. **Key Components**: List the main components visible
3. **Data Flow**: Describe how data flows through the system
4. **Architectural Patterns**: Identify patterns used (microservices, event-driven, etc.)
5. **Scalability Aspects**: Note any scalability mechanisms visible

Keep the analysis technical and under 300 words."""

SIMILARITY_SEARCH_PROMPT = """Analyze this architecture diagram and identify:
1. The type of system (social media, e-commerce, caching, etc.)
2. Key architectural patterns used
3. Main technologies or components visible

Provide a concise description suitable for finding similar architectures."""

FALLBACK_DESCRIPTION = "Unable to analyze image. Please describe the architecture you're interested in."


class GeminiImageAnalyzer(IImageAnalyzer):
    """Gemini Vision-based image analyzer."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self._config = config or get_config()
        self._client: Optional[genai.Client] = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Gemini client."""
        api_key = self._config.gemini_api_key.get_secret_value()
        self._client = genai.Client(api_key=api_key)
        logger.info(f"Gemini vision client initialized: {self._config.vision_model}")
    
    def analyze(self, image_path: Path, prompt: Optional[str] = None) -> str:
        """Analyze an image synchronously."""
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return FALLBACK_DESCRIPTION
        
        effective_prompt = prompt or ARCHITECTURE_ANALYSIS_PROMPT
        return self._analyze_with_retry(image_path, effective_prompt)
    
    async def analyze_async(self, image_path: Path, prompt: Optional[str] = None) -> str:
        """Analyze an image asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze, image_path, prompt)
    
    def get_similarity_description(self, image_path: Path) -> str:
        """Get description optimized for similarity search."""
        return self.analyze(image_path, SIMILARITY_SEARCH_PROMPT)
    
    async def get_similarity_description_async(self, image_path: Path) -> str:
        """Async version of similarity description."""
        return await self.analyze_async(image_path, SIMILARITY_SEARCH_PROMPT)
    
    def _analyze_with_retry(self, image_path: Path, prompt: str) -> str:
        """Analyze with exponential backoff retry."""
        last_error: Optional[Exception] = None
        
        for attempt in range(self._config.vision_max_retries + 1):
            try:
                return self._do_analyze(image_path, prompt)
            except Exception as e:
                last_error = e
                if attempt < self._config.vision_max_retries:
                    wait_time = self._config.vision_retry_delay * (2 ** attempt)
                    logger.warning(f"Vision API error (attempt {attempt + 1}): {e}. Retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
        
        logger.error(f"Vision analysis failed after retries: {last_error}")
        return FALLBACK_DESCRIPTION
    
    def _do_analyze(self, image_path: Path, prompt: str) -> str:
        """Execute single analysis attempt."""
        uploaded_file = self._client.files.upload(file=image_path)
        
        response = self._client.models.generate_content(
            model=self._config.vision_model,
            contents=[prompt, uploaded_file],
        )
        
        if not response.text:
            logger.warning(f"Empty response for: {image_path.name}")
            return FALLBACK_DESCRIPTION
        
        description = response.text.strip()
        logger.debug(f"Analyzed {image_path.name}: {description[:100]}...")
        return description
    
    def close(self) -> None:
        """Cleanup resources."""
        self._client = None
        logger.debug("Image analyzer closed")
    
    def __enter__(self) -> "GeminiImageAnalyzer":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()


# Singleton management
_analyzer_instance: Optional[GeminiImageAnalyzer] = None


def get_image_analyzer() -> GeminiImageAnalyzer:
    """Get or create image analyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = GeminiImageAnalyzer()
    return _analyzer_instance
