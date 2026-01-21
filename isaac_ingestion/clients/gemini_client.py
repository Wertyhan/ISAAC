import logging
import time
from pathlib import Path
from typing import Optional, Union

from google import genai
from google.genai import types

from isaac_ingestion.config import Config
from isaac_ingestion.exceptions import (
    GeminiAPIError,
    GeminiRateLimitError,
    GeminiQuotaExceededError,
)

logger = logging.getLogger(__name__)


# Constants
DEFAULT_PROMPT = """Analyze this technical diagram or image and provide a concise description.
Focus on:
- The main components or systems shown
- Data flow or relationships between components
- Key architectural patterns visible

Keep the description under 200 words and technical in nature."""

FALLBACK_DESCRIPTION = "[Image description unavailable]"


# Client
class GeminiVisionClient:
    """Gemini Vision API client for image descriptions."""
    
    def __init__(self, config: Config):
        self._config = config
        self._client: Optional[genai.Client] = None
        self._configure()
    
    def _configure(self) -> None:
        self._client = genai.Client(api_key=self._config.gemini_api_key.get_secret_value())
        logger.info(f"Gemini client configured for model: {self._config.vision_model}")
    
    def generate_description(
        self, 
        image_path: Path,
        prompt: str = DEFAULT_PROMPT,
    ) -> str:
        """Generate description for an image, returns fallback on failure."""
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return FALLBACK_DESCRIPTION
        
        return self._retry_with_backoff(image_path, prompt)
    
    def _retry_with_backoff(self, image_path: Path, prompt: str) -> str:
        last_error: Optional[Exception] = None
        
        for attempt in range(self._config.vision_max_retries + 1):
            try:
                return self._generate(image_path, prompt)
            
            except GeminiRateLimitError as e:
                last_error = e
                if attempt < self._config.vision_max_retries:
                    wait_time = getattr(e, 'retry_after', self._config.vision_retry_delay * (2 ** attempt))
                    logger.warning(f"Rate limited. Waiting {wait_time:.0f}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    
            except GeminiQuotaExceededError as e:
                logger.error(f"Quota exceeded: {e}")
                return FALLBACK_DESCRIPTION
                
            except GeminiAPIError as e:
                last_error = e
                if attempt < self._config.vision_max_retries:
                    wait_time = self._config.vision_retry_delay * (2 ** attempt)
                    logger.warning(f"API error. Retrying in {wait_time:.1f}s: {e}")
                    time.sleep(wait_time)
        
        logger.error(f"Failed after {self._config.vision_max_retries + 1} attempts: {last_error}")
        return FALLBACK_DESCRIPTION
    
    def _generate(self, image_path: Path, prompt: str) -> str:
        try:
            # Upload file first (efficient for large images)
            uploaded_file = self._load_image(image_path)
            
            response = self._client.models.generate_content(
                model=self._config.vision_model,
                contents=[prompt, uploaded_file]
            )
            
            if not response.text:
                logger.warning(f"Empty response for: {image_path.name}")
                return FALLBACK_DESCRIPTION
            
            description = response.text.strip()
            logger.debug(f"Generated description for {image_path.name}: {description[:100]}...")
            return description
            
        except genai.errors.ResourceExhausted as e:
            raise GeminiQuotaExceededError(str(e)) from e
        except genai.errors.RateLimitError as e:
            raise GeminiRateLimitError() from e
        except genai.errors.InvalidArgument as e:
            logger.warning(f"Invalid image format or request: {image_path.name} - {e}")
            return FALLBACK_DESCRIPTION
        except Exception as e:
            raise GeminiAPIError("Generation failed", str(e)) from e
    
    def _load_image(self, image_path: Path) -> types.File:
        return self._client.files.upload(path=image_path)

    def close(self) -> None:
        self._client = None
        logger.debug("Gemini client closed")
    
    def __enter__(self) -> "GeminiVisionClient":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
