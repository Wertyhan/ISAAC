import hashlib
import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from isaac_ingestion.config import Config, ImageMeta
from isaac_ingestion.exceptions import ImageDownloadError, NetworkTimeoutError

logger = logging.getLogger(__name__)


# Constants
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
CHUNK_SIZE = 8192


# Service
class ImageManager:
    """Image downloading and local storage."""
    
    def __init__(self, config: Config):
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "ISAAC-Ingestion/1.0",
            "Accept": "image/*",
        })
    
    def download(self, url: str, project_name: str) -> Optional[ImageMeta]:
        """Download image from URL, return ImageMeta or None."""
        if not self._is_valid_url(url):
            logger.warning(f"Invalid image URL: {url}")
            return None
        
        return self._retry_download(url, project_name)
    
    def _retry_download(self, url: str, project_name: str) -> Optional[ImageMeta]:
        last_error: Optional[Exception] = None
        
        for attempt in range(self._config.download_max_retries + 1):
            try:
                return self._perform_download(url, project_name)
                
            except (ImageDownloadError, NetworkTimeoutError) as e:
                last_error = e
                if attempt < self._config.download_max_retries:
                    wait_time = 2.0 * (2 ** attempt)
                    logger.warning(f"Download failed, retry {attempt + 1}: {e}")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Unexpected error downloading {url}: {e}")
                return None
        
        logger.error(f"Download failed after retries: {url} - {last_error}")
        return None
    
    def _perform_download(self, url: str, project_name: str) -> ImageMeta:
        try:
            response = self._session.get(
                url,
                timeout=self._config.request_timeout,
                stream=True,
            )
            response.raise_for_status()
            
        except requests.Timeout as e:
            raise NetworkTimeoutError(f"Timeout downloading: {url}") from e
        except requests.RequestException as e:
            raise ImageDownloadError(url, str(e)) from e
        
        # Calculate hash from content
        content = response.content
        file_hash = self._compute_hash(content)
        
        # Determine file extension
        extension = self._get_extension(url, response.headers.get("Content-Type"))
        
        # Build local path
        local_path = self._build_local_path(project_name, file_hash, extension)
        
        # Skip if already exists (deduplication by hash)
        if local_path.exists():
            logger.debug(f"Image already exists: {local_path.name}")
        else:
            self._save_image(local_path, content)
            logger.info(f"Downloaded: {local_path.name}")
        
        return ImageMeta(
            original_url=url,
            local_path=local_path,
            file_hash=file_hash,
            project_name=project_name,
        )
    
    def _compute_hash(self, content: bytes) -> str:
        return hashlib.md5(content).hexdigest()
    
    def _get_extension(self, url: str, content_type: Optional[str]) -> str:
        parsed = urlparse(url)
        path = Path(parsed.path)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return path.suffix.lower()
        
        # Fallback to content type
        if content_type:
            mime_map = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/gif": ".gif",
                "image/webp": ".webp",
                "image/svg+xml": ".svg",
            }
            for mime, ext in mime_map.items():
                if mime in content_type:
                    return ext
        
        return ".png"  # Default
    
    def _build_local_path(self, project_name: str, file_hash: str, extension: str) -> Path:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name)
        filename = f"{safe_name}_{file_hash[:12]}{extension}"
        return self._config.images_dir / filename
    
    def _save_image(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    
    def _is_valid_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False
    
    def close(self) -> None:
        self._session.close()
        logger.debug("ImageManager session closed")
    
    def __enter__(self) -> "ImageManager":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
