import logging
import re
import time
from typing import List, Optional

from github import Github, Auth, GithubException
from github import UnknownObjectException, BadCredentialsException
from github.ContentFile import ContentFile

from isaac_scraper.config import Config
from isaac_scraper.exceptions import (
    RateLimitError, 
    RepositoryNotFoundError, 
    AuthenticationError,
    ContentFetchError,
)
from isaac_scraper.interfaces import RepositoryInterface

logger = logging.getLogger(__name__)


# Utilities
def sanitize_log(data) -> str:
    """Extract useful fields from GitHub error, hide tokens."""
    if isinstance(data, dict):
        useful = {k: data[k] for k in ("message", "documentation_url") if k in data}
        text = str(useful) if useful else str(data)[:200]
    else:
        text = str(data)[:500]
    
    for pattern in [r"ghp_\w+", r"gho_\w+", r"Bearer\s+[\w.-]+"]:
        text = re.sub(pattern, "***", text, flags=re.IGNORECASE)
    return text


# Client Implementation
class GitHubClient(RepositoryInterface):
    """GitHub API client with connection management and retry logic."""
    
    def __init__(self, config: Config):
        self._config = config
        self._github: Optional[Github] = None
        self._repo = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to GitHub."""
        self._github = Github(auth=Auth.Token(self._config.github_token.get_secret_value()))
        try:
            self._repo = self._github.get_repo(self._config.repo_name)
        except BadCredentialsException as e:
            self.close()
            raise AuthenticationError("Invalid GitHub token") from e
        except UnknownObjectException as e:
            self.close()
            raise RepositoryNotFoundError(
                f"Repository not found: {self._config.repo_name}"
            ) from e
        logger.info(f"Connected to repo: {self._repo.full_name}")
    
    @property
    def full_name(self) -> str:
        return self._repo.full_name if self._repo else ""
    
    def get_contents(self, path: str, ref: str) -> List[ContentFile]:
        """Get contents with retry logic for rate limits."""
        return self._retry_with_backoff(
            lambda: self._fetch_contents(path, ref),
            path
        )
    
    def _fetch_contents(self, path: str, ref: str) -> List[ContentFile]:
        """Fetch contents from repository."""
        contents = self._repo.get_contents(path, ref=ref)
        return [contents] if isinstance(contents, ContentFile) else list(contents)
    
    def _retry_with_backoff(self, operation, path: str) -> List[ContentFile]:
        """Execute operation with exponential backoff retry."""
        
        for attempt in range(self._config.max_retries + 1):
            try:
                return operation()
            except GithubException as e:
                
                if e.status == 404:
                    logger.debug(f"Path not found: {path}")
                    return []
                
                if e.status == 403 and "rate limit" in str(e.data).lower():
                    wait = self._config.retry_delay * (2 ** attempt)
                    if attempt < self._config.max_retries:
                        logger.warning(f"Rate limit hit, waiting {wait:.0f}s (attempt {attempt + 1})")
                        time.sleep(wait)
                        continue
                    raise RateLimitError(wait) from e
                
                logger.warning(f"GitHub error at {path}: {sanitize_log(e.data)}")
                raise ContentFetchError(f"Failed to fetch {path}") from e
        
        return []
    
    def close(self) -> None:
        if self._github is not None:
            self._github.close()
            self._github = None
            self._repo = None
            logger.debug("GitHub connection closed")
    
    def __enter__(self) -> "GitHubClient":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
