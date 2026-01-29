"""GitHub Scraper - Crawls repository structure and extracts content."""

import base64
import logging
from typing import List, Optional, Set, Dict
from urllib.parse import urljoin

from isaac_scraper.config import Config, CrawlResult
from isaac_scraper.processors import MarkdownProcessor, ImageSelector
from isaac_scraper.interfaces import RepositoryInterface, ContentFileInterface
from isaac_scraper.github_client import GitHubClient
from isaac_scraper.writer import ResultWriter
from isaac_scraper.exceptions import ContentFetchError, RateLimitError

logger = logging.getLogger(__name__)


class CrawlOrchestrator:
 
    def __init__(
        self, 
        config: Config,
        repository: RepositoryInterface,
        markdown_processor: Optional[MarkdownProcessor] = None,
        image_selector: Optional[ImageSelector] = None,
    ):
        self._config = config
        self._repo = repository
        self._markdown = markdown_processor or MarkdownProcessor()
        self._images = image_selector or ImageSelector()
        self._visited: Set[str] = set()
        self._results: List[CrawlResult] = []
        self._stats = {"processed": 0, "errors": 0, "skipped": 0}
    
    def crawl(self) -> List[CrawlResult]:

        logger.info(f"Crawling: {self._config.start_path}")
        self._crawl_recursive(self._config.start_path, 0)
        return self._results
    
    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)
    
    def _crawl_recursive(self, path: str, depth: int) -> None:

        if depth > self._config.max_depth or path in self._visited:
            return
        self._visited.add(path)
        
        contents = self._get_contents_safe(path)
        if not contents:
            return
        
        # Process if has README (skip root)
        readme = self._find_readme(contents)
        if path != self._config.start_path and readme:
            result = self._process_directory(path, contents, readme)
            if result:
                self._results.append(result)
                self._stats["processed"] += 1
                logger.info(f"+ {result.project_name}")
            else:
                self._stats["skipped"] += 1
        
        # Recurse into subdirectories
        for item in contents:
            if self._should_traverse(item):
                self._crawl_recursive(item.path, depth + 1)
    
    def _get_contents_safe(self, path: str) -> List[ContentFileInterface]:

        try:
            return self._repo.get_contents(path, self._config.branch)
        except (ContentFetchError, RateLimitError) as e:
            logger.warning(f"Failed to get contents at {path}: {e}")
            self._stats["errors"] += 1
            return []
    
    def _find_readme(self, contents: List[ContentFileInterface]) -> Optional[ContentFileInterface]:
        return next((f for f in contents if f.name.lower() == "readme.md"), None)
    
    def _should_traverse(self, item: ContentFileInterface) -> bool:

        if item.type != "dir" or item.name.startswith("."):
            return False
        return item.name.lower() not in self._config.image_folders
    
    def _process_directory(
        self, 
        path: str, 
        contents: List[ContentFileInterface],
        readme: ContentFileInterface,
    ) -> Optional[CrawlResult]:
        text = self._decode_content(readme.content)
        if not text:
            return None
        
        base_url = self._build_raw_url(path)
        images = self._collect_images(contents)
        best_image = self._images.select_best(images)
        parts = path.split("/")
        
        return CrawlResult(
            category=parts[-2] if len(parts) >= 2 else "general",
            project_name=parts[-1],
            title=self._markdown.extract_title(text),
            description=self._markdown.extract_description(text),
            readme_content=self._markdown.fix_links(text, base_url),
            diagram_url=self._build_raw_url(best_image.path) if best_image else None,
            source_path=path,
        )
    
    def _build_raw_url(self, path: str) -> str:
     
        base = f"https://raw.githubusercontent.com/{self._config.repo_name}/{self._config.branch}/"
        return urljoin(base, path.lstrip("/"))
    
    def _collect_images(
        self, 
        contents: List[ContentFileInterface]
    ) -> List[ContentFileInterface]:
        # 1. Collect images from current directory
        images = [f for f in contents if self._images.is_image(f.name)]
        
        # 2. Identify potential image folders
        image_folders = [
            item for item in contents 
            if item.type == "dir" and item.name.lower() in self._config.image_folders
        ]
        

        if image_folders:
            # Change: Iterate over all matching image folders, not just the first one
            for target_folder in image_folders:
                try:
                    sub_contents = self._repo.get_contents(target_folder.path, self._config.branch)
                    images.extend(f for f in sub_contents if self._images.is_image(f.name))
                except (ContentFetchError, RateLimitError) as e:
                    logger.debug(f"Could not fetch images from {target_folder.path}: {e}")
        
        return images
    
    def _decode_content(self, content: Optional[str]) -> Optional[str]:
        if not content:
            return None
        try:
            raw = base64.b64decode(content)
            return raw.decode("utf-8", errors="replace")
        except ValueError as e:
            logger.debug(f"Base64 decode failed: {e}")
            return None


class GitScraper:
    
    def __init__(self, config: Config):
        self._config = config
        self._client: Optional[GitHubClient] = None
        self._orchestrator: Optional[CrawlOrchestrator] = None
        self._writer = ResultWriter(config.output_file)
        self._connect()
    
    def _connect(self) -> None:
        self._client = GitHubClient(self._config)
        self._orchestrator = CrawlOrchestrator(self._config, self._client)
    
    def crawl(self) -> None:
        if not self._orchestrator:
            raise RuntimeError("Scraper not initialized")
        
        results = self._orchestrator.crawl()
        self._writer.write(results)
    
    def get_stats(self) -> Dict[str, int]:
        if not self._orchestrator:
            return {"processed": 0, "errors": 0, "skipped": 0}
        return self._orchestrator.get_stats()
    
    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
        self._orchestrator = None
    
    def __enter__(self) -> "GitScraper":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
