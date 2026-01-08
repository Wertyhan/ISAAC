import json
import base64
import time
import logging
import re
import tempfile
import os
from typing import List, Optional, Set, Dict

from github import Github, Auth, GithubException
from github import UnknownObjectException, BadCredentialsException
from github.ContentFile import ContentFile

from isaac_scraper.config import Config, CrawlResult
from isaac_scraper.processors import MarkdownProcessor, ImageSelector

logger = logging.getLogger(__name__)

IMAGE_FOLDERS = frozenset(["img", "images", "assets", "diagrams", "figures"])


def sanitize_log(text: str) -> str:
    text = str(text)
    for pattern in [r"ghp_\w+", r"gho_\w+", r"Bearer\s+[\w.-]+"]:
        text = re.sub(pattern, "***", text, flags=re.IGNORECASE)
    return text[:500]


class GitScraper:
    
    def __init__(self, config: Config):
        self.config = config
        self.markdown = MarkdownProcessor()
        self.images = ImageSelector()
        self._visited: Set[str] = set()
        self._results: List[CrawlResult] = []
        self._stats = {"processed": 0, "errors": 0, "skipped": 0}
        
        # Connect
        self._github = Github(auth=Auth.Token(config.github_token))
        try:
            self._repo = self._github.get_repo(config.repo_name)
        except BadCredentialsException:
            raise ValueError("Invalid GitHub token")
        except UnknownObjectException:
            raise ValueError(f"Repository not found: {config.repo_name}")
        logger.info(f"Token: {'*' * 4}{config.github_token[-4:]} | Repo: {self._repo.full_name}")
    
    def crawl(self) -> None:
        logger.info(f"Crawling: {self.config.start_path}")
        self._crawl(self.config.start_path, 0)
        
        # Atomic save via temp file
        output_dir = os.path.dirname(self.config.output_file) or "."
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, suffix=".tmp", delete=False) as f:
            json.dump([r.model_dump() for r in self._results], f, ensure_ascii=False, indent=2)
            temp_path = f.name
        os.replace(temp_path, self.config.output_file)
        
        logger.info(f"Saved {len(self._results)} items to {self.config.output_file}")
    
    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)
    
    def _crawl(self, path: str, depth: int) -> None:
        if depth > self.config.max_depth or path in self._visited:
            return
        self._visited.add(path)
        
        contents = self._get_contents(path)
        if not contents:
            return
        
        # Process if has README (skip root)
        if path != self.config.start_path and any(f.name.lower() == "readme.md" for f in contents):
            result = self._process(path, contents)
            if result:
                self._results.append(result)
                self._stats["processed"] += 1
                logger.info(f"+ {result.project_name}")
            else:
                self._stats["skipped"] += 1
        
        # Recurse
        for item in contents:
            if item.type == "dir" and not item.name.startswith("."):
                if item.name.lower() not in IMAGE_FOLDERS:
                    self._crawl(item.path, depth + 1)
    
    def _get_contents(self, path: str) -> List[ContentFile]:
        for attempt in range(self.config.max_retries + 1):
            try:
                contents = self._repo.get_contents(path, ref=self.config.branch)
                return [contents] if isinstance(contents, ContentFile) else list(contents)
            except GithubException as e:
                if e.status == 404:
                    return []
                if e.status == 403 and "rate limit" in str(e.data).lower():
                    wait = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit, waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                logger.warning(f"Error at {path}: {sanitize_log(e.data)}")
                self._stats["errors"] += 1
                return []
        return []
    
    def _process(self, path: str, contents: List[ContentFile]) -> Optional[CrawlResult]:
        readme = next((f for f in contents if f.name.lower() == "readme.md"), None)
        if not readme:
            return None
        
        text = self._decode(readme.content)
        if not text:
            return None
        
        base_url = f"https://raw.githubusercontent.com/{self.config.repo_name}/{self.config.branch}/{path}"
        
        # Collect images
        images = [f for f in contents if self.images.is_image(f.name)]
        for item in contents:
            if item.type == "dir" and item.name.lower() in IMAGE_FOLDERS:
                try:
                    sub = self._repo.get_contents(item.path, ref=self.config.branch)
                    images.extend(f for f in sub if self.images.is_image(f.name))
                except GithubException:
                    pass
        
        best = self.images.select_best(images)
        parts = path.split("/")
        
        return CrawlResult(
            category=parts[-2] if len(parts) >= 2 else "general",
            project_name=parts[-1],
            title=self.markdown.extract_title(text),
            description=self.markdown.extract_description(text),
            readme_content=self.markdown.fix_links(text, base_url),
            diagram_url=f"https://raw.githubusercontent.com/{self.config.repo_name}/{self.config.branch}/{best.path}" if best else None,
            source_path=path,
        )
    
    def _decode(self, content: str) -> Optional[str]:
        try:
            raw = base64.b64decode(content)
            return raw.decode("utf-8")
        except (UnicodeDecodeError, Exception):
            return None
