import base64
import json
import os
import re
import logging
from typing import List, Optional, Dict
from urllib.parse import quote
from github import Github, Auth
from github.ContentFile import ContentFile  
from dotenv import load_dotenv

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class Config:
    # App config
    GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
    REPO_NAME = "donnemartin/system-design-primer"
    START_PATH = "solutions"
    OUTPUT_FILE = "isaac_raw_data.json"
    BRANCH = "master"

class MarkdownProcessor:
    # Link transformation
    @staticmethod
    def fix_links(content: str, base_url: str) -> str:
        def replace_link(match):
            prefix, path = match.group(1), match.group(2)
            if path.startswith(('http', 'https', '#', 'mailto:')):
                return match.group(0)
            return f'{prefix}({base_url}/{path.lstrip("./")})'

        # Markdown links
        content = re.sub(r'(\[.*?\])\((.*?)\)', replace_link, content)
        
        # HTML images
        content = re.sub(
            r'src="(.*?)"', 
            lambda m: m.group(0) if m.group(1).startswith('http') 
            else f'src="{base_url}/{m.group(1).lstrip("./")}"', 
            content
        )
        return content

class ImageAnalyst:
    # Diagram selection
    PRIORITY = ['architecture', 'system', 'diagram', 'flow', 'design', 'overview']
    IGNORE = ['icon', 'badge', 'logo', 'button', 'screenshot', 'demo']

    @classmethod
    def select_best_image(cls, images: List[ContentFile]) -> Optional[ContentFile]:
        candidates = [
            img for img in images 
            if not any(k in img.name.lower() for k in cls.IGNORE)
        ]
        if not candidates:
            return None

        # Sort priority
        candidates.sort(key=lambda x: (
            not any(k in x.name.lower() for k in cls.PRIORITY),
            len(x.name)
        ))
        return candidates[0]

class GitHubSource:
    # API wrapper
    def __init__(self, token: Optional[str], repo_name: str):
        if token:
            # Modern auth
            auth = Auth.Token(token)
            self.g = Github(auth=auth)
        else:
            self.g = Github()
        self.repo = self.g.get_repo(repo_name)

    def get_contents(self, path: str):
        return self.repo.get_contents(path)

    def get_raw_url(self, path: str) -> str:
        return f"https://raw.githubusercontent.com/{Config.REPO_NAME}/{Config.BRANCH}/{quote(path)}"

class IsaacCrawler:
    # Process management
    def __init__(self, source: GitHubSource):
        self.source = source
        self.processor = MarkdownProcessor()
        self.analyst = ImageAnalyst()
        self.results = []

    def _process_folder(self, contents: List[ContentFile], path: str) -> Optional[Dict]:
        readme = next((f for f in contents if f.name.lower() == "readme.md"), None)
        if not readme:
            return None

        # Image collection
        images = [f for f in contents if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))]
        
        # Check subfolders
        for item in contents:
            if item.type == "dir" and item.name.lower() in ['img', 'images', 'assets']:
                try:
                    sub_files = self.source.get_contents(item.path)
                    images.extend([f for f in sub_files if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))])
                except Exception:
                    continue

        try:
            # Data extraction
            raw_text = base64.b64decode(readme.content).decode('utf-8')
            base_url = f"https://raw.githubusercontent.com/{Config.REPO_NAME}/{Config.BRANCH}/{path}"
            
            processed_text = self.processor.fix_links(raw_text, base_url)
            best_img = self.analyst.select_best_image(images)
            
            parts = path.split('/')
            return {
                "category": parts[1] if len(parts) > 1 else "Root",
                "project_name": parts[-1],
                "readme_content": processed_text,
                "diagram_url": self.source.get_raw_url(best_img.path) if best_img else None,
                "source_path": path
            }
        except Exception as e:
            logger.error(f"Error {path}: {e}")
            return None

    def start(self, path: str):
        # Recursive crawl
        try:
            items = self.source.get_contents(path)
            has_readme = any(f.name.lower() == "readme.md" for f in items)

            if has_readme and path != Config.START_PATH:
                data = self._process_folder(items, path)
                if data:
                    self.results.append(data)
                    logger.info(f"Done: {data['project_name']}")
            else:
                for item in items:
                    if item.type == "dir" and not item.name.startswith('.'):
                        self.start(item.path)
        except Exception as e:
            logger.error(f"Access error {path}: {e}")

    def save(self, filename: str):
        # Final save
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved {len(self.results)} items")

if __name__ == "__main__":
    # Execution entry
    source = GitHubSource(Config.GITHUB_TOKEN, Config.REPO_NAME)
    crawler = IsaacCrawler(source)
    
    logger.info("Crawler started")
    crawler.start(Config.START_PATH)
    crawler.save(Config.OUTPUT_FILE)