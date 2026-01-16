# Imports
import logging
import os
from typing import Optional, List

import marko
from marko.md_renderer import MarkdownRenderer
from github.ContentFile import ContentFile

from urllib.parse import urljoin, urlparse

from isaac_scraper.constants import (
    ABSOLUTE_URL_PREFIXES,
    VALID_URL_SCHEMES,
    IMAGE_EXTENSIONS,
    IMAGE_PRIORITY_KEYWORDS,
    IMAGE_IGNORE_KEYWORDS
)

logger = logging.getLogger(__name__)


# Link Transformation
class LinkTransformer(MarkdownRenderer):
    
    def __init__(self, base_url: str):
        super().__init__()
        parsed = urlparse(base_url)
        if parsed.scheme not in VALID_URL_SCHEMES:
            raise ValueError(f"Invalid URL scheme: {base_url}")
        self.base_url = base_url.rstrip("/")
    
    def render_link(self, element) -> str:
        dest = element.dest
        if dest and not dest.startswith(ABSOLUTE_URL_PREFIXES):
            # Change: use removeprefix to avoid stripping needed characters
            dest = dest.removeprefix("./")
            dest = urljoin(self.base_url + "/", dest)
        title = f' "{element.title}"' if element.title else ""
        text = self.render_children(element)
        return f"[{text}]({dest}{title})"
    
    def render_image(self, element) -> str:
        dest = element.dest
        if not dest.startswith(ABSOLUTE_URL_PREFIXES):
            # Change: use removeprefix here as well
            dest = dest.removeprefix("./")
            dest = urljoin(self.base_url + "/", dest)
        alt = self._extract_alt_text(element)
        title = f' "{element.title}"' if element.title else ""
        return f"![{alt}]({dest}{title})"
    
    def _extract_alt_text(self, element) -> str:
        if element.children:
            first_child = element.children[0]
            if hasattr(first_child, "children"):
                return first_child.children if isinstance(first_child.children, str) else ""
        return ""


# Markdown processing
class MarkdownProcessor:

    def __init__(self):
        self._parser = marko.parser.Parser()
    
    def fix_links(self, content: str, base_url: str) -> str:
        doc = self._parser.parse(content)
        return LinkTransformer(base_url).render(doc)
    
    def extract_title(self, content: str) -> Optional[str]:
        for child in self._parser.parse(content).children:
            if child.get_type() == "Heading" and child.level == 1:
                return self._flatten_text(child).strip()
        return None
    
    def extract_description(self, content: str) -> Optional[str]:
        for child in self._parser.parse(content).children:
            if child.get_type() == "Paragraph":
                if self._is_badge_paragraph(child):
                    continue
                text = self._flatten_text(child).strip()
                if text:
                    return text[:297] + "..." if len(text) > 300 else text
        return None
    
    def _is_badge_paragraph(self, paragraph) -> bool:
        """Check if paragraph contains only images/badges."""
        if not hasattr(paragraph, "children") or not paragraph.children:
            return False
        return all(
            self._is_image_or_link_image(c) for c in paragraph.children
            if not (hasattr(c, "children") and isinstance(c.children, str) and not c.children.strip())
        )
    
    def _is_image_or_link_image(self, el) -> bool:
        """Check if element is image or link containing only image."""
        el_type = el.get_type() if hasattr(el, "get_type") else None
        if el_type == "Image":
            return True
        if el_type == "Link" and hasattr(el, "children"):
            return all(c.get_type() == "Image" for c in el.children if hasattr(c, "get_type"))
        return False
    
    def _flatten_text(self, el) -> str:

        if not hasattr(el, "children"):
            logger.debug(f"Unexpected element structure (no children): {type(el)}")
            return ""
        
        if isinstance(el.children, str):
            return el.children
        
        if not isinstance(el.children, (list, tuple)):
            logger.debug(f"Unexpected children type: {type(el.children)}")
            return ""
        
        return "".join(self._flatten_text(c) for c in el.children)


# Image selection
class ImageSelector:
    
    def is_image(self, name: str) -> bool:
        return name.lower().endswith(tuple(IMAGE_EXTENSIONS.keys()))
    
    def select_best(self, images: List[ContentFile]) -> Optional[ContentFile]:
        if not images:
            return None
        
        candidates = [i for i in images if not any(k in i.name.lower() for k in IMAGE_IGNORE_KEYWORDS)]
        if not candidates:
            return images[0]
        
        def score(img):
            name = img.name.lower()
            ext = os.path.splitext(name)[1]
            priority_score = any(k in name for k in IMAGE_PRIORITY_KEYWORDS)
            ext_score = IMAGE_EXTENSIONS.get(ext, 0)
            return (priority_score, ext_score, img.size or 0)
        
        return max(candidates, key=score)
