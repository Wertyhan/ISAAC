# Markdown & Image Processing
import os
from typing import Optional, List

import marko
from marko.md_renderer import MarkdownRenderer
from github.ContentFile import ContentFile


class LinkTransformer(MarkdownRenderer):
    """Transforms relative links to absolute GitHub raw URLs."""
    
    ABSOLUTE = ("http://", "https://", "#", "mailto:", "data:")
    
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url.rstrip("/")
    
    def render_link(self, element) -> str:
        dest = element.dest
        if not dest.startswith(self.ABSOLUTE):
            dest = f"{self.base_url}/{dest.lstrip('./')}"
        title = f' "{element.title}"' if element.title else ""
        return f"[{self.render_children(element)}]({dest}{title})"
    
    def render_image(self, element) -> str:
        dest = element.dest
        if not dest.startswith(self.ABSOLUTE):
            dest = f"{self.base_url}/{dest.lstrip('./')}"
        alt = element.children[0].children if element.children else ""
        title = f' "{element.title}"' if element.title else ""
        return f"![{alt}]({dest}{title})"


class MarkdownProcessor:
    """Parses markdown using Marko AST."""
    
    def __init__(self):
        self._parser = marko.parser.Parser()
    
    def fix_links(self, content: str, base_url: str) -> str:
        doc = self._parser.parse(content)
        return LinkTransformer(base_url).render(doc)
    
    def extract_title(self, content: str) -> Optional[str]:
        for child in self._parser.parse(content).children:
            if child.get_type() == "Heading" and child.level == 1:
                return self._text(child).strip()
        return None
    
    def extract_description(self, content: str) -> Optional[str]:
        for child in self._parser.parse(content).children:
            if child.get_type() == "Paragraph":
                text = self._text(child).strip()
                return text[:297] + "..." if len(text) > 300 else text
        return None
    
    def _text(self, el) -> str:
        if hasattr(el, "children"):
            if isinstance(el.children, str):
                return el.children
            return "".join(self._text(c) for c in el.children)
        return ""


class ImageSelector:
    """Selects best architecture diagram from images."""
    
    PRIORITY = {"architecture", "system", "diagram", "flow", "design", "overview"}
    IGNORE = {"icon", "badge", "logo", "button", "screenshot", "avatar"}
    FORMATS = (".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp")
    SCORES = {".svg": 4, ".png": 3, ".webp": 2, ".gif": 2, ".jpg": 1, ".jpeg": 1}
    
    @classmethod
    def is_image(cls, name: str) -> bool:
        return name.lower().endswith(cls.FORMATS)
    
    @classmethod
    def select_best(cls, images: List[ContentFile]) -> Optional[ContentFile]:
        if not images:
            return None
        
        candidates = [i for i in images if not any(k in i.name.lower() for k in cls.IGNORE)]
        if not candidates:
            return images[0]
        
        def score(img):
            name = img.name.lower()
            ext = os.path.splitext(name)[1]
            return (any(k in name for k in cls.PRIORITY), cls.SCORES.get(ext, 0), img.size or 0)
        
        return max(candidates, key=score)
