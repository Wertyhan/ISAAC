"""Text Processor - Cleaning, tokenization, and chunking."""

import logging
import re
from datetime import datetime
from typing import List, Dict, Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from isaac_ingestion.config import Config, generate_chunk_id

logger = logging.getLogger(__name__)

MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

IMAGE_TOKEN_TEMPLATE = "\n\n[IMAGE {image_id}]\n**Description:** {description}\n\n"


class TextProcessor:
    """Text cleaning, image token replacement, and chunking."""
    
    def __init__(self, config: Config):
        self._config = config
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS,
            strip_headers=False,
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    
    def replace_link_with_token(
        self,
        text: str,
        image_url: str,
        image_id: str,
        description: str,
    ) -> str:
        """Replace image markdown link with descriptive token."""
        pattern = self._build_image_pattern(image_url)
        
        token = IMAGE_TOKEN_TEMPLATE.format(
            image_id=image_id,
            description=description,
        )
        
        result, count = re.subn(pattern, token, text)
        
        if count > 0:
            logger.debug(f"Replaced {count} occurrence(s) of image: {image_id}")
        else:
            logger.debug(f"Image URL not found in text: {image_url[:50]}...")
        
        return result

    def _build_image_pattern(self, image_url: str) -> str:
        escaped_url = re.escape(image_url)
        return rf'!\[[^\]]*\]\({escaped_url}(?:\s*"[^"]*")?\)'
    
    def create_chunks(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Document]:
        """Split text into chunks with two-stage approach: headers then size.
        
        Each chunk receives:
        - chunk_id: Unique identifier (CHK_{doc_hash}_{index})
        - doc_id: Parent document ID
        - chunk_index: Position in document
        - section: Header context (h1, h2, etc.)
        - All base metadata (project_name, category, source_uri, created_at)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        doc_id = metadata.get("doc_id", "DOC_unknown")
        
        # Split by headers
        header_docs = self._header_splitter.split_text(text)
        
        # Split large sections
        final_docs: List[Document] = []
        
        for i, header_doc in enumerate(header_docs):
            # Merge header metadata with base metadata
            chunk_metadata = {
                **metadata,
                **header_doc.metadata,
            }
            
            content = header_doc.page_content
            
            if len(content) > self._config.chunk_size:
                sub_chunks = self._text_splitter.split_text(content)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunk_index = len(final_docs)
                    chunk_id = generate_chunk_id(doc_id, chunk_index)
                    section = chunk_metadata.get("h2") or chunk_metadata.get("h1") or None
                    
                    final_docs.append(Document(
                        page_content=sub_chunk,
                        metadata={
                            **chunk_metadata,
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_index,
                            "sub_chunk": j,
                            "section": section,
                        },
                    ))
            else:
                chunk_index = len(final_docs)
                chunk_id = generate_chunk_id(doc_id, chunk_index)
                section = chunk_metadata.get("h2") or chunk_metadata.get("h1") or None
                
                final_docs.append(Document(
                    page_content=content,
                    metadata={
                        **chunk_metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "section": section,
                    },
                ))
        
        # Add total_chunks count to all chunks
        total_chunks = len(final_docs)
        for doc in final_docs:
            doc.metadata["total_chunks"] = total_chunks
        
        logger.debug(f"Created {len(final_docs)} chunks from text ({len(text)} chars)")
        return final_docs
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize markdown text."""
        if not text:
            return ""
        
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Remove excessive blank lines (keep max 2)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        
        # Remove trailing whitespace from lines
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_image_urls(self, text: str) -> List[str]:
        """Extract unique image URLs from markdown text."""
        pattern = r'!\[[^\]]*\]\(([^)\s]+)(?:\s*"[^"]*")?\)'
        
        matches = re.findall(pattern, text)
        
        # Filter and deduplicate
        urls = []
        seen = set()
        for url in matches:
            if url not in seen and url.startswith(("http://", "https://")):
                urls.append(url)
                seen.add(url)
        
        logger.debug(f"Extracted {len(urls)} image URLs from text")
        return urls
