# Imports
import json
import logging
import os
import tempfile
from typing import List

from isaac_scraper.config import CrawlResult

logger = logging.getLogger(__name__)


# Implementation
class ResultWriter:
    """Handles writing crawl results to disk with atomic operations."""
    
    def __init__(self, output_file: str):
        self._output_file = output_file
    
    def write(self, results: List[CrawlResult]) -> None:
        """Write results to file atomically."""
        output_dir = os.path.dirname(self._output_file) or "."
        os.makedirs(output_dir, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(
            "w", 
            encoding="utf-8", 
            dir=output_dir, 
            suffix=".tmp", 
            delete=False
        ) as f:
            json.dump(
                [r.model_dump() for r in results], 
                f, 
                ensure_ascii=False, 
                indent=2
            )
            temp_path = f.name
        
        os.replace(temp_path, self._output_file)
        logger.info(f"Saved {len(results)} items to {self._output_file}")
