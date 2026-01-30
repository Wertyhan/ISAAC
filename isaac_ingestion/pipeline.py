"""Ingestion Pipeline - ETL from raw JSON to vector store."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Iterable

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

from isaac_ingestion.config import Config, ImageMeta, generate_doc_id
from isaac_ingestion.clients.gemini_client import GeminiVisionClient
from isaac_ingestion.services.image_manager import ImageManager
from isaac_ingestion.services.text_processor import TextProcessor
from isaac_ingestion.services.image_registry import ImageRegistry, get_image_registry
from isaac_ingestion.exceptions import (
    IngestionError,
    InvalidInputError,
    DatabaseConnectionError,
    VectorStoreError,
)

logger = logging.getLogger(__name__)

# GitHub base URL for source URI generation
GITHUB_BASE_URL = "https://github.com/donnemartin/system-design-primer/tree/master"


class IngestionPipeline:
    """ETL pipeline: JSON -> images -> chunks -> vector store with image registry."""
    
    def __init__(
        self,
        config: Config,
        gemini_client: GeminiVisionClient,
        image_manager: ImageManager,
        text_processor: TextProcessor,
        image_registry: Optional[ImageRegistry] = None,
    ):
        self._config = config
        self._gemini = gemini_client
        self._images = image_manager
        self._text = text_processor
        self._image_registry = image_registry or get_image_registry()
        self._vector_store: Optional[PGVector] = None
        self._stats = {
            "projects_processed": 0,
            "images_downloaded": 0,
            "images_described": 0,
            "images_registered": 0,
            "chunks_created": 0,
            "errors": 0,
        }
    
    def run(self, progress_wrapper: Optional[Callable[[Iterable], Iterable]] = None) -> Dict[str, int]:
        """Execute the full pipeline, return stats."""
        logger.info("Starting ingestion pipeline")
        
        # Load raw data
        raw_data = self._load_raw_data()
        logger.info(f"Loaded {len(raw_data)} projects from {self._config.raw_data_file}")
        
        # Initialize vector store
        self._init_vector_store()
        
        # Process each project
        all_documents: List[Document] = []
        
        iterator = progress_wrapper(raw_data) if progress_wrapper else raw_data
        
        for project_data in iterator:
            try:
                docs = self._process_project(project_data)
                all_documents.extend(docs)
                self._stats["projects_processed"] += 1
            except Exception as e:
                logger.error(f"Failed to process project: {e}")
                self._stats["errors"] += 1
        
        # Persist to vector store
        if all_documents:
            self._persist_documents(all_documents)
        
        # Commit image registry
        self._image_registry.commit()
        logger.info(f"Image registry stats: {self._image_registry.stats()}")
        
        logger.info(f"Pipeline complete. Stats: {self._stats}")
        return dict(self._stats)
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        try:
            with open(self._config.raw_data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as e:
            raise InvalidInputError(f"Raw data file not found: {self._config.raw_data_file}") from e
        except json.JSONDecodeError as e:
            raise InvalidInputError(f"Invalid JSON in raw data file: {e}") from e
        
        if not isinstance(data, list):
            raise InvalidInputError("Raw data must be a JSON array")
        
        return data
    
    def _init_vector_store(self) -> None:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=self._config.embedding_model,
                google_api_key=self._config.gemini_api_key.get_secret_value(),
            )
            
            self._vector_store = PGVector(
                embeddings=embeddings,
                collection_name=self._config.collection_name,
                connection=self._config.postgres_connection_string,
                use_jsonb=True,
            )
            logger.info(f"Vector store initialized: {self._config.collection_name}")
            
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to connect to vector store: {e}") from e
    
    def _process_project(self, project_data: Dict[str, Any]) -> List[Document]:
        """Process a single project into document chunks with full traceability."""
        metadata = self._extract_project_metadata(project_data)
        content = self._prepare_content(project_data, metadata)
        chunks = self._text.create_chunks(content, metadata)
        self._stats["chunks_created"] += len(chunks)
        return chunks

    def _extract_project_metadata(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive metadata for full traceability."""
        project_name = project_data.get("project_name", "unknown")
        source_path = project_data.get("source_path", "")
        
        # Generate stable document ID
        doc_id = generate_doc_id(project_name, source_path)
        
        # Build source URI for citation
        source_uri = f"{GITHUB_BASE_URL}/{source_path}" if source_path else ""
        
        return {
            "doc_id": doc_id,
            "project_name": project_name,
            "title": project_data.get("title", project_name),
            "category": project_data.get("category", "general"),
            "source_path": source_path,
            "source_uri": source_uri,
            "source": "isaac_scraper",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": project_data.get("description", ""),
        }

    def _prepare_content(self, project_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        project_name = metadata["project_name"]
        doc_id = metadata["doc_id"]
        content = project_data.get("readme_content", "")
        diagram_url = project_data.get("diagram_url")

        logger.debug(f"Processing: {project_name} (doc_id: {doc_id})")

        # Clean content
        content = self._text.clean_text(content)

        # Extract and process images
        image_urls = self._text.extract_image_urls(content)

        # Add diagram_url if present and not already in content
        if diagram_url and diagram_url not in image_urls:
            image_urls.insert(0, diagram_url)

        # Download and describe images (pass doc_id for linking)
        for url in image_urls:
            content = self._process_image(content, url, project_name, doc_id)
        
        return content
    
    def _process_image(self, content: str, url: str, project_name: str, doc_id: str) -> str:
        """Download image and replace link with description token."""
        # Download (pass doc_id for metadata)
        image_meta = self._images.download(url, project_name, doc_id=doc_id)
        if not image_meta:
            return content
        
        self._stats["images_downloaded"] += 1
        
        # Generate description
        description = self._gemini.generate_description(image_meta.local_path)
        self._stats["images_described"] += 1
        
        # Register image in registry for independent retrieval
        file_size = image_meta.local_path.stat().st_size if image_meta.local_path.exists() else None
        self._image_registry.register(
            image_id=image_meta.image_id,
            doc_id=doc_id,
            project_name=project_name,
            source_uri=url,
            filepath=str(image_meta.local_path),
            file_hash=image_meta.file_hash,
            description=description,
            caption=image_meta.caption,
            file_size=file_size,
        )
        self._stats["images_registered"] += 1
        
        # Replace in content
        content = self._text.replace_link_with_token(
            text=content,
            image_url=url,
            image_id=image_meta.image_id,
            description=description,
        )
        
        return content
    
    def _persist_documents(self, documents: List[Document]) -> None:
        if not self._vector_store:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            self._vector_store.add_documents(documents)
            logger.info(f"Persisted {len(documents)} documents to vector store")
        except Exception as e:
            raise VectorStoreError(f"Failed to persist documents: {e}") from e
    
    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)
    
    def close(self) -> None:
        self._gemini.close()
        self._images.close()
        logger.debug("Pipeline resources closed")
    
    def __enter__(self) -> "IngestionPipeline":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# Factory
def create_pipeline(config: Optional[Config] = None) -> IngestionPipeline:
    if config is None:
        config = Config()
    
    return IngestionPipeline(
        config=config,
        gemini_client=GeminiVisionClient(config),
        image_manager=ImageManager(config),
        text_processor=TextProcessor(config),
    )
