import logging
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from isaac_api.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    #Singleton manager for PGVector connections.
    
    _instance: Optional["VectorStoreManager"] = None
    _vector_store: Optional[PGVector] = None

    def __new__(cls) -> "VectorStoreManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._settings: Settings = get_settings()

    def get_store(self) -> PGVector:
        if self._vector_store is None:
            self._vector_store = self._create_store()
        return self._vector_store

    def _create_store(self) -> PGVector:
        logger.info(f"Initializing vector store connection to '{self._settings.collection_name}'")

        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=self._settings.embedding_model,
                google_api_key=self._settings.gemini_api_key.get_secret_value(),
            )

            vector_store = PGVector(
                embeddings=embeddings,
                collection_name=self._settings.collection_name,
                connection=self._settings.postgres_connection_string,
                use_jsonb=True,
            )

            logger.info("Vector store connection established successfully")
            return vector_store

        except Exception as e:
            logger.error(f"Failed to connect to vector store: {e}")
            raise ConnectionError(f"Database connection failed: {e}") from e

    def close(self) -> None:
        if self._vector_store is not None:
            logger.info("Closing vector store connection")
            self._vector_store = None


_manager: Optional[VectorStoreManager] = None


def get_vector_store() -> PGVector:
    global _manager
    if _manager is None:
        _manager = VectorStoreManager()
    return _manager.get_store()


def close_vector_store() -> None:
    global _manager
    if _manager is not None:
        _manager.close()
        _manager = None
