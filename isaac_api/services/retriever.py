import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from flashrank import Ranker, RerankRequest
from langchain_postgres import PGVector
from langchain_core.documents import Document

from isaac_api.core.database import get_vector_store
from isaac_api.models.schemas import (
    RetrievalResponse,
    DocumentChunk,
    ImageReference,
)

logger = logging.getLogger(__name__)

#Constants
INITIAL_RECALL_K = 25
FINAL_PRECISION_K = 5
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
FLASHRANK_CACHE_DIR = Path.home() / ".cache" / "flashrank"

IMG_REF_PATTERN = re.compile(
    r"\[IMAGE\s+(?P<id>IMG_[a-fA-F0-9]+)\]\s*\n\*\*Description:\*\*\s*(?P<desc>[^\n]+(?:\n(?!\n)[^\n]*)*)",
    re.MULTILINE,
)


class _RankerSingleton:
    #Lazy-loaded singleton to avoid model reload per request.

    _instance: Optional[Ranker] = None

    @classmethod
    def get(cls) -> Ranker:
        if cls._instance is None:
            logger.info(f"Loading reranker model: {RERANKER_MODEL}")
            cls._instance = Ranker(model_name=RERANKER_MODEL, cache_dir=str(FLASHRANK_CACHE_DIR))
        return cls._instance


class RetrieverService:
    #2-stage retrieval: vector recall → FlashRank reranking.

    def __init__(self, vector_store: PGVector) -> None:
        self._vector_store = vector_store
        self._ranker = _RankerSingleton.get()

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResponse:
        final_k = k or FINAL_PRECISION_K
        logger.info(f"Search: '{query[:50]}…' | recall={INITIAL_RECALL_K}, final={final_k}")

        candidates = self._fetch_candidates(query, filter_metadata)

        if not candidates:
            return self._empty_response(query)

        reranked = self._rerank_results(query, candidates, top_k=final_k)

        chunks, images = self._build_chunks(reranked)

        return RetrievalResponse(
            query=query,
            total_results=len(chunks),
            chunks=chunks,
            images=images,
            has_more=len(candidates) > final_k,
        )

    def _fetch_candidates(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> List[Document]:
        effective_filter = filter_metadata if filter_metadata else None

        try:
            results = self._vector_store.similarity_search(
                query=query,
                k=INITIAL_RECALL_K,
                filter=effective_filter,
            )
            logger.debug(f"Fetched {len(results)} candidates from vector store")
            return results
        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            raise RuntimeError("Vector search failed") from exc

    def _rerank_results(
        self,
        query: str,
        candidates: List[Document],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        passages = [
            {"id": idx, "text": doc.page_content}
            for idx, doc in enumerate(candidates)
        ]

        try:
            rerank_request = RerankRequest(query=query, passages=passages)
            ranked = self._ranker.rerank(rerank_request)

            results: List[Tuple[Document, float]] = []
            for item in ranked[:top_k]:
                doc_idx = item["id"]
                score = item["score"]
                results.append((candidates[doc_idx], score))

            logger.debug(f"Reranked to {len(results)} documents")
            return results

        except Exception as exc:
            logger.warning(f"Reranking failed, returning vector results: {exc}")
            return [(doc, 0.0) for doc in candidates[:top_k]]

    def _build_chunks(
        self,
        results: List[Tuple[Document, float]],
    ) -> Tuple[List[DocumentChunk], List[ImageReference]]:
        chunks: List[DocumentChunk] = []
        images: List[ImageReference] = []
        seen_image_ids: set[str] = set()

        for idx, (doc, score) in enumerate(results):
            content = doc.page_content
            metadata = doc.metadata or {}

            chunk_images = self._extract_images(content, idx)
            for img in chunk_images:
                if img.image_id not in seen_image_ids:
                    images.append(img)
                    seen_image_ids.add(img.image_id)

            chunks.append(
                DocumentChunk(
                    content=content,
                    metadata=metadata,
                    score=round(score, 4),
                    has_images=bool(chunk_images),
                )
            )

        return chunks, images

    def _extract_images(self, content: str, chunk_index: int) -> List[ImageReference]:
        images: List[ImageReference] = []

        for match in IMG_REF_PATTERN.finditer(content):
            images.append(
                ImageReference(
                    image_id=match.group("id"),
                    description=match.group("desc").strip(),
                    source_chunk_index=chunk_index,
                )
            )

        return images

    def _empty_response(self, query: str) -> RetrievalResponse:
        return RetrievalResponse(
            query=query,
            total_results=0,
            chunks=[],
            images=[],
            has_more=False,
        )


def get_retriever_service() -> RetrieverService:
    vector_store = get_vector_store()
    return RetrieverService(vector_store)
