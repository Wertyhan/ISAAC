import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from isaac_api.services.retriever import (
    RetrieverService,
    INITIAL_RECALL_K,
    FINAL_PRECISION_K,
    IMG_REF_PATTERN,
)
from isaac_api.models.schemas import RetrievalResponse


@pytest.fixture
def mock_vector_store():
    return MagicMock()


@pytest.fixture
def mock_ranker():
    ranker = MagicMock()
    ranker.rerank.return_value = [
        {"id": 0, "score": 0.95},
        {"id": 1, "score": 0.85},
        {"id": 2, "score": 0.75},
    ]
    return ranker


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="Twitter uses fanout-on-write for home timeline delivery", metadata={"source": "twitter_design"}),
        Document(page_content="Rate limiting prevents abuse using token bucket algorithm", metadata={"source": "rate_limiter"}),
        Document(page_content="Consistent hashing distributes load across cache nodes", metadata={"source": "distributed_cache"}),
    ]


@pytest.fixture
def document_with_image():
    content = """The load balancer distributes incoming requests across multiple application servers.

[IMAGE IMG_abc123def456]
**Description:** Architecture diagram showing load balancer with round-robin distribution to 3 app servers

This ensures high availability and horizontal scaling."""
    return Document(page_content=content, metadata={"source": "load_balancer_design"})


@pytest.fixture
def retriever(mock_vector_store, mock_ranker):
    with patch("isaac_api.services.retriever._RankerSingleton.get", return_value=mock_ranker):
        return RetrieverService(mock_vector_store)


class TestRetrieverServiceSearch:
    def test_search_returns_reranked_results(self, retriever, mock_vector_store, sample_documents):
        mock_vector_store.similarity_search.return_value = sample_documents

        result = retriever.search("how does Twitter handle timeline fanout", k=3)

        assert isinstance(result, RetrievalResponse)
        assert result.query == "how does Twitter handle timeline fanout"
        assert result.total_results == 3
        assert len(result.chunks) == 3

    def test_search_uses_default_k_when_not_provided(self, retriever, mock_vector_store, sample_documents):
        mock_vector_store.similarity_search.return_value = sample_documents

        result = retriever.search("distributed cache consistency")

        assert result.total_results == min(len(sample_documents), FINAL_PRECISION_K)

    def test_search_returns_empty_response_when_no_candidates(self, retriever, mock_vector_store):
        mock_vector_store.similarity_search.return_value = []

        result = retriever.search("CAP theorem tradeoffs")

        assert result.total_results == 0
        assert result.chunks == []
        assert result.images == []
        assert result.has_more is False

    def test_search_passes_filter_metadata(self, retriever, mock_vector_store, sample_documents):
        mock_vector_store.similarity_search.return_value = sample_documents
        filters = {"category": "caching"}

        retriever.search("cache invalidation strategies", filter_metadata=filters)

        # Query expansion may modify the query, so we check the filter is passed correctly
        call_args = mock_vector_store.similarity_search.call_args
        assert call_args.kwargs.get("filter") == filters
        assert call_args.kwargs.get("k") == INITIAL_RECALL_K

    def test_search_converts_empty_filter_to_none(self, retriever, mock_vector_store, sample_documents):
        mock_vector_store.similarity_search.return_value = sample_documents

        retriever.search("database sharding patterns", filter_metadata={})

        # Query expansion may modify the query, so we check the filter is None
        call_args = mock_vector_store.similarity_search.call_args
        assert call_args.kwargs.get("filter") is None
        assert call_args.kwargs.get("k") == INITIAL_RECALL_K


class TestRetrieverServiceReranking:
    def test_rerank_orders_by_score(self, retriever, mock_vector_store, mock_ranker, sample_documents):
        mock_vector_store.similarity_search.return_value = sample_documents
        mock_ranker.rerank.return_value = [
            {"id": 2, "score": 0.99},
            {"id": 0, "score": 0.50},
            {"id": 1, "score": 0.25},
        ]

        result = retriever.search("consistent hashing for distributed systems", k=3)

        assert result.chunks[0].content == "Consistent hashing distributes load across cache nodes"
        assert result.chunks[0].score == pytest.approx(0.99)

    def test_rerank_fallback_on_error(self, retriever, mock_vector_store, mock_ranker, sample_documents):
        mock_vector_store.similarity_search.return_value = sample_documents
        mock_ranker.rerank.side_effect = Exception("Rerank failed")

        result = retriever.search("message queue design", k=3)

        assert result.total_results == 3
        assert all(chunk.score == pytest.approx(0.0) for chunk in result.chunks)


class TestImageExtraction:
    def test_extract_images_from_content(self, retriever, mock_vector_store, mock_ranker, document_with_image):
        mock_vector_store.similarity_search.return_value = [document_with_image]
        mock_ranker.rerank.return_value = [{"id": 0, "score": 0.9}]

        result = retriever.search("load balancer architecture", k=1)

        assert len(result.images) == 1
        assert result.images[0].image_id == "IMG_abc123def456"
        assert "load balancer" in result.images[0].description
        assert result.chunks[0].has_images is True

    def test_no_duplicate_images_across_chunks(self, retriever, mock_vector_store, mock_ranker):
        same_image_content = "[IMAGE IMG_abc123def456]\n**Description:** System architecture overview"
        docs = [
            Document(page_content=f"Overview section: {same_image_content}", metadata={}),
            Document(page_content=f"Details section: {same_image_content}", metadata={}),
        ]
        mock_vector_store.similarity_search.return_value = docs
        mock_ranker.rerank.return_value = [
            {"id": 0, "score": 0.9},
            {"id": 1, "score": 0.8},
        ]

        result = retriever.search("system architecture diagram", k=2)

        assert len(result.images) == 1

    def test_image_pattern_matches_valid_format(self):
        valid_content = "[IMAGE IMG_abc123def456]\n**Description:** Test description"
        match = IMG_REF_PATTERN.search(valid_content)

        assert match is not None
        assert match.group("id") == "IMG_abc123def456"
        assert match.group("desc") == "Test description"

    def test_image_pattern_rejects_invalid_format(self):
        invalid_contents = [
            "[IMAGE abc123]",
            "[IMG IMG_abc123]",
            "[IMAGE IMG_abc123] Description without format",
        ]

        for content in invalid_contents:
            assert IMG_REF_PATTERN.search(content) is None


class TestFetchCandidates:
    def test_fetch_raises_runtime_error_on_db_failure(self, retriever, mock_vector_store):
        mock_vector_store.similarity_search.side_effect = Exception("DB error")

        with pytest.raises(RuntimeError, match="Vector search failed"):
            retriever.search("microservices communication patterns")


class TestHasMoreFlag:
    def test_has_more_true_when_candidates_exceed_k(self, retriever, mock_vector_store, mock_ranker):
        docs = [Document(page_content=f"Scaling pattern {i}: horizontal vs vertical", metadata={}) for i in range(10)]
        mock_vector_store.similarity_search.return_value = docs
        mock_ranker.rerank.return_value = [{"id": i, "score": 0.9 - i * 0.1} for i in range(5)]

        result = retriever.search("scaling strategies for high traffic", k=5)

        assert result.has_more is True

    def test_has_more_false_when_candidates_fit_in_k(self, retriever, mock_vector_store, mock_ranker):
        docs = [Document(page_content=f"API gateway pattern {i}", metadata={}) for i in range(3)]
        mock_vector_store.similarity_search.return_value = docs
        mock_ranker.rerank.return_value = [{"id": i, "score": 0.9 - i * 0.1} for i in range(3)]

        result = retriever.search("API gateway design", k=5)

        assert result.has_more is False
