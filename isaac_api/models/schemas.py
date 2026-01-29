from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class DocumentMetadata(BaseModel):
    """Full document metadata for traceability."""
    
    doc_id: str = Field(
        ...,
        description="Unique document identifier (hash-based)",
        examples=["DOC_abc123def456"],
    )
    title: str = Field(
        ...,
        description="Document title",
    )
    source_uri: str = Field(
        ...,
        description="Original source URL or path",
    )
    created_at: datetime = Field(
        ...,
        description="Document ingestion timestamp",
    )
    project_name: str = Field(
        ...,
        description="Project/source name",
    )
    category: str = Field(
        default="general",
        description="Document category",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Index of this chunk within the document",
    )
    total_chunks: int = Field(
        default=1,
        ge=1,
        description="Total number of chunks in the source document",
    )
    section: Optional[str] = Field(
        default=None,
        description="Section header if available",
    )


class QueryRequest(BaseModel):
    """Search query request."""
    
    model_config = {"validate_assignment": True}
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The search query text",
        examples=["How does Twitter handle timeline fanout?"],
    )
    k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of documents to retrieve (k=10 recommended for reranking)",
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters (e.g., {'category': 'caching'})",
        json_schema_extra={"example": None},
    )
    
    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or contain only whitespace")
        return stripped

    @field_validator("filter_metadata", mode="before")
    @classmethod
    def sanitize_filter_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        #Converts empty dict or Swagger defaults to None for PGVector
        if v is None:
            return None

        if not isinstance(v, dict):
            return v

        sanitized = {k: val for k, val in v.items() if val != {}}

        if len(sanitized) == 0:
            return None
        return sanitized


class ImageReference(BaseModel):
    """Extracted image reference from document chunks."""
    
    image_id: str = Field(
        ...,
        description="Unique image identifier",
        examples=["IMG_abc123def456"],
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="Parent document ID",
    )
    description: Optional[str] = Field(
        default=None,
        description="AI-generated image description",
    )
    caption: Optional[str] = Field(
        default=None,
        description="Original caption if available",
    )
    source_uri: Optional[str] = Field(
        default=None,
        description="Original image URL",
    )
    filepath: Optional[str] = Field(
        default=None,
        description="Local file path",
    )
    source_chunk_index: int = Field(
        ...,
        ge=0,
        description="Index of the source chunk in the results",
    )


class DocumentChunk(BaseModel):
    """Retrieved document chunk with full traceability."""
    
    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier",
        examples=["CHK_abc123_0"],
    )
    doc_id: str = Field(
        ...,
        description="Parent document identifier",
    )
    content: str = Field(
        ...,
        description="The text content of the document chunk",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata including project_name, category, source",
    )
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Similarity score (0-1, higher is more similar)",
    )
    has_images: bool = Field(
        default=False,
        description="Whether this chunk contains image references",
    )
    source_uri: Optional[str] = Field(
        default=None,
        description="Original source URL for citation",
    )
    section: Optional[str] = Field(
        default=None,
        description="Section/header name",
    )


class RetrievalResponse(BaseModel):
    """Search response with documents and images."""
    
    query: str = Field(
        ...,
        description="The original search query",
    )
    total_results: int = Field(
        ...,
        ge=0,
        description="Number of chunks returned",
    )
    chunks: List[DocumentChunk] = Field(
        default_factory=list,
        description="Retrieved document chunks",
    )
    images: List[ImageReference] = Field(
        default_factory=list,
        description="Extracted image references from chunks",
    )
    has_more: bool = Field(
        default=False,
        description="Indicates if more results might be available",
    )
    retrieval_mode: str = Field(
        default="hybrid",
        description="Retrieval mode used: 'hybrid', 'vector', 'bm25'",
    )
    max_score: Optional[float] = Field(
        default=None,
        description="Maximum relevance score among results",
    )


class HealthResponse(BaseModel):
    #Health check response.
    
    status: str = Field(
        default="healthy",
        description="Service health status",
    )
    version: str = Field(
        ...,
        description="API version",
    )
    vector_store_connected: bool = Field(
        default=False,
        description="Whether vector store is connected",
    )


class ErrorResponse(BaseModel):
    #Error response.
    
    error: str = Field(
        ...,
        description="Error type",
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details",
    )
