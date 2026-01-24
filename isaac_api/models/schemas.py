from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    #Search query request.
    
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
    #Extracted image reference from document chunks.
    
    image_id: str = Field(
        ...,
        description="Unique image identifier",
        examples=["IMG_abc123def456"],
    )
    description: Optional[str] = Field(
        default=None,
        description="AI-generated image description",
    )
    source_chunk_index: int = Field(
        ...,
        ge=0,
        description="Index of the source chunk in the results",
    )


class DocumentChunk(BaseModel):
    #Retrieved document chunk with metadata.
    
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


class RetrievalResponse(BaseModel):
    #Search response with documents and images.
    
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
