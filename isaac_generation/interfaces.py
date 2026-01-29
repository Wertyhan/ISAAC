"""Generation Interfaces - Protocol definitions following Interface Segregation Principle."""

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, List, Optional, Dict, Any, AsyncIterator

from langchain_core.messages import BaseMessage


@dataclass
class ResolvedImage:
    """Resolved image with path and metadata."""
    
    image_id: str
    path: Path
    description: str
    source_chunk_index: int
    
    @property
    def exists(self) -> bool:
        return self.path.exists()


@dataclass
class FormattedContext:
    """Processed context ready for generation."""
    
    text: str
    sources: List[str]
    images: List[ResolvedImage]
    is_relevant: bool = True  # Whether context is relevant to the query
    max_score: float = 0.0  # Highest relevance score among chunks
    
    @property
    def has_images(self) -> bool:
        return len(self.images) > 0


@dataclass
class GenerationInput:
    """Input for generation pipeline."""
    
    query: str
    context: FormattedContext
    chat_history: List[BaseMessage] = field(default_factory=list)
    user_image_path: Optional[Path] = None
    user_image_description: Optional[str] = None


@dataclass
class GenerationResult:
    """Result from generation pipeline."""
    
    response_text: str
    sources: List[str]
    images: List[ResolvedImage]
    user_image_analysis: Optional[str] = None


class IContextFormatter(Protocol):
    """Formats retrieval results into context for generation."""
    
    @abstractmethod
    def format(self, retrieval_result: Dict[str, Any]) -> FormattedContext:
        """Format retrieval results into structured context."""
        ...


class IImageResolver(Protocol):
    """Resolves image references to actual file paths."""
    
    @abstractmethod
    def resolve(self, image_refs: List[Dict[str, Any]]) -> List[ResolvedImage]:
        """Resolve image references to file paths."""
        ...
    
    @abstractmethod
    def find_similar_images(self, query: str) -> List[ResolvedImage]:
        """Find images related to a query."""
        ...


class IImageAnalyzer(Protocol):
    """Analyzes images using vision models."""
    
    @abstractmethod
    def analyze(self, image_path: Path, prompt: Optional[str] = None) -> str:
        """Analyze an image and return description."""
        ...
    
    @abstractmethod
    async def analyze_async(self, image_path: Path, prompt: Optional[str] = None) -> str:
        """Async version of image analysis."""
        ...


class IResponseGenerator(Protocol):
    """Generates responses using LLM."""
    
    @abstractmethod
    async def generate_stream(self, gen_input: GenerationInput) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...
    
    @abstractmethod
    async def generate(self, gen_input: GenerationInput) -> str:
        """Generate complete response."""
        ...


class IRetrieverAdapter(Protocol):
    """Adapter for retrieval service."""
    
    @abstractmethod
    def search(self, query: str, k: int) -> Dict[str, Any]:
        """Search for relevant documents."""
        ...
    
    @abstractmethod
    def search_by_image(self, image_description: str, k: int) -> Dict[str, Any]:
        """Search using image description."""
        ...
        ...
