# Base Exception
class IngestionError(Exception):
    # Base exception for ingestion pipeline errors
    pass


# API Exceptions
class GeminiAPIError(IngestionError):
    # Raised when Gemini API call fails
    
    def __init__(self, message: str = "Gemini API error", cause: str = ""):
        self.cause = cause
        super().__init__(f"{message}: {cause}" if cause else message)


class GeminiRateLimitError(GeminiAPIError):
    # Raised when Gemini API rate limit is exceeded
    
    def __init__(self, retry_after: float = 60.0):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.0f}s")


class GeminiQuotaExceededError(GeminiAPIError):
    # Raised when Gemini API quota is exhausted
    pass


# Network Exceptions
class ImageDownloadError(IngestionError):
    # Raised when image download fails
    
    def __init__(self, url: str, reason: str = "Download failed"):
        self.url = url
        self.reason = reason
        super().__init__(f"{reason}: {url}")


class NetworkTimeoutError(IngestionError):
    # Raised when network request times out
    pass


# Database Exceptions
class DatabaseConnectionError(IngestionError):
    # Raised when database connection fails
    pass


class VectorStoreError(IngestionError):
    # Raised when vector store operation fails
    pass


# Processing Exceptions
class ContentProcessingError(IngestionError):
    # Raised when content processing fails
    pass


class InvalidInputError(IngestionError):
    # Raised when input data is invalid or malformed
    pass
