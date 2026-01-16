# Base Exception
class ScraperError(Exception):
    # Base exception for scraper errors
    pass


# Specific Exceptions
class RateLimitError(ScraperError):
    # Raised when GitHub API rate limit is exceeded
    
    def __init__(self, retry_after: float, message: str = "Rate limit exceeded"):
        self.retry_after = retry_after
        super().__init__(f"{message}. Retry after {retry_after:.0f}s")


class RepositoryNotFoundError(ScraperError):
    # Raised when the specified repository does not exist or is inaccessible
    pass


class AuthenticationError(ScraperError):
    # Raised when GitHub authentication fails (invalid token)
    pass


class ContentFetchError(ScraperError):
    # Raised when content cannot be retrieved from the repository
    pass
