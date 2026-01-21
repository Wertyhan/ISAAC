from typing import Protocol, List, Optional, runtime_checkable


# Interfaces
@runtime_checkable
class ContentFileInterface(Protocol):
    # Protocol for content file objects
    
    @property
    def name(self) -> str:
        # Name of the file
        ...
    
    @property
    def path(self) -> str:
        # Full path to the file in the repository
        ...
    
    @property
    def type(self) -> str:
        # Type of the content ('file' or 'dir')
        ...
    
    @property
    def content(self) -> Optional[str]:
        # Base64 encoded content of the file
        ...
    
    @property
    def size(self) -> Optional[int]:
        # Size of the file in bytes
        ...


@runtime_checkable  
class RepositoryInterface(Protocol):
    # Protocol for repository access abstraction
    
    @property
    def full_name(self) -> str:
        # Full name of the repository (owner/repo)
        ...
    
    def get_contents(self, path: str, ref: str) -> List[ContentFileInterface]:
        # Retrieve contents of a file or directory
        ...
    
    def close(self) -> None:
        # Close any open connections
        ...
