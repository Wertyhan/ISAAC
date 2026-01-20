# Test Fixtures
import pytest
import warnings
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from pathlib import Path

# Suppress Deprecation/Future warnings from external libs (LangChain/Google)
warnings.filterwarnings("ignore", message=".*support for the `google.generativeai` package has ended.*")



@pytest.fixture
def mock_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_12345")


@pytest.fixture
def sample_markdown():
    return """# System Design Title

This is the first paragraph with a description.

## Architecture

Here's a [relative link](./docs/arch.md) and [absolute](https://example.com).

```python
link = "[not a link](should/stay)"
```

![Diagram](./images/diagram.png)
"""


@pytest.fixture
def mock_content_file():
    def _make(name: str, size: int = 1000, file_type: str = "file"):
        m = MagicMock()
        m.name = name
        m.path = name
        m.type = file_type
        m.size = size
        return m
    return _make


# Ingestion Fixtures 

@dataclass
class IngestionTestConfig:
    postgres_connection_string: str = "postgresql://test"
    collection_name: str = "test_vectors"
    embedding_model: str = "models/text-embedding-004"
    vision_model: str = "gemini-2.0-flash"
    vision_max_retries: int = 1
    vision_retry_delay: float = 0.1
    chunk_size: int = 500
    chunk_overlap: int = 50
    images_dir: Path = field(default_factory=lambda: Path("data/images"))
    raw_data_file: Path = field(default_factory=lambda: Path("data/raw/test.json"))
    request_timeout: float = 10.0
    download_max_retries: int = 1
    gemini_api_key: MagicMock = field(default_factory=lambda: MagicMock(get_secret_value=lambda: "test-key"))

@pytest.fixture
def mock_ingestion_config():
    return IngestionTestConfig()

