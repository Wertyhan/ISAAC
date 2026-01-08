# Test Fixtures
import pytest
from unittest.mock import MagicMock


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
