"""Generation Configuration."""

from functools import lru_cache
from typing import Optional

from isaac_core.config import BaseConfig


class GenerationConfig(BaseConfig):
    """Generation service configuration - inherits from BaseConfig."""
    pass


_config: Optional[GenerationConfig] = None


@lru_cache
def get_config() -> GenerationConfig:
    global _config
    if _config is None:
        _config = GenerationConfig()
    return _config
