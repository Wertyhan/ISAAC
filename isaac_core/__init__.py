"""
ISAAC Core Module
=================
Shared components, configuration, and constants across all ISAAC modules.
"""

from isaac_core.config import (
    BaseConfig,
    DatabaseConfig,
    GeminiConfig,
    PathsConfig,
    get_base_config,
)
from isaac_core.constants import (
    IMAGE_ANALYSIS_FAILURE_MARKER,
    SUPPORTED_IMAGE_EXTENSIONS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)

__all__ = [
    "BaseConfig",
    "DatabaseConfig",
    "GeminiConfig",
    "PathsConfig",
    "get_base_config",
    "IMAGE_ANALYSIS_FAILURE_MARKER",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
]
