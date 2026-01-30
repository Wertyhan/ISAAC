"""Tests for ISAAC Generation Module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from isaac_generation.interfaces import (
    ResolvedImage,
    FormattedContext,
    GenerationInput,
)
from isaac_generation.formatters import ContextFormatter, EnhancedContextFormatter


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.top_k_chunks = 5
    config.images_dir = Path("data/images")
    config.chat_history_limit = 6
    return config


@pytest.fixture
def sample_retrieval_result():
    """Sample retrieval result for testing formatters."""
    return {
        "chunks": [
            {
                "content": "Twitter uses a fan-out architecture for timeline delivery.",
                "metadata": {
                    "project_name": "twitter",
                    "h1": "Design the Twitter timeline",
                    "h2": "Core Components",
                },
                "score": 0.95,
            },
            {
                "content": "The Memory Cache stores user timelines for fast access.",
                "metadata": {
                    "project_name": "twitter",
                    "h1": "Design the Twitter timeline",
                    "h2": "Caching Strategy",
                },
                "score": 0.89,
            },
        ],
        "images": [
            {
                "image_id": "IMG_abc123",
                "description": "Twitter architecture diagram",
                "source_chunk_index": 0,
            }
        ],
    }


@pytest.fixture
def sample_retrieval_no_images():
    """Retrieval result without explicit images."""
    return {
        "chunks": [
            {
                "content": "Scaling on AWS involves multiple services.",
                "metadata": {
                    "project_name": "scaling_aws",
                    "h1": "Design a system that scales",
                },
                "score": 0.92,
            }
        ],
        "images": [],
    }


class TestContextFormatter:
    """Tests for base ContextFormatter."""
    
    def test_format_chunks_extracts_text(self, mock_config, sample_retrieval_result):
        formatter = ContextFormatter(config=mock_config)
        result = formatter.format(sample_retrieval_result)
        
        assert "Twitter uses a fan-out architecture" in result.text
        assert "Memory Cache stores user timelines" in result.text
    
    def test_format_extracts_sources(self, mock_config, sample_retrieval_result):
        formatter = ContextFormatter(config=mock_config)
        result = formatter.format(sample_retrieval_result)
        
        assert len(result.sources) >= 1
        assert "Design the Twitter timeline" in result.sources
    
    def test_format_empty_chunks(self, mock_config):
        formatter = ContextFormatter(config=mock_config)
        result = formatter.format({"chunks": [], "images": []})
        
        assert "No relevant context found" in result.text
        assert result.sources == []
        assert result.images == []
    
    def test_build_header_with_section(self, mock_config):
        formatter = ContextFormatter(config=mock_config)
        
        metadata = {"h1": "Twitter Design", "h2": "Architecture"}
        header = formatter._build_header(metadata, 1)
        
        assert "[Twitter Design]" in header
        assert "Architecture" in header
    
    def test_build_header_without_section(self, mock_config):
        formatter = ContextFormatter(config=mock_config)
        
        metadata = {"project_name": "twitter"}
        header = formatter._build_header(metadata, 1)
        
        assert "[twitter]" in header


class TestEnhancedContextFormatter:
    """Tests for EnhancedContextFormatter with image discovery."""
    
    def test_injects_image_context(self, mock_config, sample_retrieval_result):
        with patch.object(EnhancedContextFormatter, '_find_image_file', return_value=None):
            with patch.object(EnhancedContextFormatter, 'find_images_by_project', return_value=[]):
                formatter = EnhancedContextFormatter(config=mock_config)
                result = formatter.format(sample_retrieval_result)
        
        assert "Twitter uses a fan-out" in result.text
    
    def test_discovers_images_by_project(self, mock_config, sample_retrieval_no_images, tmp_path):
        test_image = tmp_path / "scaling_aws_abc123.png"
        test_image.touch()
        
        mock_config.images_dir = tmp_path
        
        formatter = EnhancedContextFormatter(config=mock_config)
        result = formatter.format(sample_retrieval_no_images)
        
        assert isinstance(result.images, list)


class TestResolvedImage:

    def test_exists_property_true(self, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.touch()
        
        image = ResolvedImage(
            image_id="IMG_test",
            path=test_file,
            description="Test image",
            source_chunk_index=0,
        )
        
        assert image.exists is True
    
    def test_exists_property_false(self, tmp_path):
        image = ResolvedImage(
            image_id="IMG_test",
            path=tmp_path / "nonexistent.png",
            description="Test image",
            source_chunk_index=0,
        )
        
        assert image.exists is False


class TestFormattedContext:

    def test_has_images_true(self, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.touch()
        
        context = FormattedContext(
            text="Test context",
            sources=["Source 1"],
            images=[ResolvedImage(
                image_id="IMG_1",
                path=test_file,
                description="Test",
                source_chunk_index=0,
            )],
        )
        
        assert context.has_images is True
    
    def test_has_images_false(self):
        context = FormattedContext(
            text="Test context",
            sources=["Source 1"],
            images=[],
        )
        
        assert context.has_images is False


class TestGenerationInput:

    def test_creation_minimal(self):
        context = FormattedContext(
            text="Context",
            sources=[],
            images=[],
        )
        
        gen_input = GenerationInput(
            query="Test query",
            context=context,
        )
        
        assert gen_input.query == "Test query"
        assert gen_input.chat_history == []
        assert gen_input.user_image_path is None
    
    def test_creation_full(self, tmp_path):
        context = FormattedContext(
            text="Context",
            sources=["Source"],
            images=[],
        )
        
        image_path = tmp_path / "user_image.png"
        
        gen_input = GenerationInput(
            query="Test query",
            context=context,
            chat_history=[],
            user_image_path=image_path,
            user_image_description="User's architecture diagram",
        )
        
        assert gen_input.user_image_path == image_path
        assert gen_input.user_image_description == "User's architecture diagram"
