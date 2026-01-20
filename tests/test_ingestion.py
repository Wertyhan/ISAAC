import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from isaac_ingestion.config import ImageMeta
from isaac_ingestion.exceptions import InvalidInputError
from isaac_ingestion.services.image_manager import ImageManager
from isaac_ingestion.clients.gemini_client import GeminiVisionClient, FALLBACK_DESCRIPTION
from isaac_ingestion.pipeline import IngestionPipeline


class TestImageManager:

    def test_is_valid_url(self, mock_ingestion_config):
        im = ImageManager(mock_ingestion_config)
        assert im._is_valid_url("https://example.com/img.png") is True
        assert im._is_valid_url("http://example.com/img.png") is True
        assert im._is_valid_url("ftp://example.com/img.png") is False
        assert im._is_valid_url("not-a-url") is False

    def test_compute_hash(self, mock_ingestion_config):
        im = ImageManager(mock_ingestion_config)
        h1 = im._compute_hash(b"test content")
        h2 = im._compute_hash(b"test content")
        h3 = im._compute_hash(b"different")
        assert h1 == h2
        assert h1 != h3

    def test_get_extension_from_url(self, mock_ingestion_config):
        im = ImageManager(mock_ingestion_config)
        assert im._get_extension("https://x.com/img.png", None) == ".png"
        assert im._get_extension("https://x.com/img.jpg", None) == ".jpg"
        assert im._get_extension("https://x.com/img.jpeg", None) == ".jpeg"

    def test_get_extension_from_content_type(self, mock_ingestion_config):
        im = ImageManager(mock_ingestion_config)
        assert im._get_extension("https://x.com/file", "image/png") == ".png"
        assert im._get_extension("https://x.com/file", "image/jpeg") == ".jpg"

    def test_build_local_path(self, mock_ingestion_config):
        im = ImageManager(mock_ingestion_config)
        path = im._build_local_path("My Project!", "abc123def456", ".png")
        assert "My_Project_" in path.name
        assert "abc123def456"[:12] in path.name
        assert path.suffix == ".png"

    def test_download_invalid_url_returns_none(self, mock_ingestion_config, caplog):
        im = ImageManager(mock_ingestion_config)
        with caplog.at_level('WARNING'):
            result = im.download("not-valid", "project")
            assert result is None
            assert "Invalid image URL" in caplog.text


class TestGeminiVisionClient:

    @patch("isaac_ingestion.clients.gemini_client.genai")
    def test_generate_description_missing_file(self, mock_genai, mock_ingestion_config):
        client = GeminiVisionClient(mock_ingestion_config)
        result = client.generate_description(Path("/nonexistent/file.png"))
        assert result == FALLBACK_DESCRIPTION

    @patch("isaac_ingestion.clients.gemini_client.genai")
    def test_close_clears_model(self, mock_genai, mock_ingestion_config):
        client = GeminiVisionClient(mock_ingestion_config)
        assert client._client is not None
        client.close()
        assert client._client is None


class TestImageMeta:

    def test_image_id_generation(self):
        meta = ImageMeta(
            original_url="https://x.com/img.png",
            local_path=Path("data/images/test.png"),
            file_hash="abcdef123456789",
            project_name="test-project"
        )
        assert meta.image_id == "IMG_abcdef123456"

    def test_repr(self):
        meta = ImageMeta(
            original_url="https://x.com/img.png",
            local_path=Path("data/images/test.png"),
            file_hash="abcdef123456789",
            project_name="test"
        )
        assert "IMG_abcdef123456" in repr(meta)


class TestIngestionPipeline:

    @patch("isaac_ingestion.pipeline.GoogleGenerativeAIEmbeddings")
    @patch("isaac_ingestion.pipeline.PGVector")
    def test_pipeline_context_manager(self, mock_pg, mock_emb, mock_ingestion_config):
        mock_gemini = Mock()
        mock_images = Mock()
        mock_text = Mock()
        
        with IngestionPipeline(mock_ingestion_config, mock_gemini, mock_images, mock_text) as pipeline:
            assert pipeline is not None
        
        mock_gemini.close.assert_called_once()
        mock_images.close.assert_called_once()

    def test_get_stats_returns_dict(self, mock_ingestion_config):
        mock_gemini = Mock()
        mock_images = Mock()
        mock_text = Mock()
        
        pipeline = IngestionPipeline(mock_ingestion_config, mock_gemini, mock_images, mock_text)
        stats = pipeline.get_stats()
        
        assert isinstance(stats, dict)
        assert "projects_processed" in stats
        assert "chunks_created" in stats

    @patch("builtins.open")
    @patch("json.load")
    def test_load_raw_data_invalid_json(self, mock_json, mock_open, mock_ingestion_config):
        import json
        mock_json.side_effect = json.JSONDecodeError("test", "doc", 0)
        
        pipeline = IngestionPipeline(
            mock_ingestion_config, Mock(), Mock(), Mock()
        )
        
        with pytest.raises(InvalidInputError):
            pipeline._load_raw_data()

    @patch("builtins.open", side_effect=FileNotFoundError())
    def test_load_raw_data_file_not_found(self, mock_open, mock_ingestion_config):
        pipeline = IngestionPipeline(
            mock_ingestion_config, Mock(), Mock(), Mock()
        )
        
        with pytest.raises(InvalidInputError):
            pipeline._load_raw_data()
