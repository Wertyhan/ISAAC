# Tests
import json
import tempfile
import os
import pytest
from unittest.mock import patch, MagicMock

from isaac_scraper.config import Config, CrawlResult
from isaac_scraper.processors import MarkdownProcessor, ImageSelector
from isaac_scraper.scraper import GitScraper, sanitize_log


# Config Tests
class TestConfig:
    
    def test_loads_from_env(self, mock_token):
        config = Config()
        assert config.github_token == "ghp_test_12345"
    
    def test_validates_repo_format(self, mock_token):
        with pytest.raises(ValueError):
            Config(github_token="x", repo_name="invalid")
    
    def test_accepts_overrides(self, mock_token):
        config = Config(github_token="x", repo_name="a/b", max_depth=20)
        assert config.max_depth == 20


# Markdown Tests
class TestMarkdown:
    
    def test_extracts_title(self, sample_markdown):
        assert MarkdownProcessor().extract_title(sample_markdown) == "System Design Title"
    
    def test_extracts_description(self, sample_markdown):
        desc = MarkdownProcessor().extract_description(sample_markdown)
        assert "first paragraph" in desc
    
    def test_transforms_relative_links(self, sample_markdown):
        result = MarkdownProcessor().fix_links(sample_markdown, "https://raw.github.com/a/b")
        assert "https://raw.github.com/a/b/docs/arch.md" in result
        assert "https://example.com" in result
    
    def test_preserves_code_blocks(self, sample_markdown):
        result = MarkdownProcessor().fix_links(sample_markdown, "https://x.com")
        assert '[not a link](should/stay)' in result


# Image Tests
class TestImages:
    
    def test_detects_image_formats(self):
        assert ImageSelector.is_image("x.png")
        assert ImageSelector.is_image("X.SVG")
        assert not ImageSelector.is_image("x.md")
    
    def test_prefers_architecture_keyword(self, mock_content_file):
        images = [
            mock_content_file("random.png", 5000),
            mock_content_file("architecture.png", 1000),
        ]
        assert ImageSelector.select_best(images).name == "architecture.png"
    
    def test_ignores_icons(self, mock_content_file):
        images = [
            mock_content_file("icon.png"),
            mock_content_file("diagram.png"),
        ]
        assert ImageSelector.select_best(images).name == "diagram.png"


# Utility Tests
class TestUtils:
    
    def test_sanitize_masks_tokens(self):
        assert "ghp_secret" not in sanitize_log("token ghp_secret123")
        assert "***" in sanitize_log("ghp_abc123")
    
    def test_sanitize_truncates(self):
        assert len(sanitize_log("x" * 1000)) == 500


# CrawlResult Tests
class TestCrawlResult:
    
    def test_serializes_to_json(self):
        result = CrawlResult(
            category="design",
            project_name="test",
            readme_content="# Test",
            source_path="a/b",
        )
        data = json.loads(result.model_dump_json())
        assert data["project_name"] == "test"


# Integration Tests (mocked)
class TestScraper:
    
    @patch("isaac_scraper.scraper.Github")
    def test_connects_to_github(self, MockGithub, mock_token):
        mock_repo = MagicMock()
        mock_repo.full_name = "test/repo"
        MockGithub.return_value.get_repo.return_value = mock_repo
        
        config = Config(github_token="x", repo_name="test/repo")
        scraper = GitScraper(config)
        assert scraper._repo.full_name == "test/repo"
    
    @patch("isaac_scraper.scraper.Github")
    def test_invalid_token_raises(self, MockGithub, mock_token):
        from github import BadCredentialsException
        MockGithub.return_value.get_repo.side_effect = BadCredentialsException(401, {}, {})
        
        with pytest.raises(ValueError, match="Invalid GitHub token"):
            GitScraper(Config(github_token="bad", repo_name="a/b"))
    
    @patch("isaac_scraper.scraper.Github")
    def test_repo_not_found_raises(self, MockGithub, mock_token):
        from github import UnknownObjectException
        MockGithub.return_value.get_repo.side_effect = UnknownObjectException(404, {}, {})
        
        with pytest.raises(ValueError, match="Repository not found"):
            GitScraper(Config(github_token="x", repo_name="no/exist"))
    
    @patch("isaac_scraper.scraper.Github")
    def test_crawl_processes_readme(self, MockGithub, mock_token, tmp_path, mock_content_file):
        import base64
        
        mock_repo = MagicMock()
        mock_repo.full_name = "test/repo"
        
        readme = mock_content_file("README.md")
        readme.content = base64.b64encode(b"# Title\nDescription").decode()
        
        def get_contents(path, ref=None):
            if path == "start":
                folder = MagicMock(type="dir", name="project", path="start/project")
                folder.name = "project"
                return [folder]
            return [readme]
        
        mock_repo.get_contents.side_effect = get_contents
        MockGithub.return_value.get_repo.return_value = mock_repo
        
        output = tmp_path / "out.json"
        config = Config(github_token="x", repo_name="test/repo", start_path="start", output_file=str(output))
        scraper = GitScraper(config)
        scraper.crawl()
        
        data = json.loads(output.read_text())
        assert len(data) == 1
        assert data[0]["project_name"] == "project"
    
    @patch("isaac_scraper.scraper.Github")
    def test_respects_max_depth(self, MockGithub, mock_token, tmp_path):
        mock_repo = MagicMock()
        mock_repo.full_name = "test/repo"
        mock_repo.get_contents.return_value = [
            MagicMock(type="dir", name="deep", path="a/b/c/d/e")
        ]
        MockGithub.return_value.get_repo.return_value = mock_repo
        
        output = tmp_path / "out.json"
        config = Config(github_token="x", repo_name="test/repo", output_file=str(output), max_depth=2)
        scraper = GitScraper(config)
        scraper.crawl()
        
        assert mock_repo.get_contents.call_count <= 3  # root + 2 levels
    
    @patch("isaac_scraper.scraper.Github")
    def test_saves_json_atomically(self, MockGithub, mock_token, tmp_path):
        mock_repo = MagicMock()
        mock_repo.full_name = "test/repo"
        mock_repo.get_contents.return_value = []
        MockGithub.return_value.get_repo.return_value = mock_repo
        
        output = tmp_path / "out.json"
        config = Config(github_token="x", repo_name="test/repo", output_file=str(output))
        scraper = GitScraper(config)
        scraper.crawl()
        
        assert output.exists()
        assert json.loads(output.read_text()) == []
