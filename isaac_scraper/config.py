from typing import Optional, FrozenSet
from pydantic import BaseModel, Field, field_validator, SecretStr
from pydantic_settings import BaseSettings


# Configuration
class Config(BaseSettings):
    """Scraper settings."""
    
    github_token: SecretStr
    repo_name: str = "donnemartin/system-design-primer"
    start_path: str = "solutions/system_design"
    output_file: str = "data/raw/isaac_raw_data.json"
    branch: str = "master"
    max_depth: int = Field(default=10, ge=1, le=50)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=60.0, ge=1.0)
    image_folders: FrozenSet[str] = Field(
        default_factory=lambda: frozenset(["img", "images", "assets", "diagrams", "figures"])
    )
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    @field_validator("repo_name")
    @classmethod
    def validate_repo_format(cls, v: str) -> str:
        if v.count("/") != 1:
            raise ValueError("Format: owner/repo")
        return v


# Models
class CrawlResult(BaseModel):
    category: str
    project_name: str
    title: Optional[str] = None
    description: Optional[str] = None
    readme_content: str
    diagram_url: Optional[str] = None
    source_path: str
