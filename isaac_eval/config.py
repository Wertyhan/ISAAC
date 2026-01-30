"""Evaluation Configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EvalConfig:
    """Configuration for evaluation runs."""
    
    eval_data_path: Path = field(default_factory=lambda: Path("isaac_eval/data/eval_queries.json"))
    results_dir: Path = field(default_factory=lambda: Path("isaac_eval/results"))
    
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    default_k: int = 5
    
    faithfulness_threshold: float = 0.7
    citation_threshold: float = 0.8
    
    num_samples: Optional[int] = None
    random_seed: int = 42
    
    def __post_init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)


def get_eval_config() -> EvalConfig:
    """Get default evaluation config."""
    return EvalConfig()
