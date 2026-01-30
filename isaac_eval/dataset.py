"""Evaluation Dataset - Query set with expected documents/images."""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Type of evaluation query."""
    TEXT_TO_ARCHITECTURE = "text_to_architecture"
    IMAGE_ANALYSIS = "image_analysis"
    SIMILARITY_SEARCH = "similarity_search"
    OFF_TOPIC = "off_topic"


class DifficultyLevel(str, Enum):
    """Query difficulty level."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class EvalQuery:
    """Single evaluation query with expected results."""
    
    query_id: str
    query: str
    query_type: QueryType
    difficulty: DifficultyLevel
    
    # Expected results (ground truth)
    expected_topics: List[str] = field(default_factory=list)  # Topics that should appear
    expected_doc_ids: List[str] = field(default_factory=list)  # Specific doc_ids expected
    expected_images: List[str] = field(default_factory=list)  # Image patterns expected
    expected_keywords: List[str] = field(default_factory=list)  # Keywords in response
    
    # Negative expectations
    should_refuse: bool = False  # Should system refuse (off-topic)
    should_say_idk: bool = False  # Should say "I don't know"
    forbidden_topics: List[str] = field(default_factory=list)  # Topics NOT to mention
    
    # Metadata
    category: str = ""
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["query_type"] = self.query_type.value
        data["difficulty"] = self.difficulty.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalQuery":
        """Create from dictionary."""
        data["query_type"] = QueryType(data["query_type"])
        data["difficulty"] = DifficultyLevel(data["difficulty"])
        return cls(**data)


@dataclass
class EvaluationDataset:
    """Collection of evaluation queries."""
    
    queries: List[EvalQuery] = field(default_factory=list)
    version: str = "1.0"
    description: str = "ISAAC Evaluation Dataset"
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __iter__(self):
        return iter(self.queries)
    
    def add_query(self, query: EvalQuery) -> None:
        """Add a query to the dataset."""
        self.queries.append(query)
    
    def filter_by_type(self, query_type: QueryType) -> List[EvalQuery]:
        """Filter queries by type."""
        return [q for q in self.queries if q.query_type == query_type]
    
    def filter_by_difficulty(self, difficulty: DifficultyLevel) -> List[EvalQuery]:
        """Filter queries by difficulty."""
        return [q for q in self.queries if q.difficulty == difficulty]
    
    def get_by_id(self, query_id: str) -> Optional[EvalQuery]:
        """Get query by ID."""
        for q in self.queries:
            if q.query_id == query_id:
                return q
        return None
    
    def save(self, path: Path) -> None:
        """Save dataset to JSON file."""
        data = {
            "version": self.version,
            "description": self.description,
            "queries": [q.to_dict() for q in self.queries],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.queries)} queries to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        dataset = cls(
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
        )
        for q_data in data.get("queries", []):
            dataset.add_query(EvalQuery.from_dict(q_data))
        
        logger.info(f"Loaded {len(dataset)} queries from {path}")
        return dataset
    
    def stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        type_counts = {}
        difficulty_counts = {}
        
        for q in self.queries:
            type_counts[q.query_type.value] = type_counts.get(q.query_type.value, 0) + 1
            difficulty_counts[q.difficulty.value] = difficulty_counts.get(q.difficulty.value, 0) + 1
        
        return {
            "total_queries": len(self.queries),
            "by_type": type_counts,
            "by_difficulty": difficulty_counts,
            "with_expected_docs": sum(1 for q in self.queries if q.expected_doc_ids),
            "with_expected_images": sum(1 for q in self.queries if q.expected_images),
            "off_topic_queries": sum(1 for q in self.queries if q.should_refuse),
        }


# Category constants
CAT_DISTRIBUTED_SYSTEMS = "Distributed Systems"
CAT_DATABASES = "Databases"
CAT_SYSTEM_DESIGN = "System Design"
CAT_SCALING = "Scaling"

# Topic constants
TOPIC_CONSISTENT_HASHING = "consistent hashing"


def create_default_dataset() -> EvaluationDataset:
    """Create the default evaluation dataset with 30 queries."""
    dataset = EvaluationDataset(
        version="1.0",
        description="ISAAC System Design Evaluation Dataset - Based on PRD examples and system-design-primer content"
    )
    
    # Easy queries
    dataset.add_query(EvalQuery(
        query_id="T001",
        query="What is a CDN and how does it improve website performance?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.EASY,
        expected_topics=["CDN", "content delivery network", "caching", "latency"],
        expected_keywords=["edge", "cache", "latency", "geographic"],
        category="CDN",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T002",
        query="Explain the difference between SQL and NoSQL databases",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.EASY,
        expected_topics=["SQL", "NoSQL", "relational", "document"],
        expected_keywords=["ACID", "schema", "scalability", "consistency"],
        category="Databases",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T003",
        query="What is a load balancer and why do we need it?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.EASY,
        expected_topics=["load balancer", "traffic distribution", "availability"],
        expected_keywords=["round robin", "health check", "horizontal scaling"],
        category="Load Balancing",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T004",
        query="What is caching and what are common caching strategies?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.EASY,
        expected_topics=["cache", "caching strategies", "Redis", "Memcached"],
        expected_keywords=["cache-aside", "write-through", "TTL", "eviction"],
        category="Caching",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T005",
        query="What is database replication?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.EASY,
        expected_topics=["replication", "master-slave", "availability"],
        expected_keywords=["replica", "sync", "failover", "read scaling"],
        category="Databases",
    ))
    
    # Medium queries
    dataset.add_query(EvalQuery(
        query_id="T006",
        query="I want to create a food delivery app like Glovo that processes many orders in real time. What architecture should I use?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["microservices", "message queue", "real-time", "scalability"],
        expected_keywords=["Kafka", "Redis", "notification", "order", "async"],
        category="Real-time Systems",
        notes="PRD Example Query #1",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T007",
        query="How do I design a URL shortener like bit.ly?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["URL shortener", "hashing", "database", "redirect"],
        expected_keywords=["base62", "hash", "unique ID", "redirect", "analytics"],
        expected_images=["url_shortener", "pastebin"],
        category=CAT_SYSTEM_DESIGN,
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T008",
        query="Design a web crawler that can index millions of pages",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["web crawler", "distributed", "queue", "deduplication"],
        expected_keywords=["URL frontier", "robots.txt", "politeness", "BFS"],
        expected_images=["web_crawler"],
        category=CAT_SYSTEM_DESIGN,
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T009",
        query="How would you design Twitter's timeline feature?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["timeline", "fan-out", "caching", "social network"],
        expected_keywords=["fan-out", "push", "pull", "cache", "followers"],
        expected_images=["twitter"],
        category="Social Media",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T010",
        query="What is the CAP theorem and how does it affect system design?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["CAP theorem", "consistency", "availability", "partition tolerance"],
        expected_keywords=["CP", "AP", "trade-off", "distributed"],
        category=CAT_DISTRIBUTED_SYSTEMS,
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T011",
        query="How to implement rate limiting in a distributed system?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["rate limiting", "API", "throttling"],
        expected_keywords=["token bucket", "sliding window", "Redis", "distributed"],
        category="API Design",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T012",
        query="Design a notification system that can handle millions of users",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["notification", "push", "message queue", "scalability"],
        expected_keywords=["push notification", "email", "SMS", "queue", "async"],
        category="Real-time Systems",
    ))
    
    # Hard queries
    dataset.add_query(EvalQuery(
        query_id="T013",
        query="Design a distributed key-value store like DynamoDB with high availability",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.HARD,
        expected_topics=["key-value store", TOPIC_CONSISTENT_HASHING, "replication", "quorum"],
        expected_keywords=[TOPIC_CONSISTENT_HASHING, "vector clock", "gossip", "eventual consistency"],
        category=CAT_DISTRIBUTED_SYSTEMS,
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T014",
        query="How would you design a system like Google Maps that handles real-time traffic updates?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.HARD,
        expected_topics=["maps", "geospatial", "real-time", "routing"],
        expected_keywords=["quadtree", "geohash", "Dijkstra", "real-time", "tiles"],
        category="Geospatial",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="T015",
        query="Design a stock exchange system that processes millions of trades per second",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.HARD,
        expected_topics=["trading", "matching engine", "low latency", "order book"],
        expected_keywords=["order matching", "FIFO", "latency", "throughput"],
        category="Financial Systems",
    ))
    
    # Image queries
    dataset.add_query(EvalQuery(
        query_id="I001",
        query="Show me architecture diagrams for scaling a web application",
        query_type=QueryType.SIMILARITY_SEARCH,
        difficulty=DifficultyLevel.EASY,
        expected_topics=["scaling", "web application", "architecture"],
        expected_images=["scaling", "web"],
        category="Scaling",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="I002",
        query="Show me the microservices diagram for a courier tracking system",
        query_type=QueryType.SIMILARITY_SEARCH,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["microservices", "tracking", "location"],
        expected_images=["microservice", "tracking"],
        category="Microservices",
        notes="PRD Example Query #2",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="I003",
        query="Find diagrams showing database sharding patterns",
        query_type=QueryType.SIMILARITY_SEARCH,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["sharding", "database", "partitioning"],
        expected_images=["shard", "partition"],
        category=CAT_DATABASES,
    ))
    
    # Grounded knowledge queries
    dataset.add_query(EvalQuery(
        query_id="G001",
        query="What are the trade-offs of using microservices vs monolithic architecture?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["microservices", "monolith", "trade-offs"],
        expected_keywords=["complexity", "deployment", "scalability", "communication"],
        category="Architecture Patterns",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="G002",
        query=f"How does {TOPIC_CONSISTENT_HASHING} work in distributed systems?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=[TOPIC_CONSISTENT_HASHING, "distributed", "load balancing"],
        expected_keywords=["ring", "virtual nodes", "rebalancing", "hash"],
        category=CAT_DISTRIBUTED_SYSTEMS,
    ))
    
    dataset.add_query(EvalQuery(
        query_id="G003",
        query="Explain the differences between horizontal and vertical scaling",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.EASY,
        expected_topics=["scaling", "horizontal", "vertical"],
        expected_keywords=["scale out", "scale up", "servers", "resources"],
        category=CAT_SCALING,
    ))
    
    # Off-topic queries (should refuse)
    dataset.add_query(EvalQuery(
        query_id="O001",
        query="What is the best recipe for chocolate cake?",
        query_type=QueryType.OFF_TOPIC,
        difficulty=DifficultyLevel.EASY,
        should_refuse=True,
        forbidden_topics=["system", "architecture", "database"],
        category="Off-topic",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="O002",
        query="Tell me about the history of the Roman Empire",
        query_type=QueryType.OFF_TOPIC,
        difficulty=DifficultyLevel.EASY,
        should_refuse=True,
        forbidden_topics=["system", "architecture", "software"],
        category="Off-topic",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="O003",
        query="What are the specs of the new iPhone?",
        query_type=QueryType.OFF_TOPIC,
        difficulty=DifficultyLevel.EASY,
        should_refuse=True,
        forbidden_topics=["system design", "architecture"],
        category="Off-topic",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="O004",
        query="Who won the FIFA World Cup in 2022?",
        query_type=QueryType.OFF_TOPIC,
        difficulty=DifficultyLevel.EASY,
        should_refuse=True,
        forbidden_topics=["architecture", "system"],
        category="Off-topic",
    ))
    
    # Edge case queries
    dataset.add_query(EvalQuery(
        query_id="E001",
        query="How does Spotify's machine learning recommendation engine work internally?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.HARD,
        should_say_idk=True,  # Likely not in knowledge base
        expected_topics=["recommendation", "ML"],
        category="Edge Case",
        notes="Specific internal details likely not in corpus",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="E002",
        query="What is the exact architecture of TikTok's video processing pipeline?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.HARD,
        should_say_idk=True,
        expected_topics=["video", "processing"],
        category="Edge Case",
        notes="Proprietary system details not available",
    ))
    
    # Multimodal queries
    dataset.add_query(EvalQuery(
        query_id="M001",
        query="How does a polyglot database strategy work for high-volume traffic?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["polyglot", "database", "traffic"],
        expected_keywords=["multiple databases", "use case", "SQL", "NoSQL"],
        category="Databases",
        notes="PRD Example Query #3",
    ))
    
    return dataset
