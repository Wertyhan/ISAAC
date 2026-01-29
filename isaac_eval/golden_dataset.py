"""Golden Dataset Generator - Creates evaluation dataset with exact doc_id mappings."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from isaac_api.core.database import fetch_all_documents
from isaac_api.services.retriever import get_retriever_service
from isaac_eval.dataset import (
    EvaluationDataset, 
    EvalQuery, 
    QueryType, 
    DifficultyLevel,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Document title constants
DOC_SCALING_AWS = "Design a system that scales to millions of users on AWS"
DOC_KV_CACHE = "Design a key-value cache to save the results of the most recent web server queries"
DOC_PASTEBIN = "Design Pastebin.com (or Bit.ly)"
DOC_WEB_CRAWLER = "Design a web crawler"
DOC_TWITTER = "Design the Twitter timeline and search"
DOC_SOCIAL_NETWORK = "Design the data structures for a social network"
DOC_MINT = "Design Mint.com"
DOC_AMAZON = "Design Amazon's sales rank by category feature"

# Category constants
CAT_SYSTEM_DESIGN = "System Design"

# Mapping of query topics to expected document h1 headers
TOPIC_TO_DOCS: Dict[str, List[str]] = {
    # CDN, Caching, Load Balancing -> scaling_aws
    "CDN": [DOC_SCALING_AWS],
    "caching": [DOC_SCALING_AWS, DOC_KV_CACHE],
    "load balancer": [DOC_SCALING_AWS],
    "scaling": [DOC_SCALING_AWS],
    
    # Database topics -> multiple
    "SQL": [DOC_SCALING_AWS],
    "NoSQL": [DOC_SCALING_AWS],
    "replication": [DOC_SCALING_AWS],
    "sharding": [DOC_SCALING_AWS],
    
    # Specific system designs
    "URL shortener": [DOC_PASTEBIN],
    "pastebin": [DOC_PASTEBIN],
    "bit.ly": [DOC_PASTEBIN],
    
    "web crawler": [DOC_WEB_CRAWLER],
    "crawler": [DOC_WEB_CRAWLER],
    
    "Twitter": [DOC_TWITTER],
    "timeline": [DOC_TWITTER],
    "social network": [DOC_SOCIAL_NETWORK, DOC_TWITTER],
    "social graph": [DOC_SOCIAL_NETWORK],
    
    "Mint": [DOC_MINT],
    "personal finance": [DOC_MINT],
    
    "Amazon": [DOC_AMAZON],
    "sales rank": [DOC_AMAZON],
    "e-commerce": [DOC_AMAZON],
    
    # General concepts (may appear in multiple docs)
    "microservices": [DOC_SCALING_AWS],
    "distributed": [DOC_SCALING_AWS],
    "CAP theorem": [DOC_SCALING_AWS],
    "consistent hashing": [DOC_SCALING_AWS],
    "message queue": [DOC_SCALING_AWS],
    "async": [DOC_SCALING_AWS],
}


def get_doc_ids_for_topic(topic: str, all_docs: List[Any]) -> List[str]:
    """Get doc_ids for documents matching a topic."""
    expected_h1s = TOPIC_TO_DOCS.get(topic.lower(), [])
    
    doc_ids = []
    for doc in all_docs:
        h1 = doc.metadata.get("h1", "")
        if h1 in expected_h1s:
            doc_id = doc.metadata.get("doc_id", "")
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
    
    return doc_ids


def find_matching_doc_ids(keywords: List[str], all_docs: List[Any]) -> List[str]:
    """Find doc_ids where content contains keywords."""
    doc_ids = set()
    
    for doc in all_docs:
        content_lower = doc.page_content.lower()
        h1_lower = doc.metadata.get("h1", "").lower()
        
        # Check if any keyword matches
        for kw in keywords:
            if kw.lower() in content_lower or kw.lower() in h1_lower:
                doc_id = doc.metadata.get("doc_id", "")
                if doc_id:
                    doc_ids.add(doc_id)
                break
    
    return list(doc_ids)


def create_golden_dataset() -> EvaluationDataset:
    """Create golden dataset with exact doc_id mappings."""
    
    # Fetch all documents to get doc_ids
    logger.info("Fetching all documents from vector store...")
    all_docs = fetch_all_documents()
    logger.info(f"Fetched {len(all_docs)} documents")
    
    # Build h1 -> doc_ids mapping
    h1_to_doc_ids: Dict[str, List[str]] = {}
    for doc in all_docs:
        h1 = doc.metadata.get("h1", "")
        doc_id = doc.metadata.get("doc_id", "")
        if h1 and doc_id:
            if h1 not in h1_to_doc_ids:
                h1_to_doc_ids[h1] = []
            if doc_id not in h1_to_doc_ids[h1]:
                h1_to_doc_ids[h1].append(doc_id)
    
    logger.info(f"Found {len(h1_to_doc_ids)} unique document topics")
    
    dataset = EvaluationDataset(
        version="2.0-golden",
        description="ISAAC Golden Dataset with exact doc_id mappings"
    )
    
    # ============================================
    # GOLDEN QUERIES - Exact doc_id mappings
    # ============================================
    
    # --- CDN / Caching / Scaling ---
    dataset.add_query(EvalQuery(
        query_id="G001",
        query="What is a CDN and how does it improve website performance?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.EASY,
        expected_topics=["CDN", "content delivery network", "caching"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_SCALING_AWS, []),
        expected_images=["scaling_aws"],
        expected_keywords=["edge", "cache", "latency", "geographic"],
        category="CDN",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="G002", 
        query="What is caching and what are common caching strategies?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.EASY,
        expected_topics=["cache", "caching strategies"],
        expected_doc_ids=(
            h1_to_doc_ids.get(DOC_SCALING_AWS, []) +
            h1_to_doc_ids.get(DOC_KV_CACHE, [])
        ),
        expected_images=["scaling_aws", "query_cache"],
        expected_keywords=["cache-aside", "write-through", "TTL", "eviction"],
        category="Caching",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="G003",
        query="What is a load balancer and why do we need it?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.EASY,
        expected_topics=["load balancer", "traffic distribution"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_SCALING_AWS, []),
        expected_images=["scaling_aws"],
        expected_keywords=["round robin", "health check", "horizontal scaling"],
        category="Load Balancing",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="G004",
        query="What is database replication and sharding?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["replication", "sharding", "database"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_SCALING_AWS, []),
        expected_images=["scaling_aws"],
        expected_keywords=["master", "slave", "partition", "consistency"],
        category="Databases",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="G005",
        query="Explain the differences between horizontal and vertical scaling",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.EASY,
        expected_topics=["scaling", "horizontal", "vertical"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_SCALING_AWS, []),
        expected_images=["scaling_aws"],
        expected_keywords=["scale out", "scale up", "servers"],
        category="Scaling",
    ))
    
    # --- URL Shortener / Pastebin ---
    dataset.add_query(EvalQuery(
        query_id="G006",
        query="How do I design a URL shortener like bit.ly?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["URL shortener", "hashing", "redirect"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_PASTEBIN, []),
        expected_images=["pastebin"],
        expected_keywords=["hash", "unique", "redirect", "base62"],
        category=CAT_SYSTEM_DESIGN,
    ))
    
    dataset.add_query(EvalQuery(
        query_id="G007",
        query="Design a pastebin service for sharing text snippets",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["pastebin", "text storage", "sharing"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_PASTEBIN, []),
        expected_images=["pastebin"],
        expected_keywords=["paste", "expire", "storage"],
        category=CAT_SYSTEM_DESIGN,
    ))
    
    # --- Web Crawler ---
    dataset.add_query(EvalQuery(
        query_id="G008",
        query="Design a web crawler that can index millions of pages",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["web crawler", "distributed", "indexing"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_WEB_CRAWLER, []),
        expected_images=["web_crawler"],
        expected_keywords=["URL frontier", "robots.txt", "politeness", "BFS"],
        category=CAT_SYSTEM_DESIGN,
    ))
    
    # --- Twitter ---
    dataset.add_query(EvalQuery(
        query_id="G009",
        query="How would you design Twitter's timeline feature?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["timeline", "fan-out", "Twitter"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_TWITTER, []),
        expected_images=["twitter"],
        expected_keywords=["fan-out", "push", "pull", "followers"],
        category="Social Media",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="G010",
        query="How does Twitter search work at scale?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["search", "Twitter", "indexing"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_TWITTER, []),
        expected_images=["twitter"],
        expected_keywords=["index", "search", "hashtag"],
        category="Social Media",
    ))
    
    # --- Social Network ---
    dataset.add_query(EvalQuery(
        query_id="G011",
        query="How to design data structures for a social network?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["social network", "graph", "data structures"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_SOCIAL_NETWORK, []),
        expected_images=["social_graph"],
        expected_keywords=["graph", "friend", "connection", "BFS"],
        category="Social Network",
    ))
    
    # --- Mint.com ---
    dataset.add_query(EvalQuery(
        query_id="G012",
        query="How would you design a personal finance app like Mint?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["Mint", "finance", "budgeting"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_MINT, []),
        expected_images=["mint"],
        expected_keywords=["account", "transaction", "budget", "category"],
        category="Finance",
    ))
    
    # --- Amazon Sales Rank ---
    dataset.add_query(EvalQuery(
        query_id="G013",
        query="How does Amazon calculate sales rank for products?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["Amazon", "sales rank", "ranking"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_AMAZON, []),
        expected_images=["sales_rank"],
        expected_keywords=["rank", "category", "sales", "update"],
        category="E-commerce",
    ))
    
    # --- Key-Value Cache ---
    dataset.add_query(EvalQuery(
        query_id="G014",
        query="Design a key-value cache for web server queries",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["cache", "key-value", "query cache"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_KV_CACHE, []),
        expected_images=["query_cache"],
        expected_keywords=["LRU", "eviction", "memory", "TTL"],
        category="Caching",
    ))
    
    # --- Cross-topic queries ---
    dataset.add_query(EvalQuery(
        query_id="G015",
        query="What are best practices for designing scalable systems?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.MEDIUM,
        expected_topics=["scalability", "best practices", "architecture"],
        expected_doc_ids=h1_to_doc_ids.get(DOC_SCALING_AWS, []),
        expected_images=["scaling_aws"],
        expected_keywords=["scale", "availability", "performance"],
        category="General",
    ))
    
    # ============================================
    # OFF-TOPIC QUERIES (Should Refuse)
    # ============================================
    
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
        forbidden_topics=["system", "architecture"],
        category="Off-topic",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="O003",
        query="What are the specs of the latest iPhone?",
        query_type=QueryType.OFF_TOPIC,
        difficulty=DifficultyLevel.EASY,
        should_refuse=True,
        category="Off-topic",
    ))
    
    # ============================================
    # EDGE CASES - "I Don't Know"
    # ============================================
    
    dataset.add_query(EvalQuery(
        query_id="E001",
        query="What is the exact internal architecture of TikTok's recommendation engine?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.HARD,
        should_say_idk=True,
        expected_topics=["TikTok", "recommendation"],
        category="Edge Case",
        notes="Specific proprietary system not in knowledge base",
    ))
    
    dataset.add_query(EvalQuery(
        query_id="E002",
        query="How does Spotify's machine learning pipeline work internally?",
        query_type=QueryType.TEXT_TO_ARCHITECTURE,
        difficulty=DifficultyLevel.HARD,
        should_say_idk=True,
        expected_topics=["Spotify", "ML"],
        category="Edge Case",
        notes="Proprietary system not in corpus",
    ))
    
    return dataset


def main():
    """Generate and save golden dataset."""
    dataset = create_golden_dataset()
    
    output_path = Path("isaac_eval/data/golden_dataset.json")
    dataset.save(output_path)
    
    print("\nGolden Dataset created!")
    print(f"Path: {output_path}")
    print("\nStatistics:")
    stats = dataset.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
