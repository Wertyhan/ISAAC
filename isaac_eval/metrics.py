"""Evaluation Metrics - Retrieval and Generation quality metrics."""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result for evaluation."""
    query_id: str
    retrieved_doc_ids: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    retrieved_images: List[str]
    scores: List[float]
    retrieval_time_ms: float
    
    @property
    def top_score(self) -> float:
        return max(self.scores) if self.scores else 0.0


@dataclass
class RetrievalMetrics:
    """Computed retrieval metrics."""
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_rate: float = 0.0
    image_hit_rate: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_top_score: float = 0.0
    query_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "hit_rate": self.hit_rate,
            "image_hit_rate": self.image_hit_rate,
            "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
            "avg_top_score": self.avg_top_score,
            "num_queries": len(self.query_results),
        }


class RetrievalEvaluator:
    """Evaluates retrieval quality."""
    
    STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for', 'with', 'as'}
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]
    
    def evaluate(self, results: List[RetrievalResult], expected: Dict[str, Dict[str, Any]]) -> RetrievalMetrics:
        """Evaluate retrieval results against ground truth."""
        metrics = RetrievalMetrics()
        if not results:
            return metrics
        
        recall_sums = defaultdict(float)
        precision_sums = defaultdict(float)
        reciprocal_ranks = []
        hits, image_hits, queries_with_images = 0, 0, 0
        total_time, total_score = 0.0, 0.0
        
        for result in results:
            exp = expected.get(result.query_id, {})
            processed = self._process_result(result, exp)
            
            for k in self.k_values:
                recall_sums[k] += processed["per_k"]["recall"][k]
                precision_sums[k] += processed["per_k"]["precision"][k]
            
            reciprocal_ranks.append(processed["rr"])
            hits += 1 if processed["hit"] else 0
            
            if processed["has_expected_images"]:
                queries_with_images += 1
                image_hits += 1 if processed["image_hit"] else 0
            
            total_time += result.retrieval_time_ms
            total_score += result.top_score
            metrics.query_results[result.query_id] = processed["query_metrics"]
        
        n = len(results)
        for k in self.k_values:
            metrics.recall_at_k[k] = recall_sums[k] / n
            metrics.precision_at_k[k] = precision_sums[k] / n
        
        metrics.mrr = sum(reciprocal_ranks) / n if reciprocal_ranks else 0.0
        metrics.hit_rate = hits / n
        metrics.image_hit_rate = image_hits / queries_with_images if queries_with_images > 0 else 0.0
        metrics.avg_retrieval_time_ms = total_time / n
        metrics.avg_top_score = total_score / n
        
        return metrics
    
    def _process_result(self, result: RetrievalResult, exp: Dict[str, Any]) -> Dict[str, Any]:
        expected_doc_ids = set(exp.get("expected_doc_ids", []))
        expected_topics = {t.lower() for t in exp.get("expected_topics", [])}
        expected_images = {img.lower() for img in exp.get("expected_images", [])}
        
        retrieved_doc_ids = set(result.retrieved_doc_ids[:max(self.k_values)])
        retrieved_topics = self._extract_topics(result.retrieved_chunks)
        retrieved_images = {img.lower() for img in result.retrieved_images}
        
        per_k = self._compute_per_k(result, expected_doc_ids, expected_topics)
        rr = self._compute_rr(result, expected_doc_ids, expected_topics)
        hit = self._check_hit(expected_topics, expected_doc_ids, retrieved_doc_ids, retrieved_topics)
        image_hit = self._check_image_hit(expected_images, retrieved_images) if expected_images else None
        
        query_metrics = {"query_id": result.query_id}
        for k in self.k_values:
            query_metrics[f"recall@{k}"] = per_k["recall"][k]
            query_metrics[f"precision@{k}"] = per_k["precision"][k]
        query_metrics.update({
            "reciprocal_rank": rr, "hit": hit, "retrieval_time_ms": result.retrieval_time_ms,
            "top_score": result.top_score
        })
        if image_hit is not None:
            query_metrics["image_hit"] = image_hit
        
        return {"query_metrics": query_metrics, "per_k": per_k, "rr": rr, 
                "hit": hit, "image_hit": image_hit, "has_expected_images": bool(expected_images)}
    
    def _compute_per_k(self, result: RetrievalResult, expected_docs: Set[str], expected_topics: Set[str]) -> Dict:
        metrics = {"recall": {}, "precision": {}}
        has_chunks = bool(result.retrieved_chunks)
        
        for k in self.k_values:
            top_k_docs = set(result.retrieved_doc_ids[:k])
            top_k_topics = self._extract_topics(result.retrieved_chunks[:k])
            
            if expected_docs:
                metrics["recall"][k] = len(top_k_docs & expected_docs) / len(expected_docs)
                metrics["precision"][k] = len(top_k_docs & expected_docs) / k if k > 0 else 0.0
            elif expected_topics:
                overlap = self._topic_overlap(expected_topics, top_k_topics)
                metrics["recall"][k] = metrics["precision"][k] = overlap
            else:
                metrics["recall"][k] = metrics["precision"][k] = 1.0 if has_chunks else 0.0
        
        return metrics
    
    def _compute_rr(self, result: RetrievalResult, expected_docs: Set[str], expected_topics: Set[str]) -> float:
        for i, (doc_id, chunk) in enumerate(zip(result.retrieved_doc_ids, result.retrieved_chunks), 1):
            if doc_id in expected_docs:
                return 1.0 / i
            if expected_topics:
                chunk_topics = self._extract_topics([chunk])
                if self._topic_overlap(expected_topics, chunk_topics) > 0.5:
                    return 1.0 / i
        return 0.0
    
    def _check_hit(self, expected_topics: Set[str], expected_docs: Set[str], 
                   retrieved_docs: Set[str], retrieved_topics: Set[str]) -> bool:
        if expected_topics:
            return self._topic_overlap(expected_topics, retrieved_topics) > 0.3
        if expected_docs:
            return len(retrieved_docs & expected_docs) > 0
        return True
    
    def _check_image_hit(self, expected: Set[str], retrieved: Set[str]) -> bool:
        for exp_img in expected:
            for img_id in retrieved:
                if exp_img in img_id or img_id in exp_img:
                    return True
                exp_words = set(exp_img.replace('_', ' ').replace('-', ' ').split())
                img_words = set(img_id.replace('_', ' ').replace('-', ' ').split())
                if exp_words & img_words:
                    return True
        return False
    
    # Topic synonyms for better semantic matching
    TOPIC_SYNONYMS = {
        "microservices": {"microservice", "service", "distributed", "services"},
        "message queue": {"queue", "rabbitmq", "kafka", "sqs", "async", "asynchronous", "queues"},
        "real-time": {"realtime", "real", "time", "instant", "live", "streaming"},
        "scalability": {"scale", "scaling", "horizontal", "vertical", "million", "millions", "billion"},
        "notification": {"notify", "alert", "push", "budget", "notifies", "notification"},
        "caching": {"cache", "redis", "memcached", "lru", "cached", "caches"},
        "database": {"db", "sql", "nosql", "store", "storage", "data"},
        "replication": {"replica", "replicate", "master", "slave", "replicated", "replicates"},
        "sharding": {"shard", "partition", "sharded", "partitioned"},
        "url shortener": {"pastebin", "hash", "shortener", "short", "redirect"},
        "web crawler": {"crawler", "crawl", "spider", "index", "indexing", "frontier"},
        "load balancer": {"load", "balancer", "nginx", "haproxy", "traffic"},
        "cdn": {"cdn", "content", "delivery", "edge", "geographic"},
        "consistent hashing": {"hash", "hashing", "consistent", "ring", "distributed"},
        "cap theorem": {"cap", "consistency", "availability", "partition"},
        "rate limiting": {"rate", "limit", "throttle", "limiting"},
    }
    
    def _extract_topics(self, chunks: List[Dict[str, Any]]) -> Set[str]:
        topics = set()
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            for key in ("h1", "h2", "project_name"):
                if metadata.get(key):
                    topics.update(self._tokenize(metadata[key]))
            # Read more content for better topic matching
            content = chunk.get("content", "")[:1500].lower()
            topics.update(self._tokenize(content))
        return topics
    
    def _tokenize(self, text: str) -> Set[str]:
        text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
        return {w for w in text.split() if len(w) > 2 and w not in self.STOPWORDS}
    
    def _topic_overlap(self, expected: Set[str], retrieved: Set[str]) -> float:
        if not expected:
            return 1.0
        matches = 0
        for exp in expected:
            exp_lower = exp.lower()
            # Direct match
            if exp_lower in retrieved:
                matches += 1
                continue
            # Check synonyms
            synonyms = self.TOPIC_SYNONYMS.get(exp_lower, set())
            if synonyms & retrieved:
                matches += 0.8
                continue
            # Check if any word from a multi-word expected topic exists
            exp_words = set(exp_lower.split())
            if len(exp_words) > 1 and exp_words & retrieved:
                matches += 0.6
                continue
            # Substring match
            for ret in retrieved:
                if exp_lower in ret or ret in exp_lower:
                    matches += 0.5
                    break
        return min(matches / len(expected), 1.0)


@dataclass
class GenerationResult:
    """Single generation result for evaluation."""
    query_id: str
    query: str
    response: str
    citations: List[str]
    retrieved_context: str
    generation_time_ms: float
    refused: bool = False
    said_idk: bool = False


@dataclass
class GenerationMetrics:
    """Computed generation metrics."""
    faithfulness_score: float = 0.0
    citation_correctness: float = 0.0
    answer_relevance: float = 0.0
    refusal_accuracy: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_response_length: float = 0.0
    query_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness_score": self.faithfulness_score,
            "citation_correctness": self.citation_correctness,
            "answer_relevance": self.answer_relevance,
            "refusal_accuracy": self.refusal_accuracy,
            "avg_generation_time_ms": self.avg_generation_time_ms,
            "avg_response_length": self.avg_response_length,
            "num_queries": len(self.query_results),
        }


class GenerationEvaluator:
    """Evaluates generation quality."""
    
    REFUSAL_PATTERNS = [
        r"i cannot|i can't|unable to|outside.*scope|not.*expertise",
        r"off.?topic|beyond.*knowledge|don't have.*information",
        r"i specialize in|my focus is|i'm designed for",
    ]
    IDK_PATTERNS = [
        r"i don't know|i do not know|i'm not sure|uncertain",
        r"no.*information|not.*in.*knowledge|cannot find",
    ]
    
    def __init__(self, faithfulness_threshold: float = 0.7):
        self.faithfulness_threshold = faithfulness_threshold
    
    def evaluate(self, results: List[GenerationResult], expected: Dict[str, Dict[str, Any]]) -> GenerationMetrics:
        """Evaluate generation results."""
        metrics = GenerationMetrics()
        if not results:
            return metrics
        
        aggregator = self._EvaluationAggregator()
        
        for result in results:
            exp = expected.get(result.query_id, {})
            query_metrics = self._evaluate_single_result(result, exp, aggregator)
            metrics.query_results[result.query_id] = query_metrics
        
        self._finalize_metrics(metrics, aggregator, len(results))
        return metrics
    
    def _evaluate_single_result(self, result: GenerationResult, exp: Dict[str, Any], 
                                 aggregator: '_EvaluationAggregator') -> Dict[str, Any]:
        """Evaluate a single generation result."""
        query_metrics: Dict[str, Any] = {"query_id": result.query_id}
        
        # Check refusal
        refusal_result = self._evaluate_refusal(result, exp)
        if refusal_result is not None:
            aggregator.refusal_total += 1
            if refusal_result:
                aggregator.refusal_correct += 1
                query_metrics["refusal_correct"] = True
        
        # Evaluate content quality
        content_scores = self._evaluate_content_quality(result, exp)
        self._update_aggregator_with_scores(aggregator, content_scores, query_metrics)
        
        # Update timing stats
        query_metrics["generation_time_ms"] = result.generation_time_ms
        query_metrics["response_length"] = len(result.response)
        aggregator.total_time += result.generation_time_ms
        aggregator.total_length += len(result.response)
        
        return query_metrics
    
    def _update_aggregator_with_scores(self, aggregator: '_EvaluationAggregator', 
                                        scores: Dict[str, Optional[float]], 
                                        query_metrics: Dict[str, Any]) -> None:
        """Update the aggregator with content quality scores."""
        if scores.get("faithfulness") is not None:
            aggregator.faithfulness_scores.append(scores["faithfulness"])
            query_metrics["faithfulness"] = scores["faithfulness"]
        if scores.get("citation") is not None:
            aggregator.citation_scores.append(scores["citation"])
            query_metrics["citation_correctness"] = scores["citation"]
        if scores.get("relevance") is not None:
            aggregator.relevance_scores.append(scores["relevance"])
            query_metrics["answer_relevance"] = scores["relevance"]
    
    def _finalize_metrics(self, metrics: GenerationMetrics, aggregator: '_EvaluationAggregator', n: int) -> None:
        """Finalize the metrics with aggregated scores."""
        metrics.faithfulness_score = self._safe_avg(aggregator.faithfulness_scores)
        metrics.citation_correctness = self._safe_avg(aggregator.citation_scores)
        metrics.answer_relevance = self._safe_avg(aggregator.relevance_scores)
        metrics.refusal_accuracy = aggregator.refusal_correct / aggregator.refusal_total if aggregator.refusal_total > 0 else 1.0
        metrics.avg_generation_time_ms = aggregator.total_time / n
        metrics.avg_response_length = aggregator.total_length / n
    
    @staticmethod
    def _safe_avg(values: List[float]) -> float:
        """Compute average, returning 0.0 for empty list."""
        return sum(values) / len(values) if values else 0.0
    
    class _EvaluationAggregator:
        """Helper class to aggregate evaluation scores."""
        def __init__(self):
            self.faithfulness_scores: List[float] = []
            self.citation_scores: List[float] = []
            self.relevance_scores: List[float] = []
            self.refusal_correct = 0
            self.refusal_total = 0
            self.total_time = 0.0
            self.total_length = 0
    
    def _evaluate_refusal(self, result: GenerationResult, exp: Dict[str, Any]) -> Optional[bool]:
        """Evaluate if refusal was handled correctly. Returns None if not a refusal case."""
        should_refuse = exp.get("should_refuse", False)
        if not should_refuse:
            return None
        return self._detected_refusal(result.response)
    
    def _evaluate_content_quality(self, result: GenerationResult, exp: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Evaluate content quality scores (faithfulness, citation, relevance)."""
        scores: Dict[str, Optional[float]] = {"faithfulness": None, "citation": None, "relevance": None}
        
        should_refuse = exp.get("should_refuse", False)
        should_say_idk = exp.get("should_say_idk", False)
        
        if should_refuse or should_say_idk:
            return scores
        
        scores["faithfulness"] = self._compute_faithfulness(
            result.response, result.retrieved_context, exp.get("forbidden_topics", [])
        )
        
        if result.citations:
            scores["citation"] = self._compute_citation_correctness(result.citations, result.retrieved_context)
        
        keywords = exp.get("expected_keywords", [])
        if keywords:
            scores["relevance"] = self._compute_relevance(result.response, keywords)
        
        return scores
    
    def _detected_refusal(self, response: str) -> bool:
        response_lower = response.lower()
        return any(re.search(p, response_lower) for p in self.REFUSAL_PATTERNS)
    
    def _detected_idk(self, response: str) -> bool:
        response_lower = response.lower()
        return any(re.search(p, response_lower) for p in self.IDK_PATTERNS)
    
    def _compute_faithfulness(self, response: str, context: str, forbidden: List[str]) -> float:
        response_lower, context_lower = response.lower(), context.lower()
        
        penalty = sum(0.2 for t in forbidden if t.lower() in response_lower and t.lower() not in context_lower)
        phrases = self._extract_key_phrases(response)
        
        if not phrases:
            return max(0.0, 1.0 - penalty)
        
        grounded = 0
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in context_lower:
                grounded += 1
            else:
                words = phrase_lower.split()
                matches = sum(1 for w in words if w in context_lower and len(w) > 3)
                if matches >= len(words) * 0.5:
                    grounded += 0.5
        
        return max(0.0, min(1.0, grounded / len(phrases) - penalty))
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        phrases = []
        phrases.extend(re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text))
        phrases.extend(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', text))
        phrases.extend(re.findall(r'\b[a-zA-Z]+[-_][a-zA-Z]+\b', text))
        
        keywords = ['database', 'cache', 'server', 'client', 'api', 'queue', 'message',
                   'load balancer', 'microservice', 'distributed', 'replication', 'sharding']
        text_lower = text.lower()
        phrases.extend(kw for kw in keywords if kw in text_lower)
        
        return list(set(phrases))[:20]
    
    def _compute_citation_correctness(self, citations: List[str], context: str) -> float:
        if not citations:
            return 1.0
        
        context_lower = context.lower()
        
        # Count how many SOURCE [N] markers are in the context
        import re
        source_markers = set(re.findall(r'source \[(\d+)\]', context_lower))
        
        correct = 0
        for cit in citations:
            cit_lower = cit.lower().strip()
            
            # Handle numbered citations (source_1, source_2, etc.)
            if cit_lower.startswith('source_'):
                num = cit_lower.replace('source_', '')
                if num in source_markers:
                    correct += 1
                continue
            
            # For named citations, use the standard scoring
            correct += self._score_single_citation(cit, context_lower)
        
        return correct / len(citations) if citations else 1.0
    
    def _score_single_citation(self, citation: str, context_lower: str) -> float:
        """Score a single citation against the context."""
        skip_words = {'the', 'a', 'an', 'of', 'for', 'to', 'in', 'on', 'and', 'or', 'with'}
        cit_lower = citation.lower().strip()
        
        # Skip generic citation markers
        if cit_lower in ('source', 'reference', 'link', 'see', 'note'):
            return 0.0
        
        # Handle numbered citations (source_1, source_2, etc.)
        # These are valid if the context has the corresponding SOURCE [N] marker
        if cit_lower.startswith('source_'):
            num = cit_lower.replace('source_', '')
            # Check if context contains this source number marker (our format: === SOURCE [N]: ===)
            if f'=== source [{num}]' in context_lower or f'source [{num}]' in context_lower:
                return 1.0
            # Also check for "[N]" in available sources section
            if f'[{num}]' in context_lower:
                return 1.0
            return 0.0
        
        # Exact match
        if cit_lower in context_lower:
            return 1.0
        
        # Partial match based on keywords
        key_words = [w for w in cit_lower.split() if len(w) > 2 and w not in skip_words]
        if not key_words:
            return 0.0
        
        matches = sum(1 for w in key_words if w in context_lower)
        ratio = matches / len(key_words)
        return self._compute_ratio_score(ratio)
    
    @staticmethod
    def _compute_ratio_score(ratio: float) -> float:
        """Convert a ratio to a score for citation correctness."""
        if ratio >= 0.7:
            return 1.0
        if ratio >= 0.4:
            return 0.5
        return 0.0
    
    def _compute_relevance(self, response: str, keywords: List[str]) -> float:
        if not keywords:
            return 1.0
        
        response_lower = response.lower()
        found = 0.0
        
        for kw in keywords:
            kw_lower = kw.lower()
            # Exact match
            if kw_lower in response_lower:
                found += 1.0
                continue
            # Stem-based match: check if keyword root appears (e.g., "cache" matches "caching")
            kw_stem = kw_lower.rstrip('seding')[:4] if len(kw_lower) > 4 else kw_lower
            if kw_stem in response_lower:
                found += 0.8
                continue
            # Word-based partial match
            words = kw_lower.split()
            if any(w in response_lower for w in words if len(w) > 3):
                found += 0.5
        
        return min(found / len(keywords), 1.0)
