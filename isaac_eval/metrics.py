"""Evaluation Metrics - Retrieval and Generation quality metrics."""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================
# RETRIEVAL METRICS
# ============================================

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
    
    # Per-query metrics
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_rate: float = 0.0
    image_hit_rate: float = 0.0
    
    # Aggregate metrics
    avg_retrieval_time_ms: float = 0.0
    avg_top_score: float = 0.0
    
    # Per-query details
    query_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
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
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]
    
    def _compute_recall_score(
        self,
        top_k_docs: Set[str],
        top_k_topics: Set[str],
        expected_doc_ids: Set[str],
        expected_topics: Set[str],
        has_chunks: bool,
    ) -> float:
        """Compute recall score based on available ground truth."""
        if expected_doc_ids:
            return len(top_k_docs & expected_doc_ids) / len(expected_doc_ids)
        if expected_topics:
            return self._topic_overlap(expected_topics, top_k_topics)
        return 1.0 if has_chunks else 0.0
    
    def _compute_precision_score(
        self,
        k: int,
        top_k_docs: Set[str],
        top_k_topics: Set[str],
        expected_doc_ids: Set[str],
        expected_topics: Set[str],
        has_chunks: bool,
    ) -> float:
        """Compute precision score based on available ground truth."""
        if expected_doc_ids:
            return len(top_k_docs & expected_doc_ids) / k if k > 0 else 0.0
        if expected_topics:
            return self._topic_overlap(expected_topics, top_k_topics)
        return 1.0 if has_chunks else 0.0
    
    def _compute_per_k_metrics(
        self,
        result: RetrievalResult,
        expected_doc_ids: Set[str],
        expected_topics: Set[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute recall and precision for each k value."""
        metrics = {"recall": {}, "precision": {}}
        has_chunks = bool(result.retrieved_chunks)
        
        for k in self.k_values:
            top_k_docs = set(result.retrieved_doc_ids[:k])
            top_k_topics = self._extract_topics_from_chunks(result.retrieved_chunks[:k])
            
            metrics["recall"][k] = self._compute_recall_score(
                top_k_docs, top_k_topics, expected_doc_ids, expected_topics, has_chunks
            )
            metrics["precision"][k] = self._compute_precision_score(
                k, top_k_docs, top_k_topics, expected_doc_ids, expected_topics, has_chunks
            )
        
        return metrics
    
    def _check_image_hit(
        self,
        expected_images: Set[str],
        retrieved_image_ids: Set[str],
    ) -> bool:
        """Check if any expected image was retrieved."""
        for exp_img in expected_images:
            exp_lower = exp_img.lower()
            for img_id in retrieved_image_ids:
                # Direct match or partial match
                if exp_lower in img_id or img_id in exp_lower:
                    return True
                # Word-level matching for multi-word patterns
                exp_words = set(exp_lower.replace('_', ' ').replace('-', ' ').split())
                img_words = set(img_id.replace('_', ' ').replace('-', ' ').split())
                if exp_words & img_words:
                    return True
        return False
    
    def _check_topic_hit(
        self,
        expected_topics: Set[str],
        expected_doc_ids: Set[str],
        retrieved_doc_ids: Set[str],
        retrieved_topics: Set[str],
    ) -> bool:
        """Check if we hit at least one relevant result."""
        if expected_topics:
            return self._topic_overlap(expected_topics, retrieved_topics) > 0.3
        if expected_doc_ids:
            return len(retrieved_doc_ids & expected_doc_ids) > 0
        return True
    
    def _process_single_result(
        self,
        result: RetrievalResult,
        exp: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single retrieval result and return query-level metrics."""
        expected_doc_ids = set(exp.get("expected_doc_ids", []))
        expected_topics = {t.lower() for t in exp.get("expected_topics", [])}
        expected_images = {img.lower() for img in exp.get("expected_images", [])}
        
        # Get retrieved content
        retrieved_doc_ids = set(result.retrieved_doc_ids[:max(self.k_values)])
        retrieved_topics = self._extract_topics_from_chunks(result.retrieved_chunks)
        retrieved_image_ids = {img.lower() for img in result.retrieved_images}
        
        # Calculate per-k metrics
        per_k = self._compute_per_k_metrics(result, expected_doc_ids, expected_topics)
        
        query_metrics = {"query_id": result.query_id}
        for k in self.k_values:
            query_metrics[f"recall@{k}"] = per_k["recall"][k]
            query_metrics[f"precision@{k}"] = per_k["precision"][k]
        
        # MRR
        rr = self._compute_reciprocal_rank(
            result.retrieved_doc_ids,
            expected_doc_ids,
            expected_topics,
            result.retrieved_chunks,
        )
        query_metrics["reciprocal_rank"] = rr
        
        # Hit rate
        hit = self._check_topic_hit(expected_topics, expected_doc_ids, retrieved_doc_ids, retrieved_topics)
        query_metrics["hit"] = hit
        
        # Image hit rate
        image_hit = None
        if expected_images:
            image_hit = self._check_image_hit(expected_images, retrieved_image_ids)
            query_metrics["image_hit"] = image_hit
        
        # Timing
        query_metrics["retrieval_time_ms"] = result.retrieval_time_ms
        query_metrics["top_score"] = result.top_score
        
        return {
            "query_metrics": query_metrics,
            "per_k": per_k,
            "rr": rr,
            "hit": hit,
            "image_hit": image_hit,
            "has_expected_images": bool(expected_images),
        }
    
    def evaluate(
        self,
        results: List[RetrievalResult],
        expected: Dict[str, Dict[str, Any]],
    ) -> RetrievalMetrics:
        """Evaluate retrieval results against ground truth."""
        metrics = RetrievalMetrics()
        
        if not results:
            return metrics
        
        # Initialize accumulators
        recall_sums = defaultdict(float)
        precision_sums = defaultdict(float)
        reciprocal_ranks = []
        hits = 0
        image_hits = 0
        queries_with_expected_images = 0
        total_time = 0.0
        total_top_score = 0.0
        
        for result in results:
            exp = expected.get(result.query_id, {})
            processed = self._process_single_result(result, exp)
            
            # Aggregate metrics
            for k in self.k_values:
                recall_sums[k] += processed["per_k"]["recall"][k]
                precision_sums[k] += processed["per_k"]["precision"][k]
            
            reciprocal_ranks.append(processed["rr"])
            if processed["hit"]:
                hits += 1
            
            if processed["has_expected_images"]:
                queries_with_expected_images += 1
                if processed["image_hit"]:
                    image_hits += 1
            
            total_time += result.retrieval_time_ms
            total_top_score += result.top_score
            
            metrics.query_results[result.query_id] = processed["query_metrics"]
        
        # Compute averages
        n = len(results)
        
        for k in self.k_values:
            metrics.recall_at_k[k] = recall_sums[k] / n
            metrics.precision_at_k[k] = precision_sums[k] / n
        
        metrics.mrr = sum(reciprocal_ranks) / n if reciprocal_ranks else 0.0
        metrics.hit_rate = hits / n
        metrics.image_hit_rate = image_hits / queries_with_expected_images if queries_with_expected_images > 0 else 0.0
        metrics.avg_retrieval_time_ms = total_time / n
        metrics.avg_top_score = total_top_score / n
        
        return metrics
    
    def _extract_topics_from_chunks(self, chunks: List[Dict[str, Any]]) -> Set[str]:
        """Extract topic keywords from chunk content and metadata."""
        topics = set()
        
        for chunk in chunks:
            # From metadata
            metadata = chunk.get("metadata", {})
            if metadata.get("h1"):
                topics.update(self._tokenize(metadata["h1"]))
            if metadata.get("h2"):
                topics.update(self._tokenize(metadata["h2"]))
            if metadata.get("project_name"):
                topics.update(self._tokenize(metadata["project_name"]))
            
            # From content (first 500 chars)
            content = chunk.get("content", "")[:500].lower()
            topics.update(self._tokenize(content))
        
        return topics
    
    def _tokenize(self, text: str) -> Set[str]:
        """Simple tokenization for topic matching."""
        text = text.lower()
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()
        # Filter short words and common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for', 'with', 'as'}
        return {w for w in words if len(w) > 2 and w not in stopwords}
    
    def _topic_overlap(self, expected: Set[str], retrieved: Set[str]) -> float:
        """Compute topic overlap ratio."""
        if not expected:
            return 1.0
        
        # Partial matching: check if expected topics appear in retrieved
        matches = 0
        for exp_topic in expected:
            if exp_topic in retrieved:
                matches += 1
            else:
                # Check partial match
                for ret_topic in retrieved:
                    if exp_topic in ret_topic or ret_topic in exp_topic:
                        matches += 0.5
                        break
        
        return min(matches / len(expected), 1.0)
    
    def _compute_reciprocal_rank(
        self,
        retrieved_doc_ids: List[str],
        expected_doc_ids: Set[str],
        expected_topics: Set[str],
        chunks: List[Dict[str, Any]],
    ) -> float:
        """Compute reciprocal rank of first relevant result."""
        for i, (doc_id, chunk) in enumerate(zip(retrieved_doc_ids, chunks), 1):
            # Check doc_id match
            if doc_id in expected_doc_ids:
                return 1.0 / i
            
            # Check topic match
            if expected_topics:
                chunk_topics = self._extract_topics_from_chunks([chunk])
                if self._topic_overlap(expected_topics, chunk_topics) > 0.5:
                    return 1.0 / i
        
        return 0.0


# ============================================
# GENERATION METRICS
# ============================================

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
    idk_accuracy: float = 0.0
    
    avg_generation_time_ms: float = 0.0
    avg_response_length: float = 0.0
    
    # Per-query details
    query_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "faithfulness_score": self.faithfulness_score,
            "citation_correctness": self.citation_correctness,
            "answer_relevance": self.answer_relevance,
            "refusal_accuracy": self.refusal_accuracy,
            "idk_accuracy": self.idk_accuracy,
            "avg_generation_time_ms": self.avg_generation_time_ms,
            "avg_response_length": self.avg_response_length,
            "num_queries": len(self.query_results),
        }


class GenerationEvaluator:
    """Evaluates generation quality."""
    
    # Patterns indicating refusal
    REFUSAL_PATTERNS = [
        r"i cannot|i can't|unable to|outside.*scope|not.*expertise",
        r"off.?topic|beyond.*knowledge|don't have.*information",
        r"i specialize in|my focus is|i'm designed for",
    ]
    
    # Patterns indicating "I don't know"
    IDK_PATTERNS = [
        r"i don't know|i do not know|i'm not sure|uncertain",
        r"no.*information|not.*in.*knowledge|cannot find",
        r"don't have.*details|insufficient.*context",
    ]
    
    def __init__(self, faithfulness_threshold: float = 0.7):
        self.faithfulness_threshold = faithfulness_threshold
    
    def _process_refusal_check(
        self,
        result: GenerationResult,
        exp: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check refusal accuracy for a single result."""
        should_refuse = exp.get("should_refuse", False)
        if not should_refuse:
            return {"checked": False}
        
        actually_refused = self._detected_refusal(result.response)
        return {
            "checked": True,
            "correct": actually_refused,
            "metrics": {
                "refusal_expected": True,
                "refusal_detected": actually_refused,
            }
        }
    
    def _process_idk_check(
        self,
        result: GenerationResult,
        exp: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check IDK accuracy for a single result."""
        should_idk = exp.get("should_say_idk", False)
        if not should_idk:
            return {"checked": False}
        
        actually_idk = self._detected_idk(result.response)
        return {
            "checked": True,
            "correct": actually_idk,
            "metrics": {
                "idk_expected": True,
                "idk_detected": actually_idk,
            }
        }
    
    def _process_content_metrics(
        self,
        result: GenerationResult,
        exp: Dict[str, Any],
        should_refuse: bool,
        should_idk: bool,
    ) -> Dict[str, Any]:
        """Compute faithfulness, citation, and relevance metrics."""
        metrics = {}
        scores = {"faithfulness": None, "citation": None, "relevance": None}
        
        if should_refuse or should_idk:
            return {"metrics": metrics, "scores": scores}
        
        # Faithfulness
        faithfulness = self._compute_faithfulness(
            result.response,
            result.retrieved_context,
            exp.get("forbidden_topics", []),
        )
        scores["faithfulness"] = faithfulness
        metrics["faithfulness"] = faithfulness
        
        # Citation correctness
        if result.citations:
            citation_score = self._compute_citation_correctness(
                result.citations,
                result.retrieved_context,
            )
            scores["citation"] = citation_score
            metrics["citation_correctness"] = citation_score
        
        # Answer relevance
        expected_keywords = exp.get("expected_keywords", [])
        if expected_keywords:
            relevance = self._compute_relevance(result.response, expected_keywords)
            scores["relevance"] = relevance
            metrics["answer_relevance"] = relevance
        
        return {"metrics": metrics, "scores": scores}
    
    def _aggregate_check_result(
        self,
        check_result: Dict[str, Any],
        total_counter: int,
        correct_counter: int,
    ) -> tuple[int, int]:
        """Aggregate refusal/idk check results into counters."""
        if check_result["checked"]:
            total_counter += 1
            if check_result["correct"]:
                correct_counter += 1
        return total_counter, correct_counter
    
    def _compute_safe_average(self, scores: List[float], default: float = 0.0) -> float:
        """Compute average of a list, returning default if empty."""
        return sum(scores) / len(scores) if scores else default
    
    def _compute_safe_ratio(self, numerator: int, denominator: int, default: float = 1.0) -> float:
        """Compute ratio, returning default if denominator is zero."""
        return numerator / denominator if denominator > 0 else default
    
    def _collect_score(self, scores_list: List[float], score: Optional[float]) -> None:
        """Append score to list if not None."""
        if score is not None:
            scores_list.append(score)
    
    def _process_single_generation_result(
        self,
        result: GenerationResult,
        exp: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single generation result and return aggregated data."""
        query_metrics = {"query_id": result.query_id}
        
        # Refusal check
        refusal_result = self._process_refusal_check(result, exp)
        if refusal_result["checked"]:
            query_metrics.update(refusal_result["metrics"])
        
        # IDK check
        idk_result = self._process_idk_check(result, exp)
        if idk_result["checked"]:
            query_metrics.update(idk_result["metrics"])
        
        # Content metrics
        content_result = self._process_content_metrics(
            result, exp,
            exp.get("should_refuse", False),
            exp.get("should_say_idk", False),
        )
        query_metrics.update(content_result["metrics"])
        
        # Timing and length
        query_metrics["generation_time_ms"] = result.generation_time_ms
        query_metrics["response_length"] = len(result.response)
        
        return {
            "query_metrics": query_metrics,
            "refusal": refusal_result,
            "idk": idk_result,
            "scores": content_result["scores"],
        }
    
    def evaluate(
        self,
        results: List[GenerationResult],
        expected: Dict[str, Dict[str, Any]],
    ) -> GenerationMetrics:
        """Evaluate generation results."""
        metrics = GenerationMetrics()
        
        if not results:
            return metrics
        
        faithfulness_scores: List[float] = []
        citation_scores: List[float] = []
        relevance_scores: List[float] = []
        refusal_correct = 0
        refusal_total = 0
        idk_correct = 0
        idk_total = 0
        total_time = 0.0
        total_length = 0
        
        for result in results:
            exp = expected.get(result.query_id, {})
            processed = self._process_single_generation_result(result, exp)
            
            # Aggregate refusal stats
            if processed["refusal"]["checked"]:
                refusal_total += 1
                refusal_correct += 1 if processed["refusal"]["correct"] else 0
            
            # Aggregate IDK stats
            if processed["idk"]["checked"]:
                idk_total += 1
                idk_correct += 1 if processed["idk"]["correct"] else 0
            
            # Collect scores
            self._collect_score(faithfulness_scores, processed["scores"]["faithfulness"])
            self._collect_score(citation_scores, processed["scores"]["citation"])
            self._collect_score(relevance_scores, processed["scores"]["relevance"])
            
            # Totals
            total_time += result.generation_time_ms
            total_length += len(result.response)
            
            metrics.query_results[result.query_id] = processed["query_metrics"]
        
        # Compute averages
        n = len(results)
        
        metrics.faithfulness_score = self._compute_safe_average(faithfulness_scores)
        metrics.citation_correctness = self._compute_safe_average(citation_scores)
        metrics.answer_relevance = self._compute_safe_average(relevance_scores)
        metrics.refusal_accuracy = self._compute_safe_ratio(refusal_correct, refusal_total)
        metrics.idk_accuracy = self._compute_safe_ratio(idk_correct, idk_total)
        metrics.avg_generation_time_ms = total_time / n
        metrics.avg_response_length = total_length / n
        
        return metrics
    
    def _detected_refusal(self, response: str) -> bool:
        """Check if response contains refusal patterns."""
        response_lower = response.lower()
        for pattern in self.REFUSAL_PATTERNS:
            if re.search(pattern, response_lower):
                return True
        return False
    
    def _detected_idk(self, response: str) -> bool:
        """Check if response indicates "I don't know"."""
        response_lower = response.lower()
        for pattern in self.IDK_PATTERNS:
            if re.search(pattern, response_lower):
                return True
        return False
    
    def _compute_faithfulness(
        self,
        response: str,
        context: str,
        forbidden_topics: List[str],
    ) -> float:
        """Compute faithfulness score based on context grounding."""
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Penalize forbidden topics
        penalty = 0.0
        for topic in forbidden_topics:
            if topic.lower() in response_lower and topic.lower() not in context_lower:
                penalty += 0.2
        
        # Extract key phrases from response
        response_phrases = self._extract_key_phrases(response)
        
        # Check how many are grounded in context
        if not response_phrases:
            return max(0.0, 1.0 - penalty)
        
        grounded = 0
        for phrase in response_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in context_lower:
                grounded += 1
            else:
                # Partial match
                words = phrase_lower.split()
                word_matches = sum(1 for w in words if w in context_lower and len(w) > 3)
                if word_matches >= len(words) * 0.5:
                    grounded += 0.5
        
        score = grounded / len(response_phrases)
        return max(0.0, min(1.0, score - penalty))
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases (technical terms, concepts) from text."""
        phrases = []
        
        # Multi-word capitalized phrases
        cap_pattern = r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'
        phrases.extend(re.findall(cap_pattern, text))
        
        # Technical terms: split into separate simpler patterns
        # CamelCase words
        phrases.extend(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', text))
        # Hyphenated/underscored terms
        phrases.extend(re.findall(r'\b[a-zA-Z]+[-_][a-zA-Z]+\b', text))
        # Words with numbers
        phrases.extend(re.findall(r'\b[a-zA-Z]+\d+\b', text))
        
        # Common technical keywords
        keywords = ['database', 'cache', 'server', 'client', 'api', 'queue', 'message',
                   'load balancer', 'microservice', 'distributed', 'replication', 'sharding']
        text_lower = text.lower()
        phrases.extend(kw for kw in keywords if kw in text_lower)
        
        return list(set(phrases))[:20]
    
    def _compute_citation_correctness(
        self,
        citations: List[str],
        context: str,
    ) -> float:
        """Check if citations match retrieved context sources."""
        if not citations:
            return 1.0
        
        context_lower = context.lower()
        correct = 0
        
        for citation in citations:
            citation_lower = citation.lower().strip()
            
            # Skip common false positives
            if citation_lower in ('source', 'reference', 'link', 'see', 'note'):
                continue
            
            # Direct match in context
            if citation_lower in context_lower:
                correct += 1
                continue
            
            # Extract meaningful terms (filter articles, prepositions)
            skip_words = {'the', 'a', 'an', 'of', 'for', 'to', 'in', 'on', 'and', 'or', 'with'}
            key_words = [w for w in citation_lower.split() if len(w) > 2 and w not in skip_words]
            
            if not key_words:
                continue
            
            # Check if key terms appear in context
            matches = sum(1 for w in key_words if w in context_lower)
            match_ratio = matches / len(key_words) if key_words else 0
            
            if match_ratio >= 0.7:
                correct += 1
            elif match_ratio >= 0.4:
                correct += 0.5
        
        return correct / len(citations) if citations else 1.0
    
    def _compute_relevance(
        self,
        response: str,
        expected_keywords: List[str],
    ) -> float:
        """Check if response covers expected keywords."""
        if not expected_keywords:
            return 1.0
        
        response_lower = response.lower()
        found = 0
        
        for keyword in expected_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in response_lower:
                found += 1
            else:
                # Check for partial/related terms
                words = keyword_lower.split()
                if any(w in response_lower for w in words if len(w) > 3):
                    found += 0.5
        
        return found / len(expected_keywords)
