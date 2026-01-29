"""Main Evaluator - Orchestrates the full evaluation pipeline."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from isaac_eval.config import EvalConfig, get_eval_config
from isaac_eval.dataset import EvaluationDataset, EvalQuery, QueryType, create_default_dataset
from isaac_eval.metrics import (
    RetrievalEvaluator,
    RetrievalResult,
    RetrievalMetrics,
    GenerationEvaluator,
    GenerationResult,
    GenerationMetrics,
)

logger = logging.getLogger(__name__)


def _load_image_registry_map() -> Dict[str, str]:
    """Load image_id -> project_name mapping from registry."""
    registry_path = Path("data/image_registry.json")
    mapping = {}
    
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for img in data.get("images", []):
                    image_id = img.get("image_id", "")
                    project_name = img.get("project_name", "")
                    if image_id and project_name:
                        mapping[image_id.lower()] = project_name.lower()
            logger.debug(f"Loaded {len(mapping)} image mappings from registry")
        except Exception as e:
            logger.warning(f"Failed to load image registry: {e}")
    
    return mapping


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    dataset_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    retrieval_metrics: Optional[RetrievalMetrics] = None
    generation_metrics: Optional[GenerationMetrics] = None
    
    # Error analysis
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "dataset_stats": self.dataset_stats,
            "retrieval_metrics": self.retrieval_metrics.to_dict() if self.retrieval_metrics else None,
            "generation_metrics": self.generation_metrics.to_dict() if self.generation_metrics else None,
            "error_analysis": self.error_analysis,
            "recommendations": self.recommendations,
        }
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {path}")
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# ISAAC Evaluation Report",
            f"\nGenerated: {self.timestamp}",
            "\n---\n",
        ]
        
        # Dataset stats - simplified
        lines.append("## Dataset")
        lines.append(f"- Total queries: {self.dataset_stats.get('total_queries', 0)}")
        
        if self.retrieval_metrics:
            lines.extend([
                "\n## Retrieval Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| MRR | {self.retrieval_metrics.mrr:.4f} |",
                f"| Hit Rate | {self.retrieval_metrics.hit_rate:.2%} |",
                f"| Image Hit Rate | {self.retrieval_metrics.image_hit_rate:.2%} |",
                f"| Avg Retrieval Time | {self.retrieval_metrics.avg_retrieval_time_ms:.0f}ms |",
            ])
            
            lines.append("\n### Recall@k")
            lines.append("| k | Recall |")
            lines.append("|---|--------|")
            for k, v in sorted(self.retrieval_metrics.recall_at_k.items()):
                lines.append(f"| {k} | {v:.4f} |")
        
        if self.generation_metrics:
            lines.extend([
                "\n## Generation Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Faithfulness | {self.generation_metrics.faithfulness_score:.2%} |",
                f"| Citation Correctness | {self.generation_metrics.citation_correctness:.2%} |",
                f"| Answer Relevance | {self.generation_metrics.answer_relevance:.2%} |",
                f"| Refusal Accuracy | {self.generation_metrics.refusal_accuracy:.2%} |",
                f"| Avg Generation Time | {self.generation_metrics.avg_generation_time_ms:.0f}ms |",
            ])
        
        if self.error_analysis and self.error_analysis.get("failure_modes"):
            lines.extend(["\n## Issues", ""])
            for mode, details in self.error_analysis["failure_modes"].items():
                lines.append(f"- {mode.replace('_', ' ').title()}: {details.get('count', 0)} cases")
        
        if self.recommendations:
            lines.extend(["\n## Recommendations", ""])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
        
        return "\n".join(lines)


class ISAACEvaluator:
    """Main evaluator orchestrating retrieval and generation evaluation."""
    
    def __init__(
        self,
        config: Optional[EvalConfig] = None,
        retriever=None,
        generator=None,
    ):
        self.config = config or get_eval_config()
        self._retriever = retriever
        self._generator = generator
        
        self._retrieval_evaluator = RetrievalEvaluator(k_values=self.config.k_values)
        self._generation_evaluator = GenerationEvaluator(
            faithfulness_threshold=self.config.faithfulness_threshold
        )
        
        logger.info("ISAACEvaluator initialized")
    
    def _get_retriever(self):
        """Lazy load retriever."""
        if self._retriever is None:
            from isaac_api.services.retriever import get_retriever_service
            self._retriever = get_retriever_service()
        return self._retriever
    
    def _get_generator(self):
        """Lazy load generator."""
        if self._generator is None:
            from isaac_generation.service import get_generation_service
            self._generator = get_generation_service()
        return self._generator
    
    async def run_evaluation(
        self,
        dataset: Optional[EvaluationDataset] = None,
        run_generation: bool = True,
    ) -> EvaluationReport:
        """Run full evaluation pipeline."""
        # Load or create dataset
        if dataset is None:
            if self.config.eval_data_path.exists():
                dataset = EvaluationDataset.load(self.config.eval_data_path)
            else:
                dataset = create_default_dataset()
                dataset.save(self.config.eval_data_path)
        
        logger.info(f"Running evaluation on {len(dataset)} queries")
        
        report = EvaluationReport(
            config={"k_values": self.config.k_values},
            dataset_stats=dataset.stats(),
        )
        
        # Build expected results mapping
        expected = self._build_expected_mapping(dataset)
        
        # Run retrieval evaluation
        retrieval_results = await self._run_retrieval(dataset)
        report.retrieval_metrics = self._retrieval_evaluator.evaluate(
            retrieval_results, expected
        )
        
        # Run generation evaluation
        if run_generation:
            generation_results = await self._run_generation(dataset, retrieval_results)
            report.generation_metrics = self._generation_evaluator.evaluate(
                generation_results, expected
            )
        
        # Error analysis
        report.error_analysis = self._analyze_errors(
            report.retrieval_metrics,
            report.generation_metrics,
            dataset,
        )
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        return report
    
    def _build_expected_mapping(
        self,
        dataset: EvaluationDataset
    ) -> Dict[str, Dict[str, Any]]:
        """Build mapping from query_id to expected results."""
        return {
            q.query_id: {
                "expected_doc_ids": q.expected_doc_ids,
                "expected_topics": q.expected_topics,
                "expected_images": q.expected_images,
                "expected_keywords": q.expected_keywords,
                "should_refuse": q.should_refuse,
                "should_say_idk": q.should_say_idk,
                "forbidden_topics": q.forbidden_topics,
            }
            for q in dataset.queries
        }
    
    def _parse_retrieval_response_object(
        self,
        response: Any,
        image_id_to_project: Dict[str, str]
    ) -> tuple[List[Dict], List[str]]:
        """Parse a RetrievalResponse object into chunks and images."""
        chunks = [
            {
                "content": c.content,
                "metadata": {
                    "doc_id": c.doc_id or "",
                    "chunk_id": c.chunk_id or "",
                    "source_uri": c.source_uri or "",
                    "h1": c.metadata.get("h1", "") if c.metadata else "",
                    "h2": c.metadata.get("h2", "") if c.metadata else "",
                    "project_name": c.metadata.get("project_name", "") if c.metadata else "",
                    "section": c.section or "",
                },
                "score": c.score,
            }
            for c in response.chunks
        ]
        raw_images = [img.image_id for img in response.images]
        images = [image_id_to_project.get(img.lower(), img) for img in raw_images]
        return chunks, images
    
    def _parse_retrieval_response_dict(
        self,
        response: Dict,
        image_id_to_project: Dict[str, str]
    ) -> tuple[List[Dict], List[str]]:
        """Parse a dict response into chunks and images."""
        chunks = response.get("chunks", [])
        raw_images = [img.get("image_id", "") for img in response.get("images", [])]
        images = [image_id_to_project.get(img.lower(), img) for img in raw_images]
        return chunks, images
    
    def _create_empty_retrieval_result(self, query_id: str) -> RetrievalResult:
        """Create an empty retrieval result for failed queries."""
        return RetrievalResult(
            query_id=query_id,
            retrieved_doc_ids=[],
            retrieved_chunks=[],
            retrieved_images=[],
            scores=[],
            retrieval_time_ms=0.0,
        )
    
    async def _run_retrieval(
        self,
        dataset: EvaluationDataset
    ) -> List[RetrievalResult]:
        """Run retrieval for all queries."""
        retriever = self._get_retriever()
        results = []
        
        # Load image registry for mapping image_id -> project_name
        image_id_to_project = _load_image_registry_map()
        
        for query in dataset.queries:
            # Skip off-topic queries for retrieval eval
            if query.query_type == QueryType.OFF_TOPIC:
                continue
            
            # Yield control to event loop periodically
            await asyncio.sleep(0)
            
            start_time = time.perf_counter()
            
            try:
                response = retriever.search(
                    query=query.query,
                    k=max(self.config.k_values),
                )
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                # Handle both dict and RetrievalResponse object
                if hasattr(response, 'chunks'):
                    chunks, images = self._parse_retrieval_response_object(response, image_id_to_project)
                else:
                    chunks, images = self._parse_retrieval_response_dict(response, image_id_to_project)
                
                result = RetrievalResult(
                    query_id=query.query_id,
                    retrieved_doc_ids=[c["metadata"].get("doc_id", "") for c in chunks],
                    retrieved_chunks=chunks,
                    retrieved_images=images,
                    scores=[c.get("score", 0.0) for c in chunks],
                    retrieval_time_ms=elapsed_ms,
                )
                results.append(result)
                
                logger.debug(f"Retrieved {len(chunks)} chunks for {query.query_id}")
                
            except Exception as e:
                logger.error(f"Retrieval failed for {query.query_id}: {e}")
                results.append(self._create_empty_retrieval_result(query.query_id))
        
        return results
    
    async def _run_generation(
        self,
        dataset: EvaluationDataset,
        retrieval_results: List[RetrievalResult],
    ) -> List[GenerationResult]:
        """Run generation for all queries."""
        generator = self._get_generator()
        results = []
        
        # Build retrieval lookup
        retrieval_lookup = {r.query_id: r for r in retrieval_results}
        
        for query in dataset.queries:
            start_time = time.perf_counter()
            
            try:
                # Use the generation service
                streaming_response = await generator.process_query(
                    query=query.query,
                    chat_history=[],
                )
                
                # Collect full response
                full_response = ""
                async for token in streaming_response.token_stream:
                    full_response += token
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                # Extract citations from response
                citations = self._extract_citations(full_response)
                
                # Get context from retrieval
                retrieval = retrieval_lookup.get(query.query_id)
                context = ""
                if retrieval:
                    context = "\n".join(
                        c.get("content", "") for c in retrieval.retrieved_chunks
                    )
                
                result = GenerationResult(
                    query_id=query.query_id,
                    query=query.query,
                    response=full_response,
                    citations=citations,
                    retrieved_context=context,
                    generation_time_ms=elapsed_ms,
                    refused=self._generation_evaluator._detected_refusal(full_response),
                    said_idk=self._generation_evaluator._detected_idk(full_response),
                )
                results.append(result)
                
                logger.debug(f"Generated {len(full_response)} chars for {query.query_id}")
                
            except Exception as e:
                logger.error(f"Generation failed for {query.query_id}: {e}")
                results.append(GenerationResult(
                    query_id=query.query_id,
                    query=query.query,
                    response=f"Error: {e}",
                    citations=[],
                    retrieved_context="",
                    generation_time_ms=0.0,
                ))
        
        return results
    
    def _extract_citations(self, response: str) -> List[str]:
        """Extract citations from response text."""
        citations = []
        
        # Pattern 1: [Source Name]
        bracket_pattern = r'\[([^\]]+)\]'
        matches = re.findall(bracket_pattern, response)
        citations.extend(
            match for match in matches
            if not match.isdigit() and len(match) > 2
        )
        
        # Pattern 2: References section
        ref_markers = ["**References:**", "Sources:"]
        for marker in ref_markers:
            if marker in response:
                ref_section = response.split(marker)[-1]
                ref_items = re.findall(r'[|\-]\s*([^|\n]+)', ref_section)
                citations.extend(item.strip() for item in ref_items if item.strip())
                break
        
        return list(set(citations))
    
    def _analyze_retrieval_errors(
        self,
        retrieval_metrics: RetrievalMetrics,
        dataset: EvaluationDataset,
    ) -> List[Dict[str, Any]]:
        """Analyze retrieval-specific errors."""
        low_score_queries = []
        recall_key = "recall@5"
        
        for query_id, results in retrieval_metrics.query_results.items():
            query = dataset.get_by_id(query_id)
            
            if results.get(recall_key, 1.0) < 0.3:
                low_score_queries.append({
                    "query_id": query_id,
                    "query": query.query if query else "",
                    recall_key: results.get(recall_key, 0),
                    "issue": "Low recall - relevant content not retrieved",
                })
            
            is_valid = query and not query.should_refuse and not query.should_say_idk
            if results.get("top_score", 1.0) < 0.35 and is_valid:
                low_score_queries.append({
                    "query_id": query_id,
                    "query": query.query if query else "",
                    "top_score": results.get("top_score", 0),
                    "issue": "Low relevance score - embedding mismatch",
                })
        
        return low_score_queries
    
    def _analyze_generation_errors(
        self,
        generation_metrics: GenerationMetrics,
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze generation-specific errors."""
        failure_modes = {}
        
        unfaithful = [
            qid for qid, res in generation_metrics.query_results.items()
            if res.get("faithfulness", 1.0) < 0.5
        ]
        if unfaithful:
            failure_modes["unfaithful_responses"] = {
                "count": len(unfaithful),
                "examples": unfaithful[:5],
                "description": "Responses not grounded in retrieved context",
            }
        
        bad_citations = [
            qid for qid, res in generation_metrics.query_results.items()
            if res.get("citation_correctness", 1.0) < 0.5
        ]
        if bad_citations:
            failure_modes["incorrect_citations"] = {
                "count": len(bad_citations),
                "examples": bad_citations[:5],
                "description": "Citations don't match retrieved content",
            }
        
        return failure_modes
    
    def _analyze_errors(
        self,
        retrieval_metrics: Optional[RetrievalMetrics],
        generation_metrics: Optional[GenerationMetrics],
        dataset: EvaluationDataset,
    ) -> Dict[str, Any]:
        """Analyze common failure modes."""
        analysis: Dict[str, Any] = {
            "failure_modes": {},
            "low_score_queries": [],
        }
        
        if retrieval_metrics:
            analysis["low_score_queries"] = self._analyze_retrieval_errors(
                retrieval_metrics, dataset
            )
        
        if generation_metrics:
            analysis["failure_modes"] = self._analyze_generation_errors(generation_metrics)
        
        return analysis
    
    def _get_retrieval_recommendations(self, rm: RetrievalMetrics) -> List[str]:
        """Generate retrieval-specific recommendations."""
        recs = []
        if rm.mrr < 0.5:
            recs.append(
                "Improve ranking quality: MRR is below 0.5. Consider tuning the "
                "reranker or adjusting hybrid search weights."
            )
        if rm.hit_rate < 0.8:
            recs.append(
                "Improve retrieval coverage: Hit rate is below 80%. Review chunking "
                "strategy and ensure key information is preserved."
            )
        if rm.image_hit_rate < 0.7:
            recs.append(
                "Improve image-text linkage: Ensure image metadata (captions, alt-text) "
                "is properly indexed and linked to documents."
            )
        if rm.recall_at_k.get(5, 0) < 0.6:
            recs.append(
                "Increase embedding quality: Recall@5 is low. Consider using "
                "domain-specific embeddings or fine-tuning."
            )
        return recs
    
    def _get_generation_recommendations(self, gm: GenerationMetrics) -> List[str]:
        """Generate generation-specific recommendations."""
        recs = []
        if gm.faithfulness_score < 0.7:
            recs.append(
                "Reduce hallucinations: Faithfulness score is below 70%. Emphasize "
                "grounding in retrieved context."
            )
        if gm.citation_correctness < 0.8:
            recs.append(
                "Improve citation accuracy: Consider explicitly listing source names "
                "in the context formatting."
            )
        if gm.refusal_accuracy < 1.0:
            recs.append(
                "Improve off-topic detection: Strengthen the off-topic detection "
                "logic or prompt instructions."
            )
        return recs
    
    def _generate_recommendations(self, report: EvaluationReport) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        if report.retrieval_metrics:
            recommendations.extend(self._get_retrieval_recommendations(report.retrieval_metrics))
        
        if report.generation_metrics:
            recommendations.extend(self._get_generation_recommendations(report.generation_metrics))
        
        return recommendations


# Import re for citation extraction
import re
