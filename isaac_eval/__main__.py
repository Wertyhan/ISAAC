"""CLI Entry Point for ISAAC Evaluation."""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Union

from isaac_eval.config import get_eval_config
from isaac_eval.dataset import create_default_dataset, EvaluationDataset
from isaac_eval.evaluator import ISAACEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Check for visualization dependencies
try:
    from isaac_eval.visualization import EvaluationVisualizer, generate_evaluation_charts
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.debug("Visualization not available. Install matplotlib for charts.")


def _write_file(path: Union[str, Path], content: str) -> None:
    """Write content to file synchronously (for use with asyncio.to_thread)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ISAAC System Evaluation - Evaluate retrieval and generation quality"
    )
    
    parser.add_argument(
        "--create-dataset",
        action="store_true",
        help="Create and save the default evaluation dataset",
    )
    
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to evaluation dataset JSON file",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for evaluation report",
    )
    
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Skip generation evaluation (retrieval only)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output report in markdown format",
    )
    
    parser.add_argument(
        "--charts",
        action="store_true",
        help="Generate visualization charts (requires matplotlib)",
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report with charts",
    )
    
    parser.add_argument(
        "--visualize",
        type=Path,
        default=None,
        help="Generate charts from existing evaluation report JSON file",
    )
    
    return parser.parse_args()


def _handle_visualize(args) -> int:
    """Handle --visualize flag to generate charts from existing report."""
    if not VISUALIZATION_AVAILABLE:
        print("Error: Visualization requires matplotlib. Install with: pip install matplotlib")
        return 1
    
    if not args.visualize.exists():
        print(f"Error: Report file not found: {args.visualize}")
        return 1
    
    print(f"\nGenerating visualizations from {args.visualize}")
    charts = generate_evaluation_charts(args.visualize)
    
    if charts:
        print(f"Generated {len(charts)} charts:")
        for name, path in charts.items():
            print(f"  - {name}: {path}")
        print(f"\nHTML Report: {args.visualize.parent / 'charts' / f'{args.visualize.stem}_report.html'}")
    else:
        print("Error: No charts generated. Check if matplotlib is installed.")
    return 0


def _handle_create_dataset(args, config) -> int:
    """Handle --create-dataset flag."""
    dataset = create_default_dataset()
    output_path = args.dataset or config.eval_data_path
    dataset.save(output_path)
    
    print(f"\nCreated evaluation dataset with {len(dataset)} queries")
    print(f"Saved to: {output_path}")
    print("\nStatistics:")
    stats = dataset.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


def _load_or_create_dataset(args, config) -> EvaluationDataset:
    """Load existing dataset or create default one."""
    if args.dataset and args.dataset.exists():
        return EvaluationDataset.load(args.dataset)
    if config.eval_data_path.exists():
        return EvaluationDataset.load(config.eval_data_path)
    
    print("No dataset found. Creating default dataset...")
    dataset = create_default_dataset()
    dataset.save(config.eval_data_path)
    return dataset


def _print_retrieval_metrics(rm) -> None:
    """Print retrieval metrics."""
    print("\nRetrieval Metrics:")
    print(f"  MRR:              {rm.mrr:.4f}")
    print(f"  Hit Rate:         {rm.hit_rate:.2%}")
    print(f"  Image Hit Rate:   {rm.image_hit_rate:.2%}")
    print(f"  Avg Time:         {rm.avg_retrieval_time_ms:.0f}ms")
    print("\n  Recall@k:")
    recall_items = sorted(rm.recall_at_k.items())
    for k, v in recall_items:
        print(f"    @{k}: {v:.4f}")


def _print_generation_metrics(gm) -> None:
    """Print generation metrics."""
    print("\nGeneration Metrics:")
    print(f"  Faithfulness:     {gm.faithfulness_score:.2%}")
    print(f"  Citation Correct: {gm.citation_correctness:.2%}")
    print(f"  Answer Relevance: {gm.answer_relevance:.2%}")
    print(f"  Refusal Accuracy: {gm.refusal_accuracy:.2%}")
    print(f"  Avg Time:         {gm.avg_generation_time_ms:.0f}ms")


def _print_error_analysis(error_analysis: dict) -> None:
    """Print failure modes from error analysis."""
    if error_analysis.get("failure_modes"):
        print("\nFailure Modes:")
        for mode, details in error_analysis["failure_modes"].items():
            print(f"  - {mode}: {details['count']} cases")


def _print_recommendations(recommendations: list) -> None:
    """Print recommendations with truncation."""
    if recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            rec_short = rec[:100] + "..." if len(rec) > 100 else rec
            print(f"  {i}. {rec_short}")


def _generate_charts(args, output_path: Path) -> None:
    """Generate charts if requested."""
    if not (args.charts or args.html):
        return
    
    if VISUALIZATION_AVAILABLE:
        print("\nGenerating visualizations...")
        charts = generate_evaluation_charts(output_path)
        
        if charts:
            print(f"Generated {len(charts)} charts in isaac_eval/results/charts/")
            
            if args.html:
                html_path = output_path.parent / "charts" / f"{output_path.stem}_report.html"
                print(f"HTML Report: {html_path}")
    else:
        print("\nNote: Charts require matplotlib. Install with: pip install matplotlib")


async def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = get_eval_config()
    
    # Generate charts from existing report
    if args.visualize:
        return _handle_visualize(args)
    
    # Create dataset only
    if args.create_dataset:
        return _handle_create_dataset(args, config)
    
    # Load dataset
    dataset = _load_or_create_dataset(args, config)
    
    print("\nISAAC System Evaluation")
    print(f"Dataset: {len(dataset)} queries")
    gen_status = "Enabled" if not args.no_generation else "Disabled"
    print(f"Generation: {gen_status}")
    print("-" * 50)
    
    # Run evaluation
    evaluator = ISAACEvaluator(config=config)
    
    try:
        report = await evaluator.run_evaluation(
            dataset=dataset,
            run_generation=not args.no_generation,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1
    
    # Output results
    print("\n" + "-"*50)
    print("EVALUATION RESULTS")
    print("-"*50)
    
    if report.retrieval_metrics:
        _print_retrieval_metrics(report.retrieval_metrics)
    
    if report.generation_metrics:
        _print_generation_metrics(report.generation_metrics)
    
    _print_error_analysis(report.error_analysis)
    _print_recommendations(report.recommendations)
    
    # Save report
    output_path = args.output or config.results_dir / f"eval_{report.timestamp.replace(':', '-')}.json"
    report.save(output_path)
    
    if args.markdown:
        md_path = output_path.with_suffix(".md")
        md_content = report.to_markdown()
        await asyncio.to_thread(_write_file, md_path, md_content)
        print(f"\nMarkdown report: {md_path}")
    
    _generate_charts(args, output_path)
    
    print(f"\nReport saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
