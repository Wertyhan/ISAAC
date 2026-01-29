"""Visualization Module - Generate charts and diagrams for evaluation results."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Install with: pip install matplotlib")

# Chart constants
TEXT_OFFSET_POINTS = "offset points"
LOC_UPPER_RIGHT = "upper right"
LOC_LOWER_RIGHT = "lower right"


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    output_dir: Path = Path("isaac_eval/results/charts")
    figsize: tuple = (10, 6)
    dpi: int = 150
    style: str = "seaborn-v0_8-whitegrid"  # Compatible with newer matplotlib
    color_palette: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            # Professional color palette
            self.color_palette = [
                "#2E86AB",  # Blue
                "#A23B72",  # Magenta
                "#F18F01",  # Orange
                "#C73E1D",  # Red
                "#3B1F2B",  # Dark
                "#44AF69",  # Green
                "#FCAB10",  # Yellow
                "#5E60CE",  # Purple
            ]
        self.output_dir = Path(self.output_dir)


class EvaluationVisualizer:
    """Generates visualizations for evaluation results."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self._ensure_output_dir()
        
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(self.config.style)
            except (OSError, ValueError):
                plt.style.use('default')
    
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_charts(
        self,
        report_path: Path,
        prefix: str = "eval"
    ) -> Dict[str, Path]:
        """Generate all charts from an evaluation report."""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib is required for visualization. Install with: pip install matplotlib")
            return {}
        
        # Load report
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        generated = {}
        
        # 1. Retrieval Metrics Overview
        if report.get("retrieval_metrics"):
            path = self.plot_retrieval_overview(
                report["retrieval_metrics"],
                prefix=prefix
            )
            generated["retrieval_overview"] = path
            
            # 2. Recall@k Chart
            path = self.plot_recall_at_k(
                report["retrieval_metrics"]["recall_at_k"],
                prefix=prefix
            )
            generated["recall_at_k"] = path
            
            # 3. Precision vs Recall
            path = self.plot_precision_recall(
                report["retrieval_metrics"],
                prefix=prefix
            )
            generated["precision_recall"] = path
        
        # 4. Generation Metrics Overview
        if report.get("generation_metrics"):
            path = self.plot_generation_overview(
                report["generation_metrics"],
                prefix=prefix
            )
            generated["generation_overview"] = path
        
        # 5. Combined Radar Chart
        if report.get("retrieval_metrics") and report.get("generation_metrics"):
            path = self.plot_radar_chart(
                report["retrieval_metrics"],
                report["generation_metrics"],
                prefix=prefix
            )
            generated["radar_chart"] = path
        
        # 6. Performance Timeline (if multiple results exist)
        results_dir = report_path.parent
        history = self._load_historical_results(results_dir)
        if len(history) > 1:
            path = self.plot_metrics_timeline(history, prefix=prefix)
            generated["metrics_timeline"] = path
        
        # 7. Error Analysis Pie Chart
        if report.get("error_analysis", {}).get("failure_modes"):
            path = self.plot_error_analysis(
                report["error_analysis"],
                prefix=prefix
            )
            generated["error_analysis"] = path
        
        # 8. Dataset Distribution
        if report.get("dataset_stats"):
            path = self.plot_dataset_distribution(
                report["dataset_stats"],
                prefix=prefix
            )
            generated["dataset_distribution"] = path
        
        logger.info(f"Generated {len(generated)} charts in {self.config.output_dir}")
        return generated
    
    def plot_retrieval_overview(
        self,
        metrics: Dict[str, Any],
        prefix: str = "eval"
    ) -> Path:
        """Create bar chart for retrieval metrics overview."""
        _, ax = plt.subplots(figsize=self.config.figsize)
        
        metrics_data = {
            "MRR": metrics.get("mrr", 0),
            "Hit Rate": metrics.get("hit_rate", 0),
            "Image Hit Rate": metrics.get("image_hit_rate", 0),
            "Recall@5": metrics.get("recall_at_k", {}).get("5", 0),
            "Precision@3": metrics.get("precision_at_k", {}).get("3", 0),
        }
        
        names = list(metrics_data.keys())
        values = list(metrics_data.values())
        colors = self.config.color_palette[:len(names)]
        
        bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=1.2)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{value:.2%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords=TEXT_OFFSET_POINTS,
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Retrieval Metrics Overview', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target (80%)')
        ax.legend(loc=LOC_UPPER_RIGHT)
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        output_path = self.config.output_dir / f"{prefix}_retrieval_overview.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_recall_at_k(
        self,
        recall_at_k: Dict[str, float],
        prefix: str = "eval"
    ) -> Path:
        """Create line chart for Recall@k."""
        _, ax = plt.subplots(figsize=self.config.figsize)
        
        # Sort by k value
        k_values = sorted([int(k) for k in recall_at_k.keys()])
        recalls = [recall_at_k[str(k)] for k in k_values]
        
        ax.plot(k_values, recalls, marker='o', markersize=10, linewidth=2.5,
                color=self.config.color_palette[0], label='Recall@k')
        
        # Fill area under curve
        ax.fill_between(k_values, recalls, alpha=0.3, color=self.config.color_palette[0])
        
        # Add value labels
        for k, recall in zip(k_values, recalls):
            ax.annotate(
                f'{recall:.3f}',
                xy=(k, recall),
                xytext=(0, 10),
                textcoords=TEXT_OFFSET_POINTS,
                ha='center', fontsize=10, fontweight='bold'
            )
        
        ax.set_xlabel('k (Top-k Retrieved)', fontsize=12)
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title('Recall@k - Retrieval Coverage', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(k_values)
        
        # Reference lines
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (90%)')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (70%)')
        ax.legend(loc=LOC_LOWER_RIGHT)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        output_path = self.config.output_dir / f"{prefix}_recall_at_k.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_precision_recall(
        self,
        metrics: Dict[str, Any],
        prefix: str = "eval"
    ) -> Path:
        """Create precision-recall curve."""
        _, ax = plt.subplots(figsize=self.config.figsize)
        
        recall_at_k = metrics.get("recall_at_k", {})
        precision_at_k = metrics.get("precision_at_k", {})
        
        k_values = sorted([int(k) for k in recall_at_k.keys()])
        
        recalls = [recall_at_k[str(k)] for k in k_values]
        precisions = [precision_at_k.get(str(k), 0) for k in k_values]
        
        # Plot both curves
        ax.plot(k_values, recalls, marker='o', markersize=8, linewidth=2,
                color=self.config.color_palette[0], label='Recall@k')
        ax.plot(k_values, precisions, marker='s', markersize=8, linewidth=2,
                color=self.config.color_palette[1], label='Precision@k')
        
        ax.set_xlabel('k (Top-k Retrieved)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision vs Recall at Different k', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(k_values)
        ax.legend(loc=LOC_UPPER_RIGHT)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        output_path = self.config.output_dir / f"{prefix}_precision_recall.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_generation_overview(
        self,
        metrics: Dict[str, Any],
        prefix: str = "eval"
    ) -> Path:
        """Create horizontal bar chart for generation metrics."""
        _, ax = plt.subplots(figsize=self.config.figsize)
        
        metrics_data = {
            "Faithfulness": metrics.get("faithfulness_score", 0),
            "Citation Correctness": metrics.get("citation_correctness", 0),
            "Answer Relevance": metrics.get("answer_relevance", 0),
            "Refusal Accuracy": metrics.get("refusal_accuracy", 0),
            "IDK Accuracy": metrics.get("idk_accuracy", 0),
        }
        
        names = list(metrics_data.keys())
        values = list(metrics_data.values())
        colors = [
            self.config.color_palette[5] if v >= 0.8 else
            self.config.color_palette[2] if v >= 0.6 else
            self.config.color_palette[3]
            for v in values
        ]
        
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=1.2)
        
        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.annotate(
                f'{value:.1%}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords=TEXT_OFFSET_POINTS,
                ha='left', va='center',
                fontsize=11, fontweight='bold'
            )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Generation Metrics Overview', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.2)
        
        # Add threshold lines
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=0.6, color='orange', linestyle='--', alpha=0.5)
        
        # Legend for thresholds
        green_patch = mpatches.Patch(color='green', alpha=0.3, label='Good (>=80%)')
        orange_patch = mpatches.Patch(color='orange', alpha=0.3, label='Acceptable (>=60%)')
        red_patch = mpatches.Patch(color='red', alpha=0.3, label='Needs Improvement (<60%)')
        ax.legend(handles=[green_patch, orange_patch, red_patch], loc=LOC_LOWER_RIGHT)
        
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        output_path = self.config.output_dir / f"{prefix}_generation_overview.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_radar_chart(
        self,
        retrieval_metrics: Dict[str, Any],
        generation_metrics: Dict[str, Any],
        prefix: str = "eval"
    ) -> Path:
        """Create radar/spider chart for overall system performance."""
        _, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
        
        # Define metrics for radar
        categories = [
            'MRR',
            'Hit Rate',
            'Image Hit Rate',
            'Recall@5',
            'Faithfulness',
            'Citation\nCorrectness',
            'Answer\nRelevance',
            'Refusal\nAccuracy'
        ]
        
        values = [
            retrieval_metrics.get("mrr", 0),
            retrieval_metrics.get("hit_rate", 0),
            retrieval_metrics.get("image_hit_rate", 0),
            retrieval_metrics.get("recall_at_k", {}).get("5", 0),
            generation_metrics.get("faithfulness_score", 0),
            generation_metrics.get("citation_correctness", 0),
            generation_metrics.get("answer_relevance", 0),
            generation_metrics.get("refusal_accuracy", 0),
        ]
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        values += values[:1]  # Complete the loop
        
        # Draw the chart
        ax.plot(angles, values, 'o-', linewidth=2, color=self.config.color_palette[0])
        ax.fill(angles, values, alpha=0.25, color=self.config.color_palette[0])
        
        # Add reference circle at 0.8
        reference_values = [0.8] * (N + 1)
        ax.plot(angles, reference_values, '--', linewidth=1.5, color='green', alpha=0.5, label='Target (80%)')
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        
        # Set y-axis
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
        
        ax.set_title('ISAAC System Performance Overview', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc=LOC_UPPER_RIGHT, bbox_to_anchor=(1.2, 1.1))
        
        plt.tight_layout()
        
        output_path = self.config.output_dir / f"{prefix}_radar_chart.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_error_analysis(
        self,
        error_analysis: Dict[str, Any],
        prefix: str = "eval"
    ) -> Path:
        """Create pie chart for error analysis."""
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Failure modes pie chart
        failure_modes = error_analysis.get("failure_modes", {})
        
        if failure_modes:
            labels = list(failure_modes.keys())
            sizes = [mode.get("count", 0) for mode in failure_modes.values()]
            colors = self.config.color_palette[:len(labels)]
            
            explode = [0.05] * len(labels)
            
            ax1.pie(
                sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90,
                textprops={'fontsize': 10}
            )
            ax1.set_title('Failure Mode Distribution', fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No failure modes detected!', ha='center', va='center',
                    fontsize=14, color='green')
            ax1.set_title('Failure Modes', fontsize=12, fontweight='bold')
        
        # Low score queries bar chart
        low_score_queries = error_analysis.get("low_score_queries", [])
        
        if low_score_queries:
            query_ids = [q.get("query_id", "Unknown")[:10] for q in low_score_queries[:8]]
            issues = [q.get("issue", "Unknown")[:25] for q in low_score_queries[:8]]
            
            y_pos = np.arange(len(query_ids))
            ax2.barh(y_pos, [1] * len(query_ids), color=self.config.color_palette[3], alpha=0.7)
            
            for i, (qid, issue) in enumerate(zip(query_ids, issues)):
                ax2.text(0.05, i, f"{qid}: {issue}", va='center', fontsize=9)
            
            ax2.set_yticks([])
            ax2.set_xlim(0, 1)
            ax2.set_xlabel('')
            ax2.set_title('Problematic Queries', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No low-score queries!', ha='center', va='center',
                    fontsize=14, color='green')
            ax2.set_title('Low-Score Queries', fontsize=12, fontweight='bold')
        
        plt.suptitle('Error Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.config.output_dir / f"{prefix}_error_analysis.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_dataset_distribution(
        self,
        dataset_stats: Dict[str, Any],
        prefix: str = "eval"
    ) -> Path:
        """Create charts showing dataset distribution."""
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Query types distribution
        by_type = dataset_stats.get("by_type", {})
        if by_type:
            labels = list(by_type.keys())
            sizes = list(by_type.values())
            colors = self.config.color_palette[:len(labels)]
            
            axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                       startangle=90, textprops={'fontsize': 9})
            axes[0].set_title('Query Types', fontsize=11, fontweight='bold')
        
        # 2. Difficulty distribution
        by_difficulty = dataset_stats.get("by_difficulty", {})
        if by_difficulty:
            labels = list(by_difficulty.keys())
            sizes = list(by_difficulty.values())
            colors = [self.config.color_palette[5], self.config.color_palette[2], self.config.color_palette[3]]
            
            axes[1].pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%',
                       startangle=90, textprops={'fontsize': 9})
            axes[1].set_title('Difficulty Distribution', fontsize=11, fontweight='bold')
        
        # 3. Dataset overview stats
        overview_data = {
            'Total Queries': dataset_stats.get('total_queries', 0),
            'With Expected Docs': dataset_stats.get('with_expected_docs', 0),
            'With Expected Images': dataset_stats.get('with_expected_images', 0),
            'Off-Topic': dataset_stats.get('off_topic_queries', 0),
        }
        
        labels = list(overview_data.keys())
        values = list(overview_data.values())
        
        bars = axes[2].bar(labels, values, color=self.config.color_palette[:len(labels)])
        
        for bar, value in zip(bars, values):
            axes[2].annotate(f'{value}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords=TEXT_OFFSET_POINTS, ha='center', fontsize=10, fontweight='bold')
        
        axes[2].set_title('Dataset Statistics', fontsize=11, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Evaluation Dataset Overview', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.config.output_dir / f"{prefix}_dataset_distribution.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_metrics_timeline(
        self,
        history: List[Dict[str, Any]],
        prefix: str = "eval"
    ) -> Path:
        """Plot metrics over multiple evaluation runs."""
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        timestamps = [h.get("timestamp", "")[:16] for h in history]
        x = range(len(timestamps))
        
        # Retrieval metrics timeline (handle None values)
        mrr_values = [(h.get("retrieval_metrics") or {}).get("mrr", 0) for h in history]
        hit_rate_values = [(h.get("retrieval_metrics") or {}).get("hit_rate", 0) for h in history]
        
        ax1.plot(x, mrr_values, marker='o', linewidth=2, label='MRR', color=self.config.color_palette[0])
        ax1.plot(x, hit_rate_values, marker='s', linewidth=2, label='Hit Rate', color=self.config.color_palette[1])
        
        ax1.set_xlabel('Evaluation Run')
        ax1.set_ylabel('Score')
        ax1.set_title('Retrieval Metrics Over Time', fontsize=12, fontweight='bold')
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(timestamps, rotation=45, ha='right')
        ax1.set_ylim(0, 1.1)
        ax1.legend(loc=LOC_LOWER_RIGHT)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Generation metrics timeline (handle None values)
        faithfulness = [(h.get("generation_metrics") or {}).get("faithfulness_score", 0) for h in history]
        citation = [(h.get("generation_metrics") or {}).get("citation_correctness", 0) for h in history]
        
        ax2.plot(x, faithfulness, marker='o', linewidth=2, label='Faithfulness', color=self.config.color_palette[2])
        ax2.plot(x, citation, marker='s', linewidth=2, label='Citation Correctness', color=self.config.color_palette[3])
        
        ax2.set_xlabel('Evaluation Run')
        ax2.set_ylabel('Score')
        ax2.set_title('Generation Metrics Over Time', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(timestamps, rotation=45, ha='right')
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc=LOC_LOWER_RIGHT)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        output_path = self.config.output_dir / f"{prefix}_metrics_timeline.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _load_historical_results(self, results_dir: Path) -> List[Dict[str, Any]]:
        """Load all historical evaluation results."""
        history = []
        
        for json_file in sorted(results_dir.glob("eval_*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    history.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        return history
    
    def generate_html_report(
        self,
        report_path: Path,
        charts: Dict[str, Path],
        prefix: str = "eval"
    ) -> Path:
        """Generate an HTML report with embedded charts."""
        # Load report data
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        html_content = self._build_html_report(report, charts)
        
        output_path = self.config.output_dir / f"{prefix}_report.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
        return output_path
    
    def _build_html_report(
        self,
        report: Dict[str, Any],
        charts: Dict[str, Path]
    ) -> str:
        """Build HTML content for the report."""
        retrieval = report.get("retrieval_metrics", {})
        generation = report.get("generation_metrics", {})
        errors = report.get("error_analysis", {})
        recommendations = report.get("recommendations", [])
        
        # Convert chart paths to relative paths
        chart_imgs = {
            name: path.name for name, path in charts.items()
        }
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISAAC Evaluation Report</title>
    <style>
        :root {{
            --primary-color: #2E86AB;
            --success-color: #44AF69;
            --warning-color: #F18F01;
            --danger-color: #C73E1D;
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-color: #333;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary-color), #1a5276);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .timestamp {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .metric-card h3 {{
            color: var(--primary-color);
            margin-bottom: 15px;
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 10px;
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .metric-row:last-child {{
            border-bottom: none;
        }}
        
        .metric-value {{
            font-weight: bold;
        }}
        
        .metric-value.good {{
            color: var(--success-color);
        }}
        
        .metric-value.warning {{
            color: var(--warning-color);
        }}
        
        .metric-value.bad {{
            color: var(--danger-color);
        }}
        
        .chart-section {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .chart-section h2 {{
            color: var(--primary-color);
            margin-bottom: 20px;
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
        }}
        
        .chart-item {{
            text-align: center;
        }}
        
        .chart-item img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .chart-item h4 {{
            margin-top: 10px;
            color: #666;
        }}
        
        .recommendations {{
            background: linear-gradient(135deg, #fff3e0, #fff8e1);
            border-left: 4px solid var(--warning-color);
            padding: 20px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 30px;
        }}
        
        .recommendations h2 {{
            color: var(--warning-color);
            margin-bottom: 15px;
        }}
        
        .recommendations ul {{
            list-style-position: inside;
        }}
        
        .recommendations li {{
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }}
        
        .error-section {{
            background: #ffebee;
            border-left: 4px solid var(--danger-color);
            padding: 20px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 30px;
        }}
        
        .error-section h2 {{
            color: var(--danger-color);
            margin-bottom: 15px;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ISAAC Evaluation Report</h1>
            <p class="timestamp">Generated: {report.get('timestamp', 'Unknown')}</p>
        </header>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Retrieval Metrics</h3>
                <div class="metric-row">
                    <span>MRR</span>
                    <span class="metric-value {self._get_score_class(retrieval.get('mrr', 0))}">{retrieval.get('mrr', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span>Hit Rate</span>
                    <span class="metric-value {self._get_score_class(retrieval.get('hit_rate', 0))}">{retrieval.get('hit_rate', 0):.2%}</span>
                </div>
                <div class="metric-row">
                    <span>Image Hit Rate</span>
                    <span class="metric-value {self._get_score_class(retrieval.get('image_hit_rate', 0))}">{retrieval.get('image_hit_rate', 0):.2%}</span>
                </div>
                <div class="metric-row">
                    <span>Avg Retrieval Time</span>
                    <span class="metric-value">{retrieval.get('avg_retrieval_time_ms', 0):.0f}ms</span>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Generation Metrics</h3>
                <div class="metric-row">
                    <span>Faithfulness</span>
                    <span class="metric-value {self._get_score_class(generation.get('faithfulness_score', 0))}">{generation.get('faithfulness_score', 0):.2%}</span>
                </div>
                <div class="metric-row">
                    <span>Citation Correctness</span>
                    <span class="metric-value {self._get_score_class(generation.get('citation_correctness', 0))}">{generation.get('citation_correctness', 0):.2%}</span>
                </div>
                <div class="metric-row">
                    <span>Answer Relevance</span>
                    <span class="metric-value {self._get_score_class(generation.get('answer_relevance', 0))}">{generation.get('answer_relevance', 0):.2%}</span>
                </div>
                <div class="metric-row">
                    <span>Refusal Accuracy</span>
                    <span class="metric-value {self._get_score_class(generation.get('refusal_accuracy', 0))}">{generation.get('refusal_accuracy', 0):.2%}</span>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Recall@k</h3>
                {''.join([f'<div class="metric-row"><span>@{k}</span><span class="metric-value">{v:.4f}</span></div>' for k, v in sorted(retrieval.get('recall_at_k', {}).items(), key=lambda x: int(x[0]))])}
            </div>
        </div>
        
        {self._build_charts_section(chart_imgs)}
        
        {self._build_errors_section(errors)}
        
        {self._build_recommendations_section(recommendations)}
        
        <footer>
            <p>ISAAC Evaluation System</p>
        </footer>
    </div>
</body>
</html>"""
        
        return html
    
    def _build_charts_section(self, chart_imgs: Dict[str, str]) -> str:
        """Build charts section HTML."""
        if not chart_imgs:
            return ""
        
        # Select only key charts to display
        key_charts = ['retrieval_overview', 'generation_overview', 'recall_at_k', 'radar_chart']
        filtered = {k: v for k, v in chart_imgs.items() if k in key_charts}
        
        if not filtered:
            filtered = dict(list(chart_imgs.items())[:4])
        
        chart_html = ''.join([
            f'<div class="chart-item"><img src="{path}" alt="{name}"><h4>{name.replace("_", " ").title()}</h4></div>'
            for name, path in filtered.items()
        ])
        
        return f'''<div class="chart-section">
            <h2>Charts</h2>
            <div class="chart-grid">{chart_html}</div>
        </div>'''
    
    def _build_errors_section(self, errors: Dict[str, Any]) -> str:
        """Build error analysis section HTML."""
        if not errors.get('failure_modes') and not errors.get('low_score_queries'):
            return ""
        
        return f'<div class="error-section"><h2>Error Analysis</h2>{self._format_errors_html(errors)}</div>'
    
    def _build_recommendations_section(self, recommendations: List[str]) -> str:
        """Build recommendations section HTML."""
        if not recommendations:
            return ""
        
        items = ''.join([f'<li>{rec}</li>' for rec in recommendations[:5]])
        return f'<div class="recommendations"><h2>Recommendations</h2><ul>{items}</ul></div>'
    
    def _get_score_class(self, score: float) -> str:
        """Return CSS class based on score."""
        if score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "warning"
        return "bad"
    
    def _format_errors_html(self, errors: Dict[str, Any]) -> str:
        """Format error analysis as HTML."""
        html_parts = []
        
        failure_modes = errors.get("failure_modes", {})
        if failure_modes:
            html_parts.append("<h3>Failure Modes</h3><ul>")
            for mode, details in failure_modes.items():
                count = details.get('count', 0)
                desc = details.get('description', '')
                html_parts.append(f"<li><strong>{mode.replace('_', ' ').title()}</strong>: {count} cases")
                if desc:
                    html_parts.append(f" - {desc}")
                html_parts.append("</li>")
            html_parts.append("</ul>")
        
        low_score = errors.get("low_score_queries", [])
        if low_score:
            html_parts.append("<h3>Problematic Queries</h3><ul>")
            for q in low_score[:5]:
                qid = q.get('query_id', '')
                issue = q.get('issue', '')
                html_parts.append(f"<li><code>{qid}</code>: {issue}</li>")
            html_parts.append("</ul>")
        
        return "".join(html_parts)


def generate_evaluation_charts(report_path: Path) -> Dict[str, Path]:
    """Convenience function to generate all charts from a report file."""
    visualizer = EvaluationVisualizer()
    prefix = report_path.stem
    
    charts = visualizer.generate_all_charts(report_path, prefix=prefix)
    
    if charts:
        visualizer.generate_html_report(report_path, charts, prefix=prefix)
    
    return charts
