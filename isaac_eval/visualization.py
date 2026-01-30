"""Visualization Module - Generate HTML reports for evaluation results."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Constants to avoid duplicate literals
OFFSET_POINTS_TEXTCOORDS = "offset points"

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    output_dir: Path = Path("isaac_eval/results/charts")
    figsize: tuple = (10, 6)
    dpi: int = 150
    color_palette: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44AF69"]
        self.output_dir = Path(self.output_dir)


class EvaluationVisualizer:
    """Generates visualizations for evaluation results."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use("seaborn-v0_8-whitegrid")
            except (OSError, ValueError):
                plt.style.use('default')
    
    def generate_all_charts(self, report_path: Path, prefix: str = "eval") -> Dict[str, Path]:
        """Generate all charts from an evaluation report."""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib required for visualization")
            return {}
        
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        generated = {}
        
        if report.get("retrieval_metrics"):
            generated["retrieval_overview"] = self._plot_retrieval(report["retrieval_metrics"], prefix)
            generated["recall_at_k"] = self._plot_recall_at_k(report["retrieval_metrics"]["recall_at_k"], prefix)
        
        if report.get("generation_metrics"):
            generated["generation_overview"] = self._plot_generation(report["generation_metrics"], prefix)
        
        if report.get("retrieval_metrics") and report.get("generation_metrics"):
            generated["radar_chart"] = self._plot_radar(report["retrieval_metrics"], report["generation_metrics"], prefix)
        
        return generated
    
    def _plot_retrieval(self, metrics: Dict, prefix: str) -> Path:
        _, ax = plt.subplots(figsize=self.config.figsize)
        
        data = {
            "MRR": metrics.get("mrr", 0),
            "Hit Rate": metrics.get("hit_rate", 0),
            "Image Hit Rate": metrics.get("image_hit_rate", 0),
            "Recall@5": metrics.get("recall_at_k", {}).get("5", 0),
        }
        
        bars = ax.bar(data.keys(), data.values(), color=self.config.color_palette[:len(data)])
        for bar, v in zip(bars, data.values()):
            ax.annotate(f'{v:.2%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords=OFFSET_POINTS_TEXTCOORDS, ha='center', fontweight='bold')
        
        ax.set_ylabel('Score')
        ax.set_title('Retrieval Metrics', fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target (80%)')
        ax.legend()
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        path = self.config.output_dir / f"{prefix}_retrieval_overview.png"
        plt.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return path
    
    def _plot_recall_at_k(self, recall_at_k: Dict, prefix: str) -> Path:
        _, ax = plt.subplots(figsize=self.config.figsize)
        
        k_values = sorted([int(k) for k in recall_at_k.keys()])
        recalls = [recall_at_k[str(k)] for k in k_values]
        
        ax.plot(k_values, recalls, marker='o', markersize=10, linewidth=2.5, color=self.config.color_palette[0])
        ax.fill_between(k_values, recalls, alpha=0.3, color=self.config.color_palette[0])
        
        for k, r in zip(k_values, recalls):
            ax.annotate(f'{r:.3f}', xy=(k, r), xytext=(0, 10), textcoords=OFFSET_POINTS_TEXTCOORDS, ha='center', fontweight='bold')
        
        ax.set_xlabel('k')
        ax.set_ylabel('Recall')
        ax.set_title('Recall@k', fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(k_values)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        path = self.config.output_dir / f"{prefix}_recall_at_k.png"
        plt.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return path
    
    def _plot_generation(self, metrics: Dict, prefix: str) -> Path:
        _, ax = plt.subplots(figsize=self.config.figsize)
        
        data = {
            "Faithfulness": metrics.get("faithfulness_score", 0),
            "Citation": metrics.get("citation_correctness", 0),
            "Relevance": metrics.get("answer_relevance", 0),
            "Refusal": metrics.get("refusal_accuracy", 0),
        }
        
        colors = [self.config.color_palette[5] if v >= 0.8 else self.config.color_palette[2] if v >= 0.6 
                  else self.config.color_palette[3] for v in data.values()]
        
        y_pos = np.arange(len(data))
        bars = ax.barh(y_pos, list(data.values()), color=colors)
        
        for bar, v in zip(bars, data.values()):
            ax.annotate(f'{v:.1%}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                       xytext=(5, 0), textcoords=OFFSET_POINTS_TEXTCOORDS, ha='left', va='center', fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data.keys())
        ax.set_xlabel('Score')
        ax.set_title('Generation Metrics', fontweight='bold')
        ax.set_xlim(0, 1.2)
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        path = self.config.output_dir / f"{prefix}_generation_overview.png"
        plt.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return path
    
    def _plot_radar(self, ret: Dict, gen: Dict, prefix: str) -> Path:
        _, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
        
        categories = ['MRR', 'Hit Rate', 'Image Hit', 'Recall@5', 'Faithfulness', 'Citation', 'Relevance', 'Refusal']
        values = [
            ret.get("mrr", 0), ret.get("hit_rate", 0), ret.get("image_hit_rate", 0),
            ret.get("recall_at_k", {}).get("5", 0), gen.get("faithfulness_score", 0),
            gen.get("citation_correctness", 0), gen.get("answer_relevance", 0), gen.get("refusal_accuracy", 0),
        ]
        
        N = len(categories)
        angles = [n / N * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=self.config.color_palette[0])
        ax.fill(angles, values, alpha=0.25, color=self.config.color_palette[0])
        ax.plot(angles, [0.8] * (N + 1), '--', linewidth=1.5, color='green', alpha=0.5)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('System Performance', fontweight='bold', pad=20)
        
        plt.tight_layout()
        path = self.config.output_dir / f"{prefix}_radar_chart.png"
        plt.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        return path
    
    def generate_html_report(self, report_path: Path, charts: Dict[str, Path], prefix: str = "eval") -> Path:
        """Generate an HTML report."""
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        html = self._build_html(report, {n: p.name for n, p in charts.items()})
        
        output_path = self.config.output_dir / f"{prefix}_report.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        return output_path
    
    def _build_html(self, report: Dict, charts: Dict[str, str]) -> str:
        ret = report.get("retrieval_metrics", {})
        gen = report.get("generation_metrics", {})
        recs = report.get("recommendations", [])
        
        def score_class(v):
            if v >= 0.8:
                return "good"
            if v >= 0.6:
                return "warning"
            return "bad"
        
        recall_rows = "".join([
            f'<div class="row"><span>@{k}</span><span class="val">{v:.4f}</span></div>'
            for k, v in sorted(ret.get('recall_at_k', {}).items(), key=lambda x: int(x[0]))
        ])
        
        chart_html = "".join([
            f'<div class="chart"><img src="{p}" alt="{n}"><p>{n.replace("_", " ").title()}</p></div>'
            for n, p in charts.items()
        ])
        
        recs_html = "".join([f'<li>{r}</li>' for r in recs[:5]]) if recs else ""
        
        return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><title>ISAAC Evaluation Report</title>
<style>
:root {{--primary:#2E86AB;--good:#44AF69;--warning:#F18F01;--bad:#C73E1D;}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Segoe UI',sans-serif;background:#f8f9fa;color:#333;padding:20px;line-height:1.6;}}
.container{{max-width:1400px;margin:0 auto;}}
header{{background:linear-gradient(135deg,var(--primary),#1a5276);color:#fff;padding:30px;border-radius:10px;margin-bottom:30px;text-align:center;}}
header h1{{font-size:2.2em;margin-bottom:8px;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-bottom:30px;}}
.card{{background:#fff;border-radius:10px;padding:20px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}}
.card h3{{color:var(--primary);margin-bottom:15px;border-bottom:2px solid var(--primary);padding-bottom:10px;}}
.row{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #eee;}}
.row:last-child{{border:none;}}
.val{{font-weight:bold;}}
.val.good{{color:var(--good);}} .val.warning{{color:var(--warning);}} .val.bad{{color:var(--bad);}}
.charts{{background:#fff;border-radius:10px;padding:20px;margin-bottom:30px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}}
.charts h2{{color:var(--primary);margin-bottom:20px;}}
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px;}}
.chart img{{max-width:100%;border-radius:8px;}}
.chart p{{text-align:center;margin-top:8px;color:#666;}}
.recs{{background:#fff3e0;border-left:4px solid var(--warning);padding:20px;border-radius:0 10px 10px 0;margin-bottom:30px;}}
.recs h2{{color:var(--warning);margin-bottom:15px;}}
.recs li{{margin-bottom:10px;padding:10px;background:#fff;border-radius:5px;}}
footer{{text-align:center;padding:20px;color:#666;}}
</style></head>
<body><div class="container">
<header><h1>ISAAC Evaluation Report</h1><p>{report.get('timestamp', '')}</p></header>
<div class="grid">
<div class="card"><h3>Retrieval</h3>
<div class="row"><span>MRR</span><span class="val {score_class(ret.get('mrr',0))}">{ret.get('mrr',0):.4f}</span></div>
<div class="row"><span>Hit Rate</span><span class="val {score_class(ret.get('hit_rate',0))}">{ret.get('hit_rate',0):.2%}</span></div>
<div class="row"><span>Image Hit Rate</span><span class="val {score_class(ret.get('image_hit_rate',0))}">{ret.get('image_hit_rate',0):.2%}</span></div>
<div class="row"><span>Avg Time</span><span class="val">{ret.get('avg_retrieval_time_ms',0):.0f}ms</span></div>
</div>
<div class="card"><h3>Generation</h3>
<div class="row"><span>Faithfulness</span><span class="val {score_class(gen.get('faithfulness_score',0))}">{gen.get('faithfulness_score',0):.2%}</span></div>
<div class="row"><span>Citation</span><span class="val {score_class(gen.get('citation_correctness',0))}">{gen.get('citation_correctness',0):.2%}</span></div>
<div class="row"><span>Relevance</span><span class="val {score_class(gen.get('answer_relevance',0))}">{gen.get('answer_relevance',0):.2%}</span></div>
<div class="row"><span>Refusal</span><span class="val {score_class(gen.get('refusal_accuracy',0))}">{gen.get('refusal_accuracy',0):.2%}</span></div>
</div>
<div class="card"><h3>Recall@k</h3>{recall_rows}</div>
</div>
<div class="charts"><h2>Charts</h2><div class="chart-grid">{chart_html}</div></div>
{"<div class='recs'><h2>Recommendations</h2><ul>" + recs_html + "</ul></div>" if recs_html else ""}
<footer><p>ISAAC Evaluation System</p></footer>
</div></body></html>"""


def generate_evaluation_charts(report_path: Path) -> Dict[str, Path]:
    """Generate all charts from a report file."""
    visualizer = EvaluationVisualizer()
    prefix = report_path.stem
    charts = visualizer.generate_all_charts(report_path, prefix=prefix)
    if charts:
        visualizer.generate_html_report(report_path, charts, prefix=prefix)
    return charts
