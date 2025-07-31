"""
Report generation module for RAG vs Agentic AI evaluation.

This module generates academic-quality reports suitable for research publication,
including statistical summaries, visualizations, and properly formatted citations.
"""

import logging
import statistics
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import AnswerEvaluation, ComparisonResult
from .enums import SystemType, EvaluationDimension
from .config import EvalConfig
from .utils import truncate_text, format_duration, calculate_percentage

logger = logging.getLogger(__name__)


def generate_academic_citation_report(
    evaluations: List[AnswerEvaluation],
    comparison: Optional[ComparisonResult] = None,
    config: Optional[EvalConfig] = None
) -> str:
    """
    Generate publication-ready results section with proper academic formatting.
    
    Args:
        evaluations: List of answer evaluations
        comparison: Optional comparison results between systems
        config: Optional evaluation configuration
        
    Returns:
        Formatted academic report suitable for research paper inclusion
    """
    logger.info(f"Generating academic report for {len(evaluations)} evaluations")
    
    n_responses = len(evaluations)
    if n_responses == 0:
        return "No evaluations available for report generation."
    
    mean_score = statistics.mean([e.weighted_total for e in evaluations])
    std_score = statistics.stdev([e.weighted_total for e in evaluations]) if n_responses > 1 else 0
    min_score = min(e.weighted_total for e in evaluations)
    max_score = max(e.weighted_total for e in evaluations)
    
    report = f"""
## Results

### Evaluation Methodology
We evaluated {n_responses} AI system responses using a comprehensive academic framework 
implementing seven evaluation dimensions based on established information retrieval 
and natural language processing evaluation standards (Voorhees & Harman, 2005; 
Papineni et al., 2002).

### Performance Metrics
Overall weighted performance scores ranged from {min_score:.2f} 
to {max_score:.2f} (M = {mean_score:.2f}, SD = {std_score:.2f}) 
on a 5-point scale.
"""

    # Add dimensional analysis
    if evaluations and evaluations[0].scores:
        report += "\n### Dimensional Performance Analysis\n"
        
        # Calculate mean scores for each dimension
        dimension_stats = {}
        sample_scores = evaluations[0].scores
        
        for dimension in sample_scores.keys():
            scores = [e.scores.get(dimension, 0) for e in evaluations if dimension in e.scores]
            if scores:
                dimension_stats[dimension] = {
                    'mean': statistics.mean(scores),
                    'std': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
        
        for dimension, stats in dimension_stats.items():
            dimension_name = dimension.value.replace('_', ' ').title() if hasattr(dimension, 'value') else str(dimension).replace('_', ' ').title()
            report += f"- **{dimension_name}**: "
            report += f"M = {stats['mean']:.2f}, SD = {stats['std']:.2f}, "
            report += f"Range = [{stats['min']:.1f}, {stats['max']:.1f}] (n = {stats['count']})\\n"

    # Add comparison section if available
    if comparison and comparison.overall_winner:
        report += "\n### Inter-System Comparison\n"
        report += f"""
{comparison.overall_winner.value.title()} systems demonstrated superior performance 
(95% CI: [{comparison.confidence_interval[0]:.3f}, {comparison.confidence_interval[1]:.3f}])

Significant differences were observed in:
"""
        significant_dims = [dim for dim, stats in comparison.statistical_significance.items() 
                          if stats.get("significant", False)]
        
        for dim in significant_dims:
            stats = comparison.statistical_significance[dim]
            effect_size = comparison.effect_sizes[dim]
            report += f"- **{dim.value.replace('_', ' ').title()}**: "
            report += f"t = {stats['t_statistic']:.3f}, "
            report += f"p = {stats['p_value']:.3f}, d = {effect_size:.3f}\\n"
        
        # Add recommendations
        if comparison.recommendations:
            report += "\n### Recommendations\n"
            for i, rec in enumerate(comparison.recommendations, 1):
                report += f"{i}. {rec}\\n"
    
    report += """
### Implications
These findings contribute to the growing body of literature on AI system evaluation 
and provide empirical evidence for comparative analysis of different architectural 
approaches in artificial intelligence.

## References
Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic 
evaluation of machine translation. In Proceedings of ACL (pp. 311-318).

Voorhees, E. M., & Harman, D. K. (2005). TREC: Experiment and evaluation in 
information retrieval. MIT Press.
"""
    
    logger.info("Academic report generated successfully")
    return report


def generate_summary_report(
    evaluations: List[AnswerEvaluation],
    comparison: Optional[ComparisonResult] = None
) -> str:
    """
    Generate a concise summary report for quick analysis.
    
    Args:
        evaluations: List of answer evaluations
        comparison: Optional comparison results
        
    Returns:
        Formatted summary report
    """
    if not evaluations:
        return "No evaluations available."
    
    logger.info(f"Generating summary report for {len(evaluations)} evaluations")
    
    # Basic statistics
    scores = [e.weighted_total for e in evaluations]
    mean_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0
    
    report = f"""
# Evaluation Summary Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Evaluations**: {len(evaluations)}
- **Mean Score**: {mean_score:.2f} ± {std_score:.2f}
- **Median Score**: {median_score:.2f}
- **Score Range**: [{min(scores):.2f}, {max(scores):.2f}]

## Top Performing Responses
"""
    
    # Sort evaluations by score and show top 3
    sorted_evals = sorted(evaluations, key=lambda e: e.weighted_total, reverse=True)
    for i, eval_result in enumerate(sorted_evals[:3], 1):
        report += f"{i}. Score: {eval_result.weighted_total:.2f}\\n"
    
    # System comparison if available
    if comparison:
        report += f"\n## System Comparison\n"
        if comparison.overall_winner:
            report += f"- **Winner**: {comparison.overall_winner.value.title()}\\n"
        
        report += f"- **RAG Evaluations**: {len(comparison.rag_evaluations)}\\n"
        report += f"- **Agentic Evaluations**: {len(comparison.agentic_evaluations)}\\n"
        
        if comparison.rag_evaluations:
            rag_mean = statistics.mean([e.weighted_total for e in comparison.rag_evaluations])
            report += f"- **RAG Mean Score**: {rag_mean:.2f}\\n"
        
        if comparison.agentic_evaluations:
            agentic_mean = statistics.mean([e.weighted_total for e in comparison.agentic_evaluations])
            report += f"- **Agentic Mean Score**: {agentic_mean:.2f}\\n"
    
    logger.info("Summary report generated successfully")
    return report


def generate_detailed_evaluation_report(evaluation: AnswerEvaluation) -> str:
    """
    Generate a detailed report for a single evaluation.
    
    Args:
        evaluation: Single answer evaluation
        
    Returns:
        Detailed evaluation report
    """
    report = f"""
# Detailed Evaluation Report

## Answer
{evaluation.answer[:200]}{'...' if len(evaluation.answer) > 200 else ''}

## Scores
"""
    
    for dimension, score in evaluation.scores.items():
        report += f"- **{dimension.value.replace('_', ' ').title()}**: {score}/5\\n"
    
    report += f"\n**Weighted Total**: {evaluation.weighted_total:.2f}/5.0\\n"
    
    report += "\n## Justifications\n"
    for dimension, justification in evaluation.justifications.items():
        report += f"### {dimension.value.replace('_', ' ').title()}\\n"
        report += f"{justification}\\n\\n"
    
    return report


def generate_csv_export(evaluations: List[AnswerEvaluation]) -> str:
    """
    Generate CSV format data for external analysis.
    
    Args:
        evaluations: List of answer evaluations
        
    Returns:
        CSV formatted string
    """
    if not evaluations:
        return "No data available for export"
    
    logger.info(f"Generating CSV export for {len(evaluations)} evaluations")
    
    # Header
    dimensions = list(evaluations[0].scores.keys())
    header = "answer,weighted_total," + ",".join([dim.value for dim in dimensions])
    
    # Data rows
    rows = [header]
    for eval_result in evaluations:
        answer_text = eval_result.answer.replace('"', '""').replace('\\n', ' ')[:100]
        scores = [str(eval_result.scores[dim]) for dim in dimensions]
        row = f'"{answer_text}",{eval_result.weighted_total:.2f},' + ",".join(scores)
        rows.append(row)
    
    return "\\n".join(rows)


def generate_json_export(evaluations: List[AnswerEvaluation]) -> Dict[str, Any]:
    """
    Generate JSON format data for external analysis.
    
    Args:
        evaluations: List of answer evaluations
        
    Returns:
        Dictionary suitable for JSON serialization
    """
    logger.info(f"Generating JSON export for {len(evaluations)} evaluations")
    
    export_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(evaluations),
            "evaluation_framework": "RAG vs Agentic AI Academic Evaluation"
        },
        "evaluations": []
    }
    
    for i, eval_result in enumerate(evaluations):
        eval_data = {
            "id": i + 1,
            "answer": eval_result.answer,
            "weighted_total": eval_result.weighted_total,
            "scores": {dim.value: score for dim, score in eval_result.scores.items()},
            "justifications": {dim.value: just for dim, just in eval_result.justifications.items()}
        }
        export_data["evaluations"].append(eval_data)
    
    return export_data


def generate_latex_table(evaluations: List[AnswerEvaluation], caption: str = "") -> str:
    """
    Generate LaTeX table for academic publication.
    
    Args:
        evaluations: List of answer evaluations
        caption: Table caption
        
    Returns:
        LaTeX table code
    """
    if not evaluations:
        return "% No data available for table generation"
    
    logger.info(f"Generating LaTeX table for {len(evaluations)} evaluations")
    
    dimensions = list(evaluations[0].scores.keys())
    
    # Table header
    latex = "\\\\begin{table}[htbp]\\n"
    latex += "\\\\centering\\n"
    latex += "\\\\begin{tabular}{|c|" + "c|" * len(dimensions) + "c|}\\n"
    latex += "\\\\hline\\n"
    
    # Column headers
    headers = ["System"] + [dim.value.replace('_', ' ').title() for dim in dimensions] + ["Weighted Total"]
    latex += " & ".join(headers) + " \\\\\\\\\\n"
    latex += "\\\\hline\\n"
    
    # Data rows
    for i, eval_result in enumerate(evaluations, 1):
        scores = [f"{eval_result.scores[dim]:.1f}" for dim in dimensions]
        row_data = [f"System {i}"] + scores + [f"{eval_result.weighted_total:.2f}"]
        latex += " & ".join(row_data) + " \\\\\\\\\\n"
    
    latex += "\\\\hline\\n"
    latex += "\\\\end{tabular}\\n"
    
    if caption:
        latex += f"\\\\caption{{{caption}}}\\n"
    
    latex += "\\\\end{table}\\n"
    
    return latex
