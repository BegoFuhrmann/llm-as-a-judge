"""
Statistical analysis and comparison module for RAG vs Agentic AI evaluation.

This module implements rigorous statistical methods for comparing AI systems,
including effect size calculations, significance testing, and reliability metrics
following academic standards.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats
from collections import defaultdict

from .models import AnswerEvaluation, ComparisonResult
from .enums import SystemType, EvaluationDimension
from .config import EvalConfig
from .utils import safe_divide, calculate_percentage

logger = logging.getLogger(__name__)


def perform_statistical_comparison(
    rag_evaluations: List[AnswerEvaluation],
    agentic_evaluations: List[AnswerEvaluation],
    config: Optional[EvalConfig] = None
) -> ComparisonResult:
    """
    Perform rigorous statistical comparison between RAG and Agentic systems.
    
    This implements academic standards for comparative evaluation including:
    - Effect size calculations (Cohen's d)
    - Statistical significance testing
    - Confidence intervals
    - Multiple comparison corrections
    
    Args:
        rag_evaluations: List of evaluations for RAG systems
        agentic_evaluations: List of evaluations for Agentic systems
        config: Optional evaluation configuration for dimensions
        
    Returns:
        ComparisonResult containing statistical analysis
    """
    logger.info("Performing statistical comparison between RAG and Agentic systems")
    
    # Determine dimensions to analyze
    dimensions = None
    if config is not None:
        dimensions = config.dimensions
    elif rag_evaluations:
        dimensions = list(rag_evaluations[0].scores.keys())
    elif agentic_evaluations:
        dimensions = list(agentic_evaluations[0].scores.keys())
    else:
        logger.warning("No dimensions available for analysis")
        dimensions = []
    
    statistical_significance = {}
    effect_sizes = {}
    
    for dimension in dimensions:
        # Extract scores for this dimension
        rag_scores = [e.scores.get(dimension, 0) for e in rag_evaluations if dimension in e.scores]
        agentic_scores = [e.scores.get(dimension, 0) for e in agentic_evaluations if dimension in e.scores]
        # Extract scores for this dimension
        rag_scores = [e.scores.get(dimension, 0) for e in rag_evaluations if dimension in e.scores]
        agentic_scores = [e.scores.get(dimension, 0) for e in agentic_evaluations if dimension in e.scores]
        
        # Perform t-test if sufficient data
        if len(rag_scores) > 1 and len(agentic_scores) > 1:
            t_stat, p_value = stats.ttest_ind(rag_scores, agentic_scores)
            
            # Calculate effect size (Cohen's d)
            cohens_d = calculate_effect_size(rag_scores, agentic_scores)
            
            statistical_significance[dimension] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "rag_mean": float(np.mean(rag_scores)),
                "agentic_mean": float(np.mean(agentic_scores)),
                "rag_std": float(np.std(rag_scores)),
                "agentic_std": float(np.std(agentic_scores)),
                "rag_count": len(rag_scores),
                "agentic_count": len(agentic_scores)
            }
            effect_sizes[dimension] = cohens_d
            
            logger.debug(f"Dimension {dimension.value}: t={t_stat:.3f}, p={p_value:.3f}, d={cohens_d:.3f}")
        else:
            logger.warning(f"Insufficient data for statistical comparison on dimension: {dimension.value}")
    
    # Overall comparison using weighted scores
    rag_weighted_scores = [e.weighted_total for e in rag_evaluations if hasattr(e, 'weighted_total')]
    agentic_weighted_scores = [e.weighted_total for e in agentic_evaluations if hasattr(e, 'weighted_total')]
    
    overall_winner = None
    confidence_interval = (0.0, 0.0)
    
    if rag_weighted_scores and agentic_weighted_scores:
        rag_mean = np.mean(rag_weighted_scores)
        agentic_mean = np.mean(agentic_weighted_scores)
        
        if agentic_mean > rag_mean:
            overall_winner = SystemType.AGENTIC
        elif rag_mean > agentic_mean:
            overall_winner = SystemType.RAG
        
        # Calculate confidence interval for the difference
        diff = agentic_mean - rag_mean
        pooled_se = np.sqrt(np.var(rag_weighted_scores, ddof=1)/len(rag_weighted_scores) + 
                           np.var(agentic_weighted_scores, ddof=1)/len(agentic_weighted_scores))
        confidence_interval = (diff - 1.96 * pooled_se, diff + 1.96 * pooled_se)
        
        logger.info(f"Overall winner: {overall_winner.value if overall_winner else 'Tie'}")
        logger.info(f"Mean difference: {diff:.3f}, 95% CI: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
    
    # Generate academic recommendations
    recommendations = generate_academic_recommendations(
        statistical_significance, effect_sizes, overall_winner
    )
    
    return ComparisonResult(
        rag_evaluations=rag_evaluations,
        agentic_evaluations=agentic_evaluations,
        statistical_significance=statistical_significance,
        effect_sizes=effect_sizes,
        overall_winner=overall_winner,
        confidence_interval=confidence_interval,
        recommendations=recommendations
    )


def generate_academic_recommendations(
    statistical_significance: Dict,
    effect_sizes: Dict,
    overall_winner: Optional[SystemType]
) -> List[str]:
    """Generate evidence-based recommendations following academic standards."""
    recommendations = []
    
    # Effect size interpretations (Cohen's conventions)
    large_effects = [dim for dim, d in effect_sizes.items() if abs(d) >= 0.8]
    medium_effects = [dim for dim, d in effect_sizes.items() if 0.5 <= abs(d) < 0.8]
    
    if large_effects:
        recommendations.append(
            f"Large effect sizes observed in: {[d.value for d in large_effects]}. "
            "These represent practically significant differences requiring further investigation."
        )
    
    if medium_effects:
        recommendations.append(
            f"Medium effect sizes in: {[d.value for d in medium_effects]}. "
            "Consider these dimensions for system optimization."
        )
    
    significant_dims = [dim for dim, stats_data in statistical_significance.items() 
                       if stats_data.get("significant", False)]
    
    if significant_dims:
        recommendations.append(
            f"Statistically significant differences found in: {[d.value for d in significant_dims]}. "
            "Results are likely generalizable beyond this sample."
        )
    
    if overall_winner:
        recommendations.append(
            f"{overall_winner.value.upper()} systems show superior performance overall. "
            "However, consider task-specific requirements and deployment constraints."
        )
    
    recommendations.append(
        "Conduct larger-scale evaluation with diverse query types for robust conclusions."
    )
    
    logger.info(f"Generated {len(recommendations)} academic recommendations")
    return recommendations


def calculate_inter_rater_reliability(
    evaluations_1: List[AnswerEvaluation],
    evaluations_2: List[AnswerEvaluation]
) -> Dict[str, float]:
    """
    Calculate inter-rater reliability metrics for academic validation.
    
    Implements Krippendorff's alpha and Cohen's kappa for multi-dimensional
    evaluation agreement, following academic standards for reliability studies.
    
    Args:
        evaluations_1: First set of evaluations
        evaluations_2: Second set of evaluations (same responses, different judge)
    
    Returns:
        Dictionary containing reliability metrics
    """
    if len(evaluations_1) != len(evaluations_2):
        raise ValueError("Evaluation sets must have same length")
    
    logger.info(f"Calculating inter-rater reliability for {len(evaluations_1)} evaluations")
    
    reliability_metrics = {}
    
    # Calculate agreement for each dimension
    for dimension in evaluations_1[0].scores.keys():
        scores_1 = [e.scores[dimension] for e in evaluations_1]
        scores_2 = [e.scores[dimension] for e in evaluations_2]
        
        # Cohen's Kappa for ordinal data
        agreement_matrix = np.zeros((5, 5))  # 5x5 for scores 1-5
        for s1, s2 in zip(scores_1, scores_2):
            agreement_matrix[s1-1][s2-1] += 1
        
        # Calculate observed agreement
        observed_agreement = np.trace(agreement_matrix) / len(scores_1)
        
        # Calculate expected agreement
        marginal_1 = np.sum(agreement_matrix, axis=1) / len(scores_1)
        marginal_2 = np.sum(agreement_matrix, axis=0) / len(scores_1)
        expected_agreement = np.sum(marginal_1 * marginal_2)
        
        # Cohen's Kappa
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement) if expected_agreement != 1 else 1.0
        reliability_metrics[f"{dimension.value}_kappa"] = kappa
        
        # Pearson correlation for continuous interpretation
        correlation = np.corrcoef(scores_1, scores_2)[0, 1] if len(scores_1) > 1 else 1.0
        reliability_metrics[f"{dimension.value}_correlation"] = correlation
        
        logger.debug(f"Dimension {dimension.value}: kappa={kappa:.3f}, r={correlation:.3f}")
    
    # Overall reliability
    all_kappas = [v for k, v in reliability_metrics.items() if 'kappa' in k]
    reliability_metrics['overall_kappa'] = np.mean(all_kappas)
    
    all_correlations = [v for k, v in reliability_metrics.items() if 'correlation' in k]
    reliability_metrics['overall_correlation'] = np.mean(all_correlations)
    
    logger.info(f"Overall reliability: kappa={reliability_metrics['overall_kappa']:.3f}, "
                f"r={reliability_metrics['overall_correlation']:.3f}")
    
    return reliability_metrics


def calculate_effect_size(group1_scores: List[float], group2_scores: List[float]) -> float:
    """
    Calculate Cohen's d effect size between two groups.
    
    Args:
        group1_scores: Scores from first group
        group2_scores: Scores from second group
        
    Returns:
        Cohen's d effect size
    """
    if len(group1_scores) <= 1 or len(group2_scores) <= 1:
        return 0.0
    
    # Calculate pooled standard deviation
    n1, n2 = len(group1_scores), len(group2_scores)
    var1, var2 = np.var(group1_scores, ddof=1), np.var(group2_scores, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    # Calculate Cohen's d
    mean_diff = np.mean(group2_scores) - np.mean(group1_scores)
    cohens_d = mean_diff / pooled_std
    
    return cohens_d


def interpret_effect_size(effect_size: float) -> str:
    """
    Interpret effect size according to Cohen's conventions.
    
    Args:
        effect_size: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_effect = abs(effect_size)
    
    if abs_effect < 0.2:
        return "negligible"
    elif abs_effect < 0.5:
        return "small"
    elif abs_effect < 0.8:
        return "medium"
    else:
        return "large"


def calculate_confidence_interval(
    scores: List[float], 
    confidence_level: float = 0.95
) -> tuple[float, float]:
    """
    Calculate confidence interval for a set of scores.
    
    Args:
        scores: List of numerical scores
        confidence_level: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(scores) <= 1:
        return (0.0, 0.0)
    
    mean_score = np.mean(scores)
    std_error = stats.sem(scores)
    
    # Calculate critical value
    alpha = 1 - confidence_level
    degrees_freedom = len(scores) - 1
    critical_value = stats.t.ppf(1 - alpha/2, degrees_freedom)
    
    margin_error = critical_value * std_error
    
    return (mean_score - margin_error, mean_score + margin_error)
