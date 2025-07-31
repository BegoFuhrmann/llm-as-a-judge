"""
Data models for the RAG vs Agentic AI Evaluation Framework.

This module contains all dataclass definitions used throughout the evaluation
framework, providing structured data containers with type hints.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from .enums import SystemType, EvaluationDimension


@dataclass
class SourceSnippet:
    """
    Represents a source snippet retrieved by RAG systems.
    
    Based on academic standards for source attribution and grounding
    evaluation in retrieval-augmented generation systems.
    """
    content: str
    snippet_id: Optional[str] = None
    source_document: Optional[str] = None
    relevance_score: Optional[float] = None
    position_in_source: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationRubric:
    """
    Academic rubric defining evaluation criteria with inter-rater reliability considerations.
    Based on Krippendorff's alpha and Cohen's kappa standards.
    """
    dimension: EvaluationDimension
    description: str
    criteria_5: str  # Excellent (90-100%)
    criteria_4: str  # Good (80-89%)
    criteria_3: str  # Satisfactory (70-79%)
    criteria_2: str  # Needs Improvement (60-69%)
    criteria_1: str  # Inadequate (<60%)
    weight: float = 1.0


@dataclass
class AnswerEvaluation:
    """Enhanced evaluation result with statistical measures and metadata."""
    answer: str
    system_type: Optional[SystemType]
    scores: Dict[EvaluationDimension, int]
    justifications: Dict[EvaluationDimension, str]
    confidence_scores: Dict[EvaluationDimension, float]
    weighted_total: float
    raw_total: float
    evaluation_metadata: Dict[str, Any]
    overall_assessment: str
    comparative_notes: str
    # Enhanced for RAG evaluation with snippets
    source_snippets: Optional[List[SourceSnippet]] = None
    snippet_grounding_score: Optional[float] = None
    citation_snippet_alignment: Optional[Dict[str, float]] = None


@dataclass
class EvaluationInput:
    """
    Input data structure for evaluation with support for RAG snippets.
    Supports both Excel and programmatic input formats.
    """
    query: str
    answer: str
    system_type: Optional[SystemType] = None
    source_snippets: Optional[List[SourceSnippet]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class ComparisonResult:
    """Statistical comparison between RAG and Agentic systems."""
    rag_evaluations: List[AnswerEvaluation]
    agentic_evaluations: List[AnswerEvaluation]
    statistical_significance: Dict[EvaluationDimension, Dict[str, float]]
    effect_sizes: Dict[EvaluationDimension, float]
    overall_winner: Optional[SystemType]
    confidence_interval: Tuple[float, float]
    recommendations: List[str]
