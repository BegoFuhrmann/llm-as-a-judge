"""
Enumeration types for the RAG vs Agentic AI Evaluation Framework.

This module defines all enumeration types used across the evaluation framework,
providing type safety and clear categorization of system types and evaluation dimensions.
"""

from enum import Enum


class SystemType(Enum):
    """Enumeration of AI system architectures being evaluated."""
    RAG = "rag"
    AGENTIC = "agentic"
    HYBRID = "hybrid"


class EvaluationDimension(Enum):
    """
    Academic evaluation dimensions based on established IR and NLP metrics.
    Each dimension follows peer-review standards with clear operational definitions.
    """
    # Core Information Quality (TREC-style metrics)
    FACTUAL_ACCURACY = "factual_accuracy"
    RELEVANCE = "relevance" 
    COMPLETENESS = "completeness"
    
    # Linguistic and Cognitive Quality
    CLARITY = "clarity"
    
    # Source Attribution and Trustworthiness
    CITATION_QUALITY = "citation_quality"
    
    # System-Specific Capabilities
    REASONING_DEPTH = "reasoning_depth"
    ADAPTABILITY = "adaptability"
    EFFICIENCY = "efficiency"
    
    # Regulatory Compliance Dimensions
    GDPR_COMPLIANCE = "gdpr_compliance"
    EU_AI_ACT_ALIGNMENT = "eu_ai_act_alignment"
    AUDIT_TRAIL_QUALITY = "audit_trail_quality"
