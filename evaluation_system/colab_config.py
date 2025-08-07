"""
Enhanced configuration for Google Colab/Discovery Engine RAG evaluation.

This module provides specialized configuration for evaluating RAG systems
that may not include explicit citations by default, such as Google's 
Discovery Engine implementations.
"""

import logging
from typing import Dict
from .enums import EvaluationDimension
from .models import EvaluationRubric

# Configure logger for this module
logger = logging.getLogger(__name__)


class ColabRAGConfig:
    """
    Configuration specifically designed for evaluating RAG systems
    that focus on content accuracy rather than explicit citations.
    """
    
    def __init__(self):
        self.dimensions = list(EvaluationDimension)
        self.rubrics = self._create_colab_rubrics()
        
    def _create_colab_rubrics(self) -> Dict[EvaluationDimension, EvaluationRubric]:
        """
        Create evaluation rubrics adapted for Google Colab/Discovery Engine style RAG systems.
        
        Returns:
            Dictionary of evaluation rubrics with adjusted citation criteria
        """
        return {
            EvaluationDimension.FACTUAL_ACCURACY: EvaluationRubric(
                dimension=EvaluationDimension.FACTUAL_ACCURACY,
                description="Accuracy of factual claims and information presented in the response",
                criteria_5="All factual claims are accurate and verifiable against source snippets",
                criteria_4="Most factual claims are accurate with minor inaccuracies",
                criteria_3="Generally accurate with some factual errors that don't affect main message",
                criteria_2="Several factual errors that impact reliability",
                criteria_1="Multiple significant factual errors or misinformation",
                weight=1.2
            ),
            
            EvaluationDimension.RELEVANCE: EvaluationRubric(
                dimension=EvaluationDimension.RELEVANCE,
                description="How well the response addresses the specific query and user intent",
                criteria_5="Directly and comprehensively addresses all aspects of the query",
                criteria_4="Addresses most query aspects with minor gaps",
                criteria_3="Generally relevant but misses some important aspects",
                criteria_2="Partially relevant with significant gaps or tangential content",
                criteria_1="Largely irrelevant or off-topic response",
                weight=1.1
            ),
            
            EvaluationDimension.COMPLETENESS: EvaluationRubric(
                dimension=EvaluationDimension.COMPLETENESS,
                description="Thoroughness and comprehensiveness of the response",
                criteria_5="Comprehensive response covering all relevant aspects from snippets",
                criteria_4="Covers most relevant aspects with minor omissions",
                criteria_3="Adequate coverage of main points",
                criteria_2="Missing several important aspects or details",
                criteria_1="Incomplete or superficial treatment of the topic",
                weight=1.0
            ),
            
            EvaluationDimension.CLARITY: EvaluationRubric(
                dimension=EvaluationDimension.CLARITY,
                description="Clarity, readability, and ease of understanding",
                criteria_5="Exceptionally clear, well-structured, and easy to understand",
                criteria_4="Clear and well-organized with good flow",
                criteria_3="Generally clear with minor organizational issues",
                criteria_2="Somewhat unclear or poorly organized",
                criteria_1="Confusing, poorly written, or hard to follow",
                weight=0.9
            ),
            
            EvaluationDimension.COHERENCE: EvaluationRubric(
                dimension=EvaluationDimension.COHERENCE,
                description="Logical flow and consistency of the response",
                criteria_5="Perfectly coherent with excellent logical flow and consistency",
                criteria_4="Coherent and logical with minor inconsistencies",
                criteria_3="Generally coherent with some logical gaps",
                criteria_2="Some incoherence or contradictory statements",
                criteria_1="Incoherent or contradictory response",
                weight=0.8
            ),
            
            EvaluationDimension.CITATION_QUALITY: EvaluationRubric(
                dimension=EvaluationDimension.CITATION_QUALITY,
                description="Quality of source attribution and grounding in provided snippets (adapted for systems without explicit citations)",
                criteria_5="Content clearly demonstrates strong grounding in snippets; if citations present, they are accurate and complete",
                criteria_4="Good implicit grounding in source material; clear content derivation from snippets",
                criteria_3="Adequate grounding with content reasonably derived from provided sources",
                criteria_2="Weak grounding; some content appears unrelated to snippets",
                criteria_1="Poor or no grounding; content contradicts or ignores provided snippets",
                weight=1.0  # Adjusted weight to focus on grounding rather than citation format
            ),
            
            EvaluationDimension.PROVENANCE: EvaluationRubric(
                dimension=EvaluationDimension.PROVENANCE,
                description="Ability to trace information back to source documents and transparency about source reliability",
                criteria_5="Information clearly traceable to specific snippets; high transparency about sources",
                criteria_4="Most information traceable with good source transparency",
                criteria_3="Adequate source traceability and transparency",
                criteria_2="Limited traceability or transparency about sources",
                criteria_1="Poor or no source traceability; unclear information origin",
                weight=0.9
            ),
            
            EvaluationDimension.REASONING_DEPTH: EvaluationRubric(
                dimension=EvaluationDimension.REASONING_DEPTH,
                description="Depth of analysis and reasoning demonstrated in the response",
                criteria_5="Demonstrates deep analysis and sophisticated reasoning",
                criteria_4="Good analytical depth with clear reasoning",
                criteria_3="Adequate reasoning and analysis",
                criteria_2="Shallow reasoning with limited analysis",
                criteria_1="Minimal or superficial reasoning",
                weight=0.8
            ),
            
            EvaluationDimension.ADAPTABILITY: EvaluationRubric(
                dimension=EvaluationDimension.ADAPTABILITY,
                description="Ability to synthesize information from multiple sources and adapt to query context",
                criteria_5="Excellent synthesis across sources with perfect contextual adaptation",
                criteria_4="Good multi-source synthesis and contextual awareness",
                criteria_3="Adequate synthesis and context adaptation",
                criteria_2="Limited synthesis or poor context adaptation",
                criteria_1="Poor synthesis and contextual understanding",
                weight=0.9
            ),
            
            EvaluationDimension.EFFICIENCY: EvaluationRubric(
                dimension=EvaluationDimension.EFFICIENCY,
                description="Conciseness and efficiency of communication while maintaining completeness",
                criteria_5="Perfectly concise yet comprehensive; optimal information density",
                criteria_4="Well-balanced conciseness and completeness",
                criteria_3="Adequate balance between brevity and detail",
                criteria_2="Either too verbose or too brief for the context",
                criteria_1="Poor information density; inappropriate length",
                weight=0.7
            )
        }
    
    def get_citation_guidance(self) -> str:
        """
        Provide specific guidance for evaluating citation quality in systems
        without explicit citation mechanisms.
        
        Returns:
            Guidance text for evaluators
        """
        return """
        CITATION EVALUATION GUIDANCE FOR NON-CITING SYSTEMS:
        
        When evaluating systems that do not provide explicit citations:
        
        1. FOCUS ON CONTENT GROUNDING:
           - Does the answer content align with information in the provided snippets?
           - Are factual claims supported by the source material?
           - Is the information synthesis accurate and faithful to sources?
        
        2. IMPLICIT SOURCE USAGE:
           - Can you trace answer content back to specific snippets?
           - Does the response demonstrate understanding of source context?
           - Are there clear patterns of information derivation?
        
        3. DO NOT PENALIZE FOR ARCHITECTURAL CHOICES:
           - Do not reduce scores simply because explicit citations are absent
           - Focus on the quality of information use rather than citation format
           - Evaluate based on the system's intended design capabilities
        
        4. SCORING GUIDELINES:
           - Score 5: Strong implicit grounding, accurate content derivation
           - Score 4: Good content alignment with clear source usage
           - Score 3: Adequate grounding with reasonable source fidelity
           - Score 2: Weak grounding, some content misalignment
           - Score 1: Poor grounding, content contradicts or ignores sources
        """


def create_colab_evaluation_config():
    """
    Factory function to create a properly configured evaluation setup
    for Google Colab/Discovery Engine RAG systems.
    
    Returns:
        ColabRAGConfig instance
    """
    config = ColabRAGConfig()
    logger.info("Created Colab RAG evaluation configuration")
    logger.info(f"Configured {len(config.dimensions)} evaluation dimensions")
    logger.info("Special handling enabled for citation-free systems")
    
    return config
