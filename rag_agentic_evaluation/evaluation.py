"""
Core evaluation logic for the RAG vs Agentic AI Evaluation Framework.

This module contains the main evaluation functions for scoring responses,
orchestrating LLM-based evaluation calls, and handling batch processing
of evaluation requests with robust error handling and logging.

Key Functions:
    - evaluate_answers: Core function for evaluating multiple answers
    - batch_evaluate_from_file: Batch processing from JSON files
    
The module implements academic standards for AI system evaluation with
proper error handling, logging, and statistical rigor.
"""

import statistics
import json
from typing import List, Dict, Any, Optional, Union

from .config import logger, EvalConfig
from .advanced_config import AdvancedEvalConfig
from .enums import EvaluationDimension, SystemType
from .models import EvaluationRubric, AnswerEvaluation, EvaluationInput, SourceSnippet
from .utils import parse_file_content, normalize_system_type_name, handle_exceptions, log_operation, calculate_snippet_grounding_score
from .llm_client import call_llm_evaluator


def evaluate_answers_with_snippets(
    evaluation_inputs: List[EvaluationInput], 
    config: Union[EvalConfig, AdvancedEvalConfig]
) -> List[AnswerEvaluation]:
    """
    Evaluate answers with snippet support for enhanced RAG assessment.
    
    Implements academic standards for RAG evaluation including:
    - Source grounding analysis
    - Citation-snippet alignment
    - Source traceability
    
    Args:
        evaluation_inputs: List of EvaluationInput objects with queries, answers, and snippets
        config: Evaluation configuration (basic or advanced) with enhanced RAG rubrics
    
    Returns:
        List of evaluations with snippet grounding scores
    """
    # Convert basic config to advanced if needed
    if isinstance(config, EvalConfig):
        # Use basic config functionality
        actual_config = config
    else:
        # Use advanced config functionality
        actual_config = config
    results = []
    
    logger.info(f"Evaluating {len(evaluation_inputs)} responses with snippet analysis")
    
    for i, eval_input in enumerate(evaluation_inputs):
        logger.info(f"Evaluating response {i+1}/{len(evaluation_inputs)} "
                   f"({eval_input.system_type.value if eval_input.system_type else 'unknown type'})")
        
        # Perform evaluation with snippet context (pass configuration through)
        evaluation = call_llm_evaluator(
            eval_input.query, 
            eval_input.answer, 
            actual_config, 
            eval_input.system_type,
            snippets=eval_input.source_snippets
        )
        
        if evaluation:
            # Add snippet-specific analysis
            if eval_input.source_snippets:
                evaluation.source_snippets = eval_input.source_snippets
                evaluation.snippet_grounding_score = calculate_snippet_grounding_score(
                    eval_input.answer, 
                    eval_input.source_snippets
                )
                
                # Calculate citation-snippet alignment for each snippet
                citation_alignment = {}
                explicit_citations_present = _detect_explicit_citations(eval_input.answer)
                
                for snippet in eval_input.source_snippets:
                    if snippet.snippet_id:
                        if explicit_citations_present:
                            # For systems with explicit citations, check citation accuracy
                            citation_score = _calculate_citation_accuracy(
                                eval_input.answer, snippet
                            )
                        else:
                            # For systems without explicit citations, calculate content grounding
                            citation_score = _calculate_content_grounding(
                                eval_input.answer, snippet
                            )
                        citation_alignment[snippet.snippet_id] = citation_score
                
                evaluation.citation_snippet_alignment = citation_alignment
                
                # Update evaluation metadata
                evaluation.evaluation_metadata.update({
                    "snippet_count": len(eval_input.source_snippets),
                    "grounding_score": evaluation.snippet_grounding_score,
                    "has_snippet_support": True
                })
                
                logger.info(f"Response {i+1} - Grounding score: {evaluation.snippet_grounding_score:.3f}")
            else:
                evaluation.evaluation_metadata["has_snippet_support"] = False
            
            # Add metadata from evaluation input
            if eval_input.metadata:
                evaluation.evaluation_metadata.update(eval_input.metadata)
            
            results.append(evaluation)
            logger.info(f"Response {i+1} evaluated - Weighted score: {evaluation.weighted_total:.2f}")
        else:
            logger.error(f"Failed to evaluate response {i+1}")
    
    # DO NOT sort results - maintain original order for proper query-evaluation alignment
    # results.sort(key=lambda e: e.weighted_total, reverse=True)
    
    logger.info(f"Evaluation complete. {len(results)}/{len(evaluation_inputs)} responses successfully evaluated")
    return results


def evaluate_answers(
    query: str, 
    answers: List[str], 
    config: Union[EvalConfig, AdvancedEvalConfig],
    system_types: Optional[List[SystemType]] = None
) -> List[AnswerEvaluation]:
    """
    Evaluate multiple answers with enhanced academic rigor.
    
    Args:
        query: The user query
        answers: List of system responses
        config: Evaluation configuration
        system_types: Optional list of system types for each answer
    
    Returns:
        List of evaluations sorted by weighted score (descending)
    """
    results = []
    system_types = system_types or [None] * len(answers)
    
    logger.info(f"Evaluating {len(answers)} responses for query: {query[:100]}...")
    
    for i, (answer, sys_type) in enumerate(zip(answers, system_types)):
        logger.info(f"Evaluating response {i+1}/{len(answers)} "
                   f"({sys_type.value if sys_type else 'unknown type'})")
        
        evaluation = call_llm_evaluator(query, answer, config, sys_type)
        if evaluation:
            results.append(evaluation)
            logger.info(f"Response {i+1} evaluated - Weighted score: {evaluation.weighted_total:.2f}")
        else:
            logger.error(f"Failed to evaluate response {i+1}")
    
    # Sort by weighted score (descending)
    results.sort(key=lambda e: e.weighted_total, reverse=True)
    
    logger.info(f"Evaluation complete. {len(results)}/{len(answers)} responses successfully evaluated")
    return results


def batch_evaluate_from_file(
    input_file_path: str,
    config: Union[EvalConfig, AdvancedEvalConfig]
) -> List[AnswerEvaluation]:
    """
    Perform batch evaluation from a JSON file containing query-answer pairs.
    
    Expected file format:
    {
        "evaluations": [
            {
                "query": "What is machine learning?",
                "answers": ["Answer 1", "Answer 2"],
                "system_types": ["rag", "agentic"]  # Optional
            },
            ...
        ]
    }
    
    Args:
        input_file_path: Path to JSON file containing evaluation data
        config: Evaluation configuration
        
    Returns:
        List of all answer evaluations from the batch
    """
    logger.info(f"Starting batch evaluation from file: {input_file_path}")
    
    try:
        data = parse_file_content(input_file_path, "json")
        if not isinstance(data, dict):
            raise ValueError("Input file must contain a JSON object")
    except Exception as e:
        logger.error(f"Failed to load input file {input_file_path}: {e}")
        raise
    
    all_evaluations = []
    
    if "evaluations" not in data:
        logger.error("Input file must contain 'evaluations' key")
        raise ValueError("Invalid input file format: missing 'evaluations' key")
    
    evaluations_data = data["evaluations"]
    if not isinstance(evaluations_data, list):
        raise ValueError("'evaluations' must be a list")
    
    for i, eval_item in enumerate(evaluations_data):
        validated_item = validate_evaluation_item(eval_item, i)
        if validated_item is None:
            continue  # Skip this item, already logged
        
        query = validated_item["query"]
        answers = validated_item["answers"]
        system_types = validated_item["system_types"]
        
        logger.info(f"Evaluating batch item {i+1}/{len(evaluations_data)}: {len(answers)} answers")
        
        try:
            # Evaluate answers for this query
            evaluations = evaluate_answers(query, answers, config, system_types)
            all_evaluations.extend(evaluations)
        except Exception as e:
            logger.error(f"Failed to evaluate batch item {i}: {e}")
            continue
    
    logger.info(f"Batch evaluation completed: {len(all_evaluations)} total evaluations from {len(evaluations_data)} items")
    return all_evaluations


def validate_evaluation_item(eval_item: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """
    Validate and normalize a single evaluation item from batch input.
    
    Args:
        eval_item: Dictionary containing evaluation data
        index: Index in the batch for logging
        
    Returns:
        Normalized evaluation data or None if invalid
    """
    if not isinstance(eval_item, dict):
        logger.warning(f"Skipping evaluation {index}: not a dictionary")
        return None
        
    if "query" not in eval_item or "answers" not in eval_item:
        logger.warning(f"Skipping evaluation {index}: missing 'query' or 'answers'")
        return None
    
    query = eval_item["query"]
    answers = eval_item["answers"]
    
    if not isinstance(answers, list):
        logger.warning(f"Skipping evaluation {index}: 'answers' must be a list")
        return None
    
    # Parse system types if provided
    system_types = None
    if "system_types" in eval_item:
        if not isinstance(eval_item["system_types"], list):
            logger.warning(f"Evaluation {index}: 'system_types' must be a list")
        else:
            try:
                system_types = [SystemType(normalize_system_type_name(st)) for st in eval_item["system_types"]]
            except ValueError as e:
                logger.warning(f"Invalid system type in evaluation {index}: {e}")
                system_types = None
    
    return {
        "query": query,
        "answers": answers,
        "system_types": system_types
    }


def _detect_explicit_citations(answer: str) -> bool:
    """
    Detect if the answer contains explicit citations or references.
    
    Args:
        answer: The generated answer text
        
    Returns:
        True if explicit citations are detected, False otherwise
    """
    import re
    
    # Common citation patterns
    citation_patterns = [
        r'\[\d+\]',  # [1], [2], etc.
        r'\(\d+\)',  # (1), (2), etc.
        r'Source:',  # Explicit source mentions
        r'According to',  # Source attribution phrases
        r'Reference:',  # Reference mentions
        r'Snippet \d+',  # Direct snippet references
        r'Document \d+',  # Document references
        r'\*[^*]+\*',  # Italicized source names
    ]
    
    for pattern in citation_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            return True
    
    return False


def _calculate_citation_accuracy(answer: str, snippet: SourceSnippet) -> float:
    """
    Calculate citation accuracy for systems with explicit citations.
    
    Args:
        answer: The generated answer text
        snippet: Source snippet being referenced
        
    Returns:
        Citation accuracy score between 0.0 and 1.0
    """
    # For systems with explicit citations, check if the content
    # accurately reflects what's claimed to be cited
    snippet_tokens = set(snippet.content.lower().split())
    answer_tokens = set(answer.lower().split())
    
    # Look for direct content overlap
    overlap = len(snippet_tokens.intersection(answer_tokens))
    overlap_ratio = overlap / max(len(snippet_tokens), 1)
    
    # Bonus for explicit reference patterns near relevant content
    import re
    has_explicit_ref = bool(re.search(r'(source|snippet|reference|according to)', 
                                    answer.lower()))
    
    if has_explicit_ref:
        overlap_ratio *= 1.2  # Boost for explicit attribution
    
    return min(1.0, overlap_ratio)


def _calculate_content_grounding(answer: str, snippet: SourceSnippet) -> float:
    """
    Calculate content grounding for systems without explicit citations.
    
    Args:
        answer: The generated answer text
        snippet: Source snippet being used
        
    Returns:
        Content grounding score between 0.0 and 1.0
    """
    # For systems without explicit citations, focus on content derivation
    snippet_tokens = set(snippet.content.lower().split())
    answer_tokens = set(answer.lower().split())
    
    # Calculate semantic overlap
    overlap = len(snippet_tokens.intersection(answer_tokens))
    overlap_ratio = overlap / max(len(answer_tokens), 1)
    
    # Additional scoring for concept overlap (simplified)
    # This could be enhanced with semantic similarity models
    snippet_concepts = [word for word in snippet_tokens if len(word) > 3]
    answer_concepts = [word for word in answer_tokens if len(word) > 3]
    
    concept_overlap = len(set(snippet_concepts).intersection(set(answer_concepts)))
    concept_ratio = concept_overlap / max(len(answer_concepts), 1)
    
    # Combined score weighing both token and concept overlap
    grounding_score = (overlap_ratio * 0.6) + (concept_ratio * 0.4)
    
    return min(1.0, grounding_score)



