#!/usr/bin/env python3
"""
Debug script to check what dimensions are expected vs what LLM returns
"""

import json
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag_agentic_evaluation"))

from rag_agentic_evaluation.enums import EvaluationDimension
from rag_agentic_evaluation.utils import validate_evaluation_response

# Sample LLM response from the log
sample_response = {
    "scores": {
        "factual_accuracy": 5,
        "relevance": 5,
        "completeness": 5,
        "clarity": 5,
        "coherence": 5,
        "citation_quality": 1,
        "provenance": 1
    },
    "justifications": {
        "factual_accuracy": "The response accurately describes the offerings of the cafeteria.",
        "relevance": "The response directly addresses the user's query about what the cafeteria offers.",
        "completeness": "The response covers all necessary aspects of the cafeteria's offerings.",
        "clarity": "The response is well-organized and easy to read.",
        "coherence": "The information flows logically, with a clear structure.",
        "citation_quality": "The response lacks citations or references to support the claims.",
        "provenance": "There is no indication of the source of the information provided."
    },
    "confidence_scores": {
        "factual_accuracy": 1.0,
        "relevance": 1.0,
        "completeness": 1.0,
        "clarity": 1.0,
        "coherence": 1.0,
        "citation_quality": 0.0,
        "provenance": 0.0
    },
    "overall_assessment": "The AI response is excellent in content but lacks academic rigor."
}

# Test different dimension sets
print("Testing validation with different dimension combinations...")
print()

# Test 1: Basic dimensions (likely what the config uses)
basic_dimensions = [
    EvaluationDimension.FACTUAL_ACCURACY,
    EvaluationDimension.RELEVANCE,
    EvaluationDimension.COMPLETENESS,
    EvaluationDimension.CLARITY,
    EvaluationDimension.COHERENCE,
    EvaluationDimension.CITATION_QUALITY,
    EvaluationDimension.PROVENANCE
]

print("=== Test 1: Basic RAG dimensions ===")
print("Expected dimensions:", [dim.value for dim in basic_dimensions])
print("LLM response dimensions:", list(sample_response["scores"].keys()))
print("Expected set:", {dim.value for dim in basic_dimensions})
print("LLM set:", set(sample_response["scores"].keys()))
print("Are they equal?", {dim.value for dim in basic_dimensions} == set(sample_response["scores"].keys()))
print()

result = validate_evaluation_response(sample_response, basic_dimensions)
print(f"Validation result: {result}")
print()

# Test 2: All dimensions
all_dimensions = list(EvaluationDimension)
print("=== Test 2: All possible dimensions ===")
print("Expected dimensions:", [dim.value for dim in all_dimensions])
print("LLM response dimensions:", list(sample_response["scores"].keys()))
print("Expected set:", {dim.value for dim in all_dimensions})
print("LLM set:", set(sample_response["scores"].keys()))
print("Are they equal?", {dim.value for dim in all_dimensions} == set(sample_response["scores"].keys()))
print()

result2 = validate_evaluation_response(sample_response, all_dimensions)
print(f"Validation result: {result2}")
print()

# Test 3: Check what dimensions are missing
if not result:
    expected_set = {dim.value for dim in basic_dimensions}
    actual_set = set(sample_response["scores"].keys())
    
    missing_in_response = expected_set - actual_set
    extra_in_response = actual_set - expected_set
    
    print("=== Dimension Analysis ===")
    print(f"Missing in LLM response: {missing_in_response}")
    print(f"Extra in LLM response: {extra_in_response}")
    print()

# Test 4: Check the config class dimensions
try:
    from rag_agentic_evaluation.config import EvalConfig
    config = EvalConfig()
    print("=== Test 4: Default config dimensions ===")
    print(f"Config dimensions: {[dim.value for dim in config.dimensions]}")
    result3 = validate_evaluation_response(sample_response, config.dimensions)
    print(f"Validation with config dimensions: {result3}")
    
    if not result3:
        expected_config = {dim.value for dim in config.dimensions}
        actual_set = set(sample_response["scores"].keys())
        
        missing_in_response = expected_config - actual_set
        extra_in_response = actual_set - expected_config
        
        print(f"Missing in LLM response: {missing_in_response}")
        print(f"Extra in LLM response: {extra_in_response}")
    
except Exception as e:
    print(f"Could not load config: {e}")
