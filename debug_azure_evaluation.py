#!/usr/bin/env python3
"""
Simple evaluation test to debug the validation issue.
"""

import os
import sys
import json
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rag_agentic_evaluation.llm_client import test_llm_connection, call_llm_evaluator
from rag_agentic_evaluation.config import EvalConfig
from rag_agentic_evaluation.enums import EvaluationDimension
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Debug the evaluation response validation."""
    
    logger.info("=== Azure OpenAI Evaluation Debug ===")
    
    # Initialize configuration
    config = EvalConfig()
    
    # Test connection
    logger.info("Testing Azure OpenAI connection...")
    if not test_llm_connection(config):
        logger.error("Failed to connect to Azure OpenAI")
        return
    
    logger.info("✓ Azure OpenAI connection successful!")
    
    # Print expected dimensions
    expected_dims = [dim.value for dim in config.dimensions]
    logger.info(f"Expected dimensions: {expected_dims}")
    
    # Test a simple evaluation
    test_query = "Where can I park my car?"
    test_answer = "You can park in the internal parking garage (100 spaces, €2/hour) or external lot (50 spaces, €1/hour)."
    
    logger.info("Testing evaluation...")
    result = call_llm_evaluator(
        query=test_query,
        answer=test_answer,
        config=config
    )
    
    if result:
        logger.info("✓ Evaluation successful!")
        logger.info(f"Weighted Score: {result.weighted_total:.2f}")
        logger.info(f"Raw Score: {result.raw_total:.2f}")
        logger.info(f"Scores: {result.scores}")
        logger.info(f"Confidence: {result.confidence_scores}")
    else:
        logger.error("✗ Evaluation failed")

if __name__ == "__main__":
    main()
