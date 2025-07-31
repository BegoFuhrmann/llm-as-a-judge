#!/usr/bin/env python3
"""
Evaluate test_snippets.xlsx using the updated Azure OpenAI configuration.

This script uses the academic RAG vs Agentic AI evaluation framework 
with your working Azure OpenAI setup.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the evaluation framework
from rag_agentic_evaluation.excel_processor import load_excel_with_snippets
from rag_agentic_evaluation.evaluation import evaluate_answers_with_snippets
from rag_agentic_evaluation.config import EvalConfig
from rag_agentic_evaluation.llm_client import test_llm_connection
from rag_agentic_evaluation.report import generate_summary_report, generate_academic_citation_report
from rag_agentic_evaluation.enums import SystemType
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    excel_file = "test_snippets.xlsx"
    
    if not Path(excel_file).exists():
        logger.error(f"Excel file not found: {excel_file}")
        return
    
    logger.info("=== RAG vs Agentic AI Evaluation Framework ===")
    logger.info(f"Evaluating: {excel_file}")
    
    # Initialize configuration
    config = EvalConfig()
    
    # Test Azure OpenAI connection first
    logger.info("Testing Azure OpenAI connection...")
    if not test_llm_connection(config):
        logger.error("Failed to connect to Azure OpenAI. Please check your .env configuration.")
        logger.error("Required variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
        return
    
    logger.info("Azure OpenAI connection successful!")
    
    try:
        # Load Excel data with snippets
        logger.info("Loading Excel file...")
        evaluation_inputs = load_excel_with_snippets(
            excel_file,
            question_col="Question",
            answer_col="Answer", 
            snippet_col="Snippet"
        )
        
        logger.info(f"Loaded {len(evaluation_inputs)} evaluation items")
        
        # Run evaluations (treating all as RAG system since they have snippets)
        logger.info("Starting RAG evaluations with snippet analysis...")
        
        # Evaluate all items at once
        results = evaluate_answers_with_snippets(evaluation_inputs, config)
        
        if not results:
            logger.error("No successful evaluations completed")
            return
        
        logger.info(f"Completed {len(results)} evaluations successfully")
        
        # Log individual results
        for i, result in enumerate(results, 1):
            if result:
                logger.info(f"Evaluation {i} completed successfully")
                logger.info(f"  Weighted Score: {result.weighted_total:.2f}")
                logger.info(f"  Raw Score: {result.raw_total:.2f}")
                logger.info(f"  Confidence: {sum(result.confidence_scores.values())/len(result.confidence_scores):.2f}")
            else:
                logger.error(f"Evaluation {i} failed")
        
        # Generate comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"evaluation_results_{timestamp}"
        
        logger.info(f"Generating comprehensive report in: {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate academic report
        academic_report = generate_academic_citation_report(results, config=config)
        summary_report = generate_summary_report(results)
        
        # Save reports
        with open(f"{output_dir}/academic_report.md", "w", encoding="utf-8") as f:
            f.write(academic_report)
        
        with open(f"{output_dir}/summary_report.md", "w", encoding="utf-8") as f:
            f.write(summary_report)
        
        # Save raw results as JSON
        results_json = []
        for result in results:
            result_dict = {
                "answer": result.answer,
                "system_type": result.system_type.value if result.system_type else None,
                "scores": {dim.value: score for dim, score in result.scores.items()},
                "justifications": {dim.value: just for dim, just in result.justifications.items()},
                "confidence_scores": {dim.value: conf for dim, conf in result.confidence_scores.items()},
                "weighted_total": result.weighted_total,
                "raw_total": result.raw_total,
                "overall_assessment": result.overall_assessment,
                "evaluation_metadata": result.evaluation_metadata
            }
            results_json.append(result_dict)
        
        with open(f"{output_dir}/evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        # Create simple CSV summary
        import csv
        with open(f"{output_dir}/evaluation_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "System_Type", "Weighted_Score", "Raw_Score", "Avg_Confidence"])
            
            for i, result in enumerate(results):
                question = evaluation_inputs[i].query[:100] + "..." if len(evaluation_inputs[i].query) > 100 else evaluation_inputs[i].query
                avg_confidence = sum(result.confidence_scores.values()) / len(result.confidence_scores)
                writer.writerow([
                    question,
                    result.system_type.value if result.system_type else "Unknown",
                    f"{result.weighted_total:.2f}",
                    f"{result.raw_total:.2f}",
                    f"{avg_confidence:.2f}"
                ])
        
        logger.info("=== Evaluation Complete ===")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Check the following files:")
        logger.info(f"  - {output_dir}/evaluation_summary.csv")
        logger.info(f"  - {output_dir}/evaluation_results.json")
        logger.info(f"  - {output_dir}/academic_report.md")
        logger.info(f"  - {output_dir}/summary_report.md")
        
        # Print summary statistics
        weighted_scores = [r.weighted_total for r in results]
        raw_scores = [r.raw_total for r in results]
        confidence_scores = [sum(r.confidence_scores.values())/len(r.confidence_scores) for r in results]
        
        logger.info("=== Summary Statistics ===")
        logger.info(f"Average Weighted Score: {sum(weighted_scores)/len(weighted_scores):.2f}")
        logger.info(f"Average Raw Score: {sum(raw_scores)/len(raw_scores):.2f}")
        logger.info(f"Average Confidence: {sum(confidence_scores)/len(confidence_scores):.2f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
