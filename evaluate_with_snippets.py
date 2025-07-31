#!/usr/bin/env python3
"""
Enhanced evaluation script that properly includes snippets for citation and provenance scoring.

This script loads Excel data with Question, Answer, and Snippet columns,
then evaluates the answers with proper snippet context for accurate 
citation quality and provenance assessment.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from rag_agentic_evaluation.config import EvalConfig, logger
from rag_agentic_evaluation.excel_processor import load_excel_with_snippets, validate_excel_structure
from rag_agentic_evaluation.evaluation import evaluate_answers_with_snippets
from rag_agentic_evaluation.report import generate_summary_report
from rag_agentic_evaluation.utils import ensure_directory


def main():
    """Main evaluation function with snippet-enhanced context."""
    
    # Configuration
    excel_file = "test_snippets.xlsx"
    output_dir = f"snippet_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("=== RAG Evaluation with Snippet Context ===")
    logger.info(f"Excel file: {excel_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Validate Excel structure first
    logger.info("Validating Excel file structure...")
    try:
        validation_result = validate_excel_structure(excel_file)
        logger.info(f"Excel validation: {validation_result}")
        
        if not validation_result["valid"]:
            logger.error("Excel file validation failed!")
            for error in validation_result["errors"]:
                logger.error(f"  - {error}")
            return
            
        logger.info(f"Excel file is valid: {validation_result['valid_rows']}/{validation_result['total_rows']} valid rows")
        
    except Exception as e:
        logger.error(f"Failed to validate Excel file: {e}")
        return
    
    # Load evaluation data with snippets
    logger.info("Loading evaluation data with snippets...")
    try:
        evaluation_inputs = load_excel_with_snippets(
            excel_file,
            question_col="Question",
            answer_col="Answer", 
            snippet_col="Snippet"
        )
        
        logger.info(f"Loaded {len(evaluation_inputs)} evaluation items")
        
        # Log snippet statistics
        total_snippets = sum(len(item.source_snippets) for item in evaluation_inputs)
        items_with_snippets = sum(1 for item in evaluation_inputs if item.source_snippets)
        
        logger.info(f"Snippet statistics:")
        logger.info(f"  - Total snippets: {total_snippets}")
        logger.info(f"  - Items with snippets: {items_with_snippets}/{len(evaluation_inputs)}")
        
        # Show example of loaded data
        if evaluation_inputs:
            example = evaluation_inputs[0]
            logger.info(f"Example loaded item:")
            logger.info(f"  - Query: {example.query[:100]}...")
            logger.info(f"  - Answer: {example.answer[:100]}...")
            logger.info(f"  - Snippets: {len(example.source_snippets)}")
            if example.source_snippets:
                logger.info(f"  - First snippet: {example.source_snippets[0].content[:100]}...")
        
    except Exception as e:
        logger.error(f"Failed to load Excel data: {e}")
        return
    
    # Create evaluation configuration
    logger.info("Setting up evaluation configuration...")
    config = EvalConfig()
    
    # Log configuration details
    logger.info(f"Evaluation configuration:")
    logger.info(f"  - Model: {config.model_name}")
    logger.info(f"  - Temperature: {config.temperature}")
    logger.info(f"  - Dimensions: {[dim.value for dim in config.dimensions]}")
    logger.info(f"  - Max retries: {config.max_retries}")
    
    # Perform evaluation with snippet support
    logger.info("Starting evaluation with snippet context...")
    try:
        evaluations = evaluate_answers_with_snippets(evaluation_inputs, config)
        
        logger.info(f"Evaluation completed: {len(evaluations)} items evaluated")
        
        # Log evaluation summary
        if evaluations:
            scores = [eval.weighted_total for eval in evaluations]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            logger.info(f"Score statistics:")
            logger.info(f"  - Average: {avg_score:.3f}")
            logger.info(f"  - Range: {min_score:.3f} - {max_score:.3f}")
            
            # Log snippet grounding scores
            grounding_scores = [eval.snippet_grounding_score for eval in evaluations if hasattr(eval, 'snippet_grounding_score') and eval.snippet_grounding_score is not None]
            if grounding_scores:
                avg_grounding = sum(grounding_scores) / len(grounding_scores)
                logger.info(f"  - Average snippet grounding: {avg_grounding:.3f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
    
    # Generate comprehensive report
    logger.info("Generating evaluation report...")
    try:
        report_path = Path(output_dir) / "evaluation_report.md"
        report_content = generate_summary_report(evaluations)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {report_path}")
        
        # Save detailed results as JSON
        json_path = Path(output_dir) / "detailed_results.json"
        detailed_results = []
        
        for i, evaluation in enumerate(evaluations):
            result = {
                "item_number": i + 1,
                "query": evaluation_inputs[i].query if i < len(evaluation_inputs) else "",
                "answer": evaluation_inputs[i].answer if i < len(evaluation_inputs) else "",
                "system_type": evaluation.system_type.value if evaluation.system_type else None,
                "scores": {dim.value: evaluation.scores.get(dim, 0) for dim in config.dimensions},
                "weighted_total": evaluation.weighted_total,
                "raw_total": evaluation.raw_total,
                "snippet_grounding_score": getattr(evaluation, 'snippet_grounding_score', None),
                "snippet_count": len(evaluation.source_snippets) if evaluation.source_snippets else 0,
                "citation_alignment": getattr(evaluation, 'citation_snippet_alignment', {}),
                "justifications": {dim.value: evaluation.justifications.get(dim, "") for dim in config.dimensions},
                "overall_assessment": evaluation.overall_assessment,
                "evaluation_metadata": evaluation.evaluation_metadata,
                "timestamp": evaluation.evaluation_metadata.get("timestamp", datetime.now().isoformat())
            }
            detailed_results.append(result)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluation_summary": {
                    "total_items": len(evaluations),
                    "average_score": sum(eval.weighted_total for eval in evaluations) / len(evaluations) if evaluations else 0,
                    "average_grounding_score": sum(getattr(eval, 'snippet_grounding_score', 0) for eval in evaluations) / len(evaluations) if evaluations else 0,
                    "evaluation_timestamp": datetime.now().isoformat()
                },
                "detailed_results": detailed_results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to: {json_path}")
        
        # Save summary CSV
        csv_path = Path(output_dir) / "evaluation_summary.csv"
        import pandas as pd
        
        csv_data = []
        for i, evaluation in enumerate(evaluations):
            row = {
                "Item": i + 1,
                "Query": (evaluation_inputs[i].query if i < len(evaluation_inputs) else "")[:100] + "...",
                "System_Type": evaluation.system_type.value if evaluation.system_type else "unknown",
                "Weighted_Total": f"{evaluation.weighted_total:.3f}",
                "Snippet_Grounding": f"{getattr(evaluation, 'snippet_grounding_score', 0):.3f}",
                "Snippet_Count": len(evaluation.source_snippets) if evaluation.source_snippets else 0,
                **{f"{dim.value.title()}": f"{evaluation.scores.get(dim, 0):.2f}" for dim in config.dimensions}
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Summary CSV saved to: {csv_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
    
    logger.info("=== Evaluation Complete ===")
    logger.info(f"Results saved in: {output_dir}")
    logger.info("Check the markdown report for detailed analysis including snippet grounding scores.")


if __name__ == "__main__":
    main()
