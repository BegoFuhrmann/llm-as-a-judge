"""
End-to-End Evaluation Pipeline Test Script.

This script orchestrates the full evaluation pipeline:
1. Loads evaluation data from an Excel file.
2. Configures the evaluation system with an advanced profile.
3. Initializes the core evaluation engine.
4. Runs the evaluation for each data row.
5. Generates and saves a comprehensive report.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from evaluation_system.excel_processor import load_excel_with_snippets
from evaluation_system.evaluation import evaluate_answers_with_snippets
from evaluation_system.advanced_config import AdvancedEvalConfig, ConfigurationProfile
from evaluation_system.report import generate_academic_citation_report
from evaluation_system.enums import EvaluationDimension

def run_evaluation_pipeline():
    """
    Executes the complete evaluation pipeline from data loading to report generation.
    """
    print("--- Starting End-to-End Evaluation Test ---")

    # --- 1. Configuration ---
    input_excel_path = "sample_evaluation_data.xlsx"
    
    # Create timestamped subfolder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path("test_evaluation_results") / f"run_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {run_folder}")

    if not os.path.exists(input_excel_path):
        print(f"Error: Input file not found at '{input_excel_path}'")
        return

    # --- 2. Load Evaluation Data ---
    print(f"Loading evaluation data from: {input_excel_path}")
    try:
        evaluation_data = load_excel_with_snippets(input_excel_path)
        print(f"Successfully loaded {len(evaluation_data)} records for evaluation.")
    except Exception as e:
        print(f"Failed to load or process Excel file: {e}")
        return

    # --- 3. Initialize Evaluation System ---
    print("Initializing evaluation system...")
    try:
        # Use advanced evaluation configuration with all dimensions including regulatory
        config = AdvancedEvalConfig(profile=ConfigurationProfile.ACADEMIC_RESEARCH)
        print("Evaluation system initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize evaluation system: {e}")
        return

    # --- 4. Run Evaluation ---
    print("\n--- Running Evaluation on Sample Data ---")
    try:
        results = evaluate_answers_with_snippets(evaluation_data, config)
        print(f"Successfully evaluated {len(results)} records.")
    except Exception as e:
        print(f"Failed to run evaluation: {e}")
        return
    
    if not results:
        print("No results were generated. Aborting report generation.")
        return
        
    print("\n--- Evaluation Complete ---")

    # --- 5. Generate Report ---
    print("Generating evaluation reports...")
    try:
        # Generate academic report
        academic_report = generate_academic_citation_report(results, config=config)
        
        # Save academic report to file
        academic_report_path = run_folder / "academic_evaluation_report.txt"
        with open(academic_report_path, 'w', encoding='utf-8') as f:
            f.write(academic_report)
        print(f"Academic report saved to: {academic_report_path}")

        # Create detailed JSON results
        detailed_results = []
        for i, (eval_input, result) in enumerate(zip(evaluation_data, results)):
            result_dict = {
                "row_number": i + 1,
                "question": eval_input.query,
                "answer": eval_input.answer,
                "system_type": eval_input.system_type.value if eval_input.system_type else None,
                "source_snippets": [
                    {
                        "content": snippet.content,
                        "metadata": snippet.metadata
                    } for snippet in eval_input.source_snippets
                ],
                "evaluation_scores": {
                    "factual_accuracy": getattr(result, 'scores', {}).get(EvaluationDimension.FACTUAL_ACCURACY) if hasattr(result, 'scores') else None,
                    "relevance": getattr(result, 'scores', {}).get(EvaluationDimension.RELEVANCE) if hasattr(result, 'scores') else None,
                    "completeness": getattr(result, 'scores', {}).get(EvaluationDimension.COMPLETENESS) if hasattr(result, 'scores') else None,
                    "clarity": getattr(result, 'scores', {}).get(EvaluationDimension.CLARITY) if hasattr(result, 'scores') else None,
                    "citation_quality": getattr(result, 'scores', {}).get(EvaluationDimension.CITATION_QUALITY) if hasattr(result, 'scores') else None,
                    # Regulatory dimensions
                    "gdpr_compliance": getattr(result, 'scores', {}).get(EvaluationDimension.GDPR_COMPLIANCE) if hasattr(result, 'scores') else None,
                    "eu_ai_act_alignment": getattr(result, 'scores', {}).get(EvaluationDimension.EU_AI_ACT_ALIGNMENT) if hasattr(result, 'scores') else None,
                    "audit_trail_quality": getattr(result, 'scores', {}).get(EvaluationDimension.AUDIT_TRAIL_QUALITY) if hasattr(result, 'scores') else None,
                    # Additional dimensions
                    "reasoning_depth": getattr(result, 'scores', {}).get(EvaluationDimension.REASONING_DEPTH) if hasattr(result, 'scores') else None,
                    "adaptability": getattr(result, 'scores', {}).get(EvaluationDimension.ADAPTABILITY) if hasattr(result, 'scores') else None,
                    "efficiency": getattr(result, 'scores', {}).get(EvaluationDimension.EFFICIENCY) if hasattr(result, 'scores') else None,
                },
                "confidence_scores": {
                    key.value if hasattr(key, 'value') else str(key): value 
                    for key, value in getattr(result, 'confidence_scores', {}).items()
                },
                "justifications": {
                    key.value if hasattr(key, 'value') else str(key): value 
                    for key, value in getattr(result, 'justifications', {}).items()
                },
                "weighted_total": result.weighted_total if hasattr(result, 'weighted_total') else None,
                "raw_total": result.raw_total if hasattr(result, 'raw_total') else None,
                "snippet_grounding_score": result.snippet_grounding_score if hasattr(result, 'snippet_grounding_score') else None,
                "evaluation_metadata": {
                    "timestamp": result.timestamp if hasattr(result, 'timestamp') else None,
                    "model_used": result.model_used if hasattr(result, 'model_used') else None,
                    "evaluation_duration": result.evaluation_duration if hasattr(result, 'evaluation_duration') else None
                }
            }
            detailed_results.append(result_dict)

        # Save detailed JSON results
        json_results_path = run_folder / "detailed_evaluation_results.json"
        with open(json_results_path, 'w', encoding='utf-8') as f:
            json.dump({
                "run_metadata": {
                    "timestamp": timestamp,
                    "input_file": input_excel_path,
                    "total_evaluations": len(results),
                    "evaluation_config": "AdvancedEvalConfig (academic_research)"
                },
                "results": detailed_results
            }, f, indent=2, ensure_ascii=False)
        print(f"Detailed JSON results saved to: {json_results_path}")

        # Save run summary
        summary_data = {
            "run_timestamp": timestamp,
            "input_file": input_excel_path,
            "total_records": len(evaluation_data),
            "successful_evaluations": len(results),
            "average_scores": {},
            "files_generated": [
                str(academic_report_path.name),
                str(json_results_path.name)
            ]
        }

        # Calculate average scores if available
        if results:
            score_attributes = [
                'factual_accuracy', 'relevance', 'completeness', 'clarity', 'citation_quality',
                'gdpr_compliance', 'eu_ai_act_alignment', 'audit_trail_quality',
                'reasoning_depth', 'adaptability', 'efficiency', 'weighted_total'
            ]
            for attr in score_attributes:
                scores = [getattr(result, attr) for result in results if hasattr(result, attr) and getattr(result, attr) is not None]
                if scores:
                    summary_data["average_scores"][attr] = {
                        "mean": sum(scores) / len(scores),
                        "count": len(scores)
                    }

        summary_path = run_folder / "run_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Run summary saved to: {summary_path}")

        # Print a brief summary to console
        print("\n--- Evaluation Summary ---")
        if results:
            print(f"Total evaluations: {len(results)}")
            print(f"Results saved in folder: {run_folder}")
            for attr, data in summary_data["average_scores"].items():
                print(f"Average {attr}: {data['mean']:.2f} (n={data['count']})")
        else:
            print("No results to summarize.")

    except Exception as e:
        print(f"Failed to generate report: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- End-to-End Evaluation Test Finished ---")


if __name__ == "__main__":
    run_evaluation_pipeline()
