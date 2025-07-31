#!/usr/bin/env python3
"""
Excel to Evaluation Framework Processor with Azure OpenAI Support

This script processes an Excel file with questions and answers into the format
expected by the RAG vs Agentic AI evaluation framework. It supports both
single-answer evaluation and batch processing with Azure OpenAI integration.

Usage:
    python process_excel_for_evaluation.py input_questions.xlsx --output evaluation_data.json
    python process_excel_for_evaluation.py input_questions.xlsx --evaluate-now --system-type rag
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# Import the evaluation framework
from rag_agentic_evaluation import (
    EvalConfig, 
    SystemType,
    evaluate_answers,
    batch_evaluate_from_file,
    generate_summary_report,
    generate_csv_export,
    setup_logging,
    save_json_file,
    ensure_directory,
    handle_exceptions,
    log_operation
)

logger = logging.getLogger(__name__)


@handle_exceptions("Failed to load Excel file", return_value=None, raise_on_error=True)
@log_operation("Loading Excel file")
def load_excel_file(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load Excel file and return as DataFrame.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Optional sheet name (uses first sheet if None)
        
    Returns:
        DataFrame containing the Excel data
    """
    try:
        # Try to read the Excel file
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        
        logger.info(f"Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns found: {list(df.columns)}")
        
        return df
    except Exception as e:
        logger.error(f"Failed to load Excel file {file_path}: {e}")
        raise


def validate_excel_structure(df: pd.DataFrame) -> bool:
    """
    Validate that the Excel file has the required structure.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid structure, False otherwise
    """
    required_columns = ['Question', 'Answer']
    
    # Check if required columns exist (case-insensitive)
    df_columns_lower = [col.lower() for col in df.columns]
    missing_columns = []
    
    for required_col in required_columns:
        if required_col.lower() not in df_columns_lower:
            missing_columns.append(required_col)
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {list(df.columns)}")
        return False
    
    # Check for empty data
    if len(df) == 0:
        logger.error("Excel file contains no data rows")
        return False
    
    return True


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to match expected format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    # Create a mapping for common column name variations
    column_mapping = {}
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['question', 'query', 'q']:
            column_mapping[col] = 'Question'
        elif col_lower in ['answer', 'response', 'a']:
            column_mapping[col] = 'Answer'
        elif col_lower in ['system_type', 'system', 'type']:
            column_mapping[col] = 'SystemType'
    
    # Rename columns
    df_renamed = df.rename(columns=column_mapping)
    logger.info(f"Normalized column names: {column_mapping}")
    
    return df_renamed


@log_operation("Converting Excel to evaluation format")
def convert_to_evaluation_format(
    df: pd.DataFrame, 
    system_type: Optional[SystemType] = None
) -> Dict[str, Any]:
    """
    Convert DataFrame to evaluation framework format.
    
    Args:
        df: Input DataFrame with Question and Answer columns
        system_type: Optional system type to assign to all answers
        
    Returns:
        Dictionary in evaluation framework format
    """
    evaluations = []
    
    for idx, row in df.iterrows():
        # Skip rows with missing data
        if pd.isna(row['Question']) or pd.isna(row['Answer']):
            logger.warning(f"Skipping row {idx + 1}: missing question or answer")
            continue
        
        question = str(row['Question']).strip()
        answer = str(row['Answer']).strip()
        
        if not question or not answer:
            logger.warning(f"Skipping row {idx + 1}: empty question or answer")
            continue
        
        eval_item = {
            "query": question,
            "answers": [answer]  # Single answer per question
        }
        
        # Add system type if provided or if column exists
        if system_type:
            eval_item["system_types"] = [system_type.value]
        elif 'SystemType' in df.columns and not pd.isna(row['SystemType']):
            try:
                sys_type = SystemType(str(row['SystemType']).lower())
                eval_item["system_types"] = [sys_type.value]
            except ValueError:
                logger.warning(f"Invalid system type in row {idx + 1}: {row['SystemType']}")
        
        evaluations.append(eval_item)
    
    logger.info(f"Converted {len(evaluations)} valid question-answer pairs")
    
    return {"evaluations": evaluations}


def validate_azure_environment() -> bool:
    """
    Validate Azure OpenAI environment configuration.
    
    Returns:
        True if environment is properly configured, False otherwise
    """
    azure_endpoint = os.getenv("ENDPOINT_URL", os.getenv("AZURE_OPENAI_ENDPOINT"))
    deployment_name = os.getenv("DEPLOYMENT_NAME", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if azure_endpoint:
        logger.info(f"Azure OpenAI endpoint configured: {azure_endpoint}")
        if deployment_name:
            logger.info(f"Azure deployment configured: {deployment_name}")
        else:
            logger.warning("DEPLOYMENT_NAME not set, will use default model name")
        return True
    elif openai_api_key:
        logger.info("Standard OpenAI API key configured")
        return True
    else:
        logger.error("Neither Azure OpenAI nor standard OpenAI is properly configured")
        return False


@log_operation("Running immediate evaluation")
def run_immediate_evaluation(
    evaluation_data: Dict[str, Any],
    output_dir: str = "evaluation_results"
) -> None:
    """
    Run evaluation immediately and generate reports.
    
    Args:
        evaluation_data: Evaluation data in framework format
        output_dir: Directory to save results
    """
    config = EvalConfig()
    
    # Save the data to a temporary file for batch processing
    temp_file = Path(output_dir) / "temp_evaluation_data.json"
    ensure_directory(temp_file.parent)
    save_json_file(evaluation_data, temp_file)
    
    try:
        # Run batch evaluation
        logger.info("Starting evaluation of Excel data...")
        evaluations = batch_evaluate_from_file(str(temp_file), config)
        
        if not evaluations:
            logger.error("No evaluations were completed successfully")
            return
        
        # Generate reports
        report_base = Path(output_dir) / "excel_evaluation_report"
        
        # Summary report
        summary = generate_summary_report(evaluations)
        summary_file = f"{report_base}_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Summary report saved to: {summary_file}")
        
        # CSV export
        csv_file = f"{report_base}.csv"
        generate_csv_export(evaluations, csv_file)
        logger.info(f"CSV results saved to: {csv_file}")
        
        # JSON export with full evaluation data
        json_file = f"{report_base}.json"
        evaluation_results = {
            "metadata": {
                "total_evaluations": len(evaluations),
                "source_file": "input_questions.xlsx",
                "evaluation_timestamp": pd.Timestamp.now().isoformat(),
                "azure_openai_configured": bool(os.getenv("ENDPOINT_URL", os.getenv("AZURE_OPENAI_ENDPOINT")))
            },
            "evaluations": [eval_result.model_dump() for eval_result in evaluations]
        }
        save_json_file(evaluation_results, json_file)
        logger.info(f"Detailed results saved to: {json_file}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total questions evaluated: {len(evaluations)}")
        
        if evaluations:
            scores = [e.weighted_total for e in evaluations]
            print(f"Average weighted score: {sum(scores)/len(scores):.2f}")
            print(f"Highest score: {max(scores):.2f}")
            print(f"Lowest score: {min(scores):.2f}")
        
        azure_configured = os.getenv("ENDPOINT_URL", os.getenv("AZURE_OPENAI_ENDPOINT"))
        api_type = "Azure OpenAI" if azure_configured else "Standard OpenAI"
        print(f"API used: {api_type}")
        
        print(f"\nDetailed reports saved to: {output_dir}/")
        print("="*60)
        
    finally:
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()


def main():
    """Main entry point for Excel processing script."""
    parser = argparse.ArgumentParser(
        description="Process Excel file for RAG vs Agentic AI evaluation framework with Azure OpenAI support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables for Azure OpenAI:
    ENDPOINT_URL or AZURE_OPENAI_ENDPOINT    Azure OpenAI endpoint URL
    DEPLOYMENT_NAME or AZURE_OPENAI_DEPLOYMENT    Azure deployment name
    OPENAI_API_KEY                           Standard OpenAI API key (fallback)

Examples:
    # Convert Excel to JSON format for later evaluation
    python process_excel_for_evaluation.py input_questions.xlsx --output evaluation_data.json
    
    # Evaluate immediately with Azure OpenAI
    set ENDPOINT_URL=https://your-resource.openai.azure.com/
    set DEPLOYMENT_NAME=gpt-4o-mini
    python process_excel_for_evaluation.py input_questions.xlsx --evaluate-now --system-type rag
    
    # Evaluate with standard OpenAI API
    set OPENAI_API_KEY=your-api-key-here
    python process_excel_for_evaluation.py input_questions.xlsx --evaluate-now --system-type rag
        """
    )
    
    parser.add_argument(
        'excel_file',
        help='Path to Excel file containing questions and answers'
    )
    
    parser.add_argument(
        '--output',
        help='Output JSON file path (default: evaluation_data.json)',
        default='evaluation_data.json'
    )
    
    parser.add_argument(
        '--sheet',
        help='Excel sheet name to process (uses first sheet if not specified)'
    )
    
    parser.add_argument(
        '--system-type',
        choices=['rag', 'agentic'],
        help='System type to assign to all answers'
    )
    
    parser.add_argument(
        '--evaluate-now',
        action='store_true',
        help='Run evaluation immediately instead of just converting to JSON'
    )
    
    parser.add_argument(
        '--output-dir',
        default='evaluation_results',
        help='Directory to save evaluation results (default: evaluation_results)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=getattr(logging, args.log_level.upper()))
    
    # Validate environment if evaluating now
    if args.evaluate_now:
        if not validate_azure_environment():
            print("ERROR: Neither Azure OpenAI nor standard OpenAI is properly configured.")
            print("\nFor Azure OpenAI, set environment variables:")
            print("  ENDPOINT_URL=https://your-resource.openai.azure.com/")
            print("  DEPLOYMENT_NAME=gpt-4o-mini")
            print("\nFor standard OpenAI, set:")
            print("  OPENAI_API_KEY=your-api-key-here")
            return 1
    
    try:
        # Load and validate Excel file
        logger.info(f"Processing Excel file: {args.excel_file}")
        df = load_excel_file(args.excel_file, args.sheet)
        
        # Normalize column names
        df = normalize_column_names(df)
        
        # Validate structure
        if not validate_excel_structure(df):
            logger.error("Excel file does not have the required structure")
            print("ERROR: Excel file must have 'Question' and 'Answer' columns")
            return 1
        
        # Convert system type if provided
        system_type = None
        if args.system_type:
            system_type = SystemType(args.system_type)
        
        # Convert to evaluation format
        evaluation_data = convert_to_evaluation_format(df, system_type)
        
        if args.evaluate_now:
            # Run evaluation immediately
            run_immediate_evaluation(evaluation_data, args.output_dir)
        else:
            # Save to JSON file for later evaluation
            ensure_directory(Path(args.output).parent)
            save_json_file(evaluation_data, args.output)
            logger.info(f"Evaluation data saved to: {args.output}")
            print(f"SUCCESS: Converted {len(evaluation_data['evaluations'])} questions to evaluation format")
            print(f"Saved to: {args.output}")
            print(f"To evaluate now, run:")
            print(f"python -m rag_agentic_evaluation batch --input {args.output} --output results.json")
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"ERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
