"""
Command-line interface and main entry point for the RAG vs Agentic AI evaluation framework.

This module provides the CLI interface and orchestrates the evaluation pipeline,
integrating all components of the modular evaluation system.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import EvalConfig, setup_logging
from .enums import SystemType
from .evaluation import evaluate_answers, batch_evaluate_from_file, evaluate_answers_with_snippets
from .excel_processor import load_excel_with_snippets, validate_excel_structure, export_evaluation_results_to_excel
from .statistics import perform_statistical_comparison
from .report import (
    generate_academic_citation_report, 
    generate_summary_report,
    generate_csv_export,
    generate_json_export
)
from .utils import (
    validate_environment_variables, ensure_directory, save_json_file, 
    handle_exceptions, log_operation, load_json_file, write_text_file
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the evaluation framework."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file=args.log_file)
    
    logger.info("Starting RAG vs Agentic AI Evaluation Framework")
    
    # Skip environment validation for validation-only modes
    validation_only = (
        hasattr(args, 'validate_only') and args.validate_only
    )
    
    if not validation_only:
        # Validate environment
        azure_endpoint = os.getenv("ENDPOINT_URL", os.getenv("AZURE_OPENAI_ENDPOINT"))
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not azure_endpoint and not openai_api_key:
            logger.error("Neither Azure OpenAI endpoint nor OPENAI_API_KEY is configured.")
            logger.info("For Azure OpenAI, set ENDPOINT_URL or AZURE_OPENAI_ENDPOINT")
            logger.info("For standard OpenAI, set OPENAI_API_KEY")
            return 1
        
        if azure_endpoint:
            logger.info(f"Using Azure OpenAI with endpoint: {azure_endpoint}")
            # Azure OpenAI doesn't require API key validation here (uses Entra ID)
        elif openai_api_key:
            logger.info("Using standard OpenAI API")
            # Validate OpenAI API key
            required_env_vars = ['OPENAI_API_KEY']
            env_status = validate_environment_variables(required_env_vars)
            if not all(env_status.values()):
                logger.error("Missing required environment variables. Please set OPENAI_API_KEY.")
                return 1
    
    # Load configuration
    config = EvalConfig() if not args.config else load_configuration_as_evalconfig(args.config)
    
    try:
        if args.command == 'evaluate':
            return run_evaluation(args, config)
        elif args.command == 'batch':
            return run_batch_evaluation(args, config)
        elif args.command == 'excel':
            return run_excel_evaluation(args, config)
        elif args.command == 'compare':
            return run_comparison(args, config)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


def add_common_output_arguments(parser) -> None:
    """Add common output-related arguments to a parser."""
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'report'],
        default='json',
        help='Output format (default: json)'
    )


def add_common_file_arguments(parser) -> None:
    """Add common file-related arguments to a parser."""
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: auto-generated)'
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Academic RAG vs Agentic AI Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate individual answers
  python -m rag_agentic_evaluation evaluate "What is machine learning?" "Answer 1" "Answer 2"
  
  # Batch evaluation from file
  python -m rag_agentic_evaluation batch --input data.json --output results.json
  
  # Compare RAG vs Agentic systems
  python -m rag_agentic_evaluation compare --rag-file rag_results.json --agentic-file agentic_results.json
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='rag_agentic_evaluation.log',
        help='Log file path (default: rag_agentic_evaluation.log)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate individual answers'
    )
    eval_parser.add_argument(
        'query',
        type=str,
        help='User question to evaluate'
    )
    eval_parser.add_argument(
        'answers',
        type=str,
        nargs='+',
        help='List of candidate answers to evaluate'
    )
    eval_parser.add_argument(
        '--system-types',
        type=str,
        nargs='*',
        choices=['rag', 'agentic', 'hybrid'],
        help='System types corresponding to each answer'
    )
    add_common_output_arguments(eval_parser)
    
    # Batch command
    batch_parser = subparsers.add_parser(
        'batch',
        help='Batch evaluation from file'
    )
    batch_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input file path (JSON format)'
    )
    add_common_output_arguments(batch_parser)
    
    # Excel command - NEW
    excel_parser = subparsers.add_parser(
        'excel',
        help='Evaluate from Excel file with snippet support'
    )
    excel_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input Excel file path'
    )
    excel_parser.add_argument(
        '--question-col',
        type=str,
        default='Question',
        help='Question column name (default: Question)'
    )
    excel_parser.add_argument(
        '--answer-col',
        type=str,
        default='Answer',
        help='Answer column name (default: Answer)'
    )
    excel_parser.add_argument(
        '--snippet-col',
        type=str,
        default='Snippet',
        help='Snippet column name (default: Snippet)'
    )
    excel_parser.add_argument(
        '--system-type-col',
        type=str,
        help='Optional system type column name'
    )
    excel_parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate Excel structure without evaluation'
    )
    add_common_output_arguments(excel_parser)
    
    # Compare command
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare RAG vs Agentic systems'
    )
    compare_parser.add_argument(
        '--rag-file',
        type=str,
        required=True,
        help='RAG evaluation results file'
    )
    compare_parser.add_argument(
        '--agentic-file',
        type=str,
        required=True,
        help='Agentic evaluation results file'
    )
    add_common_output_arguments(compare_parser)
    compare_parser.add_argument(
        '--report-type',
        choices=['academic', 'summary', 'detailed'],
        default='academic',
        help='Type of report to generate (default: academic)'
    )
    
    return parser


@handle_exceptions("Failed to load configuration, using defaults", EvalConfig())
def load_configuration_as_evalconfig(config_path: str) -> EvalConfig:
    """Load configuration from JSON file and create EvalConfig."""
    config_data = load_json_file(config_path)
    return EvalConfig(
        model_name=config_data.get("model_name", "gpt-4o"),
        temperature=config_data.get("temperature", 0.1),
        max_retries=config_data.get("max_retries", 3),
        confidence_threshold=config_data.get("confidence_threshold", 0.7)
    )


@handle_exceptions("Failed to load configuration", raise_on_error=True)
def load_configuration(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_data = load_json_file(config_path)
    logger.info(f"Loaded configuration from: {config_path}")
    return config_data


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "model_name": "gpt-4",
        "temperature": 0.3,
        "max_retries": 3,
        "confidence_threshold": 0.7,
        "dimensions": [
            "factual_accuracy",
            "relevance", 
            "completeness",
            "clarity",
            "citation_quality"
        ]
    }


def run_evaluation(args, config: EvalConfig) -> int:
    """Run individual answer evaluation."""
    logger.info(f"Evaluating {len(args.answers)} answers for query: {args.query[:100]}...")
    
    # Parse system types if provided
    system_types = None
    if args.system_types:
        if len(args.system_types) != len(args.answers):
            logger.error("Number of system types must match number of answers")
            return 1
        system_types = [SystemType(st) for st in args.system_types]
    
    # Perform evaluation
    evaluations = evaluate_answers(args.query, args.answers, config, system_types)
    
    if not evaluations:
        logger.error("No successful evaluations")
        return 1
    
    # Generate output
    output_dir = ensure_directory(args.output_dir)
    return save_evaluation_results(evaluations, args.format, output_dir, "evaluation_results")


@log_operation("excel evaluation")
@handle_exceptions("Excel evaluation failed", 1)
def run_excel_evaluation(args, config: EvalConfig) -> int:
    """Run evaluation from Excel file with snippet support."""
    logger.info(f"Running Excel evaluation from: {args.input}")
    
    # Validate Excel structure if requested
    if args.validate_only:
        validation_result = validate_excel_structure(
            args.input, 
            args.question_col, 
            args.answer_col, 
            args.snippet_col
        )
        
        logger.info(f"Excel validation completed:")
        logger.info(f"  Total rows: {validation_result['total_rows']}")
        logger.info(f"  Valid rows: {validation_result['valid_rows']}")
        logger.info(f"  Missing columns: {validation_result['missing_columns']}")
        
        if validation_result['snippet_statistics']:
            stats = validation_result['snippet_statistics']
            logger.info(f"  Snippet statistics:")
            logger.info(f"    Rows with snippets: {stats['total_with_snippets']}")
            logger.info(f"    Numbered format: {stats['numbered_format_count']}")
            logger.info(f"    Average length: {stats['average_length']:.1f}")
        
        if validation_result['errors']:
            logger.error("Validation errors:")
            for error in validation_result['errors']:
                logger.error(f"  - {error}")
            return 1
        
        logger.info("Excel file validation passed!")
        return 0
    
    # Load evaluation inputs from Excel
    try:
        evaluation_inputs = load_excel_with_snippets(
            args.input,
            args.question_col,
            args.answer_col,
            args.snippet_col,
            args.system_type_col
        )
    except Exception as e:
        logger.error(f"Failed to load Excel file: {e}")
        return 1
    
    if not evaluation_inputs:
        logger.error("No valid evaluation data found in Excel file")
        return 1
    
    # Perform evaluation with snippet support
    evaluations = evaluate_answers_with_snippets(evaluation_inputs, config)
    
    if not evaluations:
        logger.error("No successful evaluations")
        return 1
    
    # Generate output
    output_dir = ensure_directory(args.output_dir)
    base_name = args.output if args.output else f"{Path(args.input).stem}_evaluation_results"
    
    # Save in requested format
    result = save_evaluation_results(evaluations, args.format, output_dir, base_name)
    
    # Also save Excel format with snippet analysis
    excel_output_path = output_dir / f"{base_name}.xlsx"
    try:
        # Convert evaluations to dictionary format for Excel export
        evaluation_dicts = []
        for eval_result in evaluations:
            eval_dict = {
                "question": evaluation_inputs[evaluations.index(eval_result)].query,
                "answer": eval_result.answer,
                "system_type": eval_result.system_type.value if eval_result.system_type else "",
                "scores": {dim.value: score for dim, score in eval_result.scores.items()},
                "weighted_total": eval_result.weighted_total,
                "raw_total": eval_result.raw_total,
                "snippet_grounding_score": eval_result.snippet_grounding_score,
                "source_snippets": eval_result.source_snippets or [],
                "overall_assessment": eval_result.overall_assessment
            }
            evaluation_dicts.append(eval_dict)
        
        export_evaluation_results_to_excel(evaluation_dicts, str(excel_output_path))
        logger.info(f"Excel results saved to: {excel_output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save Excel results: {e}")
    
    return result


@log_operation("batch evaluation")
@handle_exceptions("Batch evaluation failed", 1)
def run_batch_evaluation(args, config: EvalConfig) -> int:
    """Run batch evaluation from file."""
    logger.info(f"Running batch evaluation from: {args.input}")
    
    evaluations = batch_evaluate_from_file(args.input, config)
    
    if not evaluations:
        logger.error("No successful evaluations in batch")
        return 1
    
    # Generate output
    output_path = args.output if args.output else f"{Path(args.input).stem}_results"
    output_dir = ensure_directory(args.output_dir)
    
    return save_evaluation_results(evaluations, args.format, output_dir, output_path)


@log_operation("excel evaluation")
@handle_exceptions("Excel evaluation failed", 1)
def run_excel_evaluation(args, config: EvalConfig) -> int:
    """Run evaluation from Excel file with snippet support."""
    logger.info(f"Running Excel evaluation from: {args.input}")
    
    # Validate Excel structure if requested
    if args.validate_only:
        validation_result = validate_excel_structure(
            args.input, 
            args.question_col, 
            args.answer_col, 
            args.snippet_col
        )
        
        logger.info(f"Excel validation completed:")
        logger.info(f"  Total rows: {validation_result['total_rows']}")
        logger.info(f"  Valid rows: {validation_result['valid_rows']}")
        logger.info(f"  Missing columns: {validation_result['missing_columns']}")
        
        if validation_result['snippet_statistics']:
            stats = validation_result['snippet_statistics']
            logger.info(f"  Snippet statistics:")
            logger.info(f"    Rows with snippets: {stats['total_with_snippets']}")
            logger.info(f"    Numbered format: {stats['numbered_format_count']}")
            logger.info(f"    Average length: {stats['average_length']:.1f}")
        
        if validation_result['errors']:
            logger.error("Validation errors:")
            for error in validation_result['errors']:
                logger.error(f"  - {error}")
            return 1
        
        logger.info("Excel file validation passed!")
        return 0
    
    # Load evaluation inputs from Excel
    try:
        evaluation_inputs = load_excel_with_snippets(
            args.input,
            args.question_col,
            args.answer_col,
            args.snippet_col,
            args.system_type_col
        )
    except Exception as e:
        logger.error(f"Failed to load Excel file: {e}")
        return 1
    
    if not evaluation_inputs:
        logger.error("No valid evaluation data found in Excel file")
        return 1
    
    # Perform evaluation with snippet support
    evaluations = evaluate_answers_with_snippets(evaluation_inputs, config)
    
    if not evaluations:
        logger.error("No successful evaluations")
        return 1
    
    # Generate output
    output_dir = ensure_directory(args.output_dir)
    base_name = args.output if args.output else f"{Path(args.input).stem}_evaluation_results"
    
    # Save in requested format
    result = save_evaluation_results(evaluations, args.format, output_dir, base_name)
    
    # Also save Excel format with snippet analysis
    excel_output_path = output_dir / f"{base_name}.xlsx"
    try:
        # Convert evaluations to dictionary format for Excel export
        evaluation_dicts = []
        for eval_result in evaluations:
            eval_dict = {
                "question": evaluation_inputs[evaluations.index(eval_result)].query,
                "answer": eval_result.answer,
                "system_type": eval_result.system_type.value if eval_result.system_type else "",
                "scores": {dim.value: score for dim, score in eval_result.scores.items()},
                "weighted_total": eval_result.weighted_total,
                "raw_total": eval_result.raw_total,
                "snippet_grounding_score": eval_result.snippet_grounding_score,
                "source_snippets": eval_result.source_snippets or [],
                "overall_assessment": eval_result.overall_assessment
            }
            evaluation_dicts.append(eval_dict)
        
        export_evaluation_results_to_excel(evaluation_dicts, str(excel_output_path))
        logger.info(f"Excel results saved to: {excel_output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save Excel results: {e}")
    
    return result


@log_operation("system comparison")
@handle_exceptions("Comparison failed", 1)
def run_comparison(args, config: EvalConfig) -> int:
    """Run comparison between RAG and Agentic systems."""
    logger.info(f"Comparing systems: RAG={args.rag_file}, Agentic={args.agentic_file}")
    
    # Load evaluation results
    # Note: This assumes the files contain evaluation results in a specific format
    # You would need to implement proper loading based on your file format
    
    # For now, create placeholder data structure
    # In practice, you'd load from the files and convert to AnswerEvaluation objects
    logger.warning("Comparison functionality requires implementation of result file loading")
    
    # Placeholder for comparison logic
    logger.info("Comparison completed (placeholder)")
    return 0


@handle_exceptions("Failed to save results", 1)
def save_evaluation_results(
    evaluations: List,
    format_type: str,
    output_dir: Path,
    base_name: str
) -> int:
    """Save evaluation results in the specified format."""
    if format_type == 'json':
        output_path = output_dir / f"{base_name}.json"
        json_data = generate_json_export(evaluations)
        save_json_file(json_data, output_path)
        logger.info(f"Results saved to: {output_path}")
        
    elif format_type == 'csv':
        output_path = output_dir / f"{base_name}.csv"
        csv_data = generate_csv_export(evaluations)
        write_text_file(csv_data, output_path)
        logger.info(f"Results saved to: {output_path}")
        
    elif format_type == 'report':
        output_path = output_dir / f"{base_name}.md"
        report = generate_academic_citation_report(evaluations)
        write_text_file(report, output_path)
        logger.info(f"Report saved to: {output_path}")
        
    # Always save JSON for programmatic access
    if format_type != 'json':
        json_path = output_dir / f"{base_name}.json"
        json_data = generate_json_export(evaluations)
        save_json_file(json_data, json_path)
    
    return 0


def create_example_config() -> Dict[str, Any]:
    """Create an example configuration file."""
    return {
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 2000,
        "dimensions": [
            "factual_accuracy",
            "relevance", 
            "completeness",
            "clarity",
            "citation_quality"
        ],
        "weights": {
            "factual_accuracy": 0.25,
            "relevance": 0.20,
            "completeness": 0.15,
            "clarity": 0.13,
            "citation_quality": 0.07
        }
    }


if __name__ == "__main__":
    exit(main())
