"""
RAG vs Agentic AI Evaluation Framework
=====================================

A rigorous, academically-grounded framework for evaluating and comparing
Retrieval-Augmented Generation (RAG) and Agentic AI systems based on
established information retrieval and AI evaluation metrics.

This modular framework implements best practices from:
- TREC evaluation methodologies
- BLEU/ROUGE evaluation standards
- Academic peer review criteria
- Information retrieval evaluation metrics

Modules:
    config: Configuration and setup
    enums: Type definitions and enumerations
    models: Data models and structures
    templates: Template rendering utilities
    llm_client: LLM API interaction
    evaluation: Core evaluation logic
    statistics: Statistical analysis and comparison
    report: Report generation
    utils: Utility functions
    main: CLI interface and orchestration
"""

__version__ = "1.0.0"
__author__ = "Academic Research Team"
__email__ = "research@institution.edu"

# Core imports for easy access
from .config import EvalConfig, setup_logging
from .enums import SystemType, EvaluationDimension
from .models import AnswerEvaluation, EvaluationRubric, ComparisonResult
from .evaluation import evaluate_answers, batch_evaluate_from_file
from .statistics import perform_statistical_comparison, calculate_inter_rater_reliability
from .report import (
    generate_academic_citation_report, 
    generate_summary_report,
    generate_csv_export,
    generate_json_export
)
from .utils import (
    save_json_file,
    ensure_directory,
    validate_environment_variables,
    handle_exceptions,
    log_operation
)
from .main import main

# Define what gets imported with "from rag_agentic_evaluation import *"
__all__ = [
    # Configuration
    'EvalConfig',
    'setup_logging',
    
    # Enums and Types
    'SystemType',
    'EvaluationDimension',
    
    # Data Models
    'AnswerEvaluation',
    'EvaluationRubric', 
    'ComparisonResult',
    
    # Core Functions
    'evaluate_answers',
    'batch_evaluate_from_file',
    'perform_statistical_comparison',
    'calculate_inter_rater_reliability',
    
    # Reporting
    'generate_academic_citation_report',
    'generate_summary_report',
    'generate_csv_export',
    'generate_json_export',
    
    # Utilities
    'save_json_file',
    'ensure_directory',
    'validate_environment_variables',
    'handle_exceptions',
    'log_operation',
    
    # CLI
    'main',
]
