"""
Excel processing module for RAG evaluation with snippet support.

This module handles the specific requirements for processing Excel files
containing questions, answers, and RAG snippets according to academic standards.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

from .models import EvaluationInput, SourceSnippet
from .utils import parse_snippets_from_text, validate_snippet_data, handle_exceptions, log_operation
from .enums import SystemType
from .config import logger


@handle_exceptions("Failed to load Excel file", raise_on_error=True)
@log_operation("Excel file loading")
def load_excel_with_snippets(file_path: str, 
                            question_col: str = "Question",
                            answer_col: str = "Answer", 
                            snippet_col: str = "Snippet",
                            system_type_col: Optional[str] = None) -> List[EvaluationInput]:
    """
    Load Excel file containing questions, answers, and snippets for RAG evaluation.
    
    Based on academic standards for RAG evaluation data preparation.
    Supports flexible column naming and automatic data validation.
    
    Args:
        file_path: Path to Excel file
        question_col: Name of question column
        answer_col: Name of answer column  
        snippet_col: Name of snippet column
        system_type_col: Optional column for system type specification
    
    Returns:
        List of EvaluationInput objects ready for evaluation
        
    Raises:
        FileNotFoundError: If Excel file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    # Load Excel file
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")
    
    # Validate required columns
    required_cols = [question_col, answer_col, snippet_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. "
                        f"Available columns: {list(df.columns)}")
    
    # Process each row
    evaluation_inputs = []
    processed_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        try:
            # Extract basic data
            query = str(row[question_col]).strip()
            answer = str(row[answer_col]).strip()
            snippet_text = str(row[snippet_col]).strip()
            
            # Skip empty rows
            if not query or not answer or query == "nan" or answer == "nan":
                logger.warning(f"Skipping row {idx + 1}: Missing question or answer")
                continue
            
            # Parse snippets
            snippets = []
            if snippet_text and snippet_text != "nan":
                snippets = parse_snippets_from_text(snippet_text)
            
            # Determine system type
            system_type = None
            if system_type_col and system_type_col in df.columns:
                sys_type_str = str(row[system_type_col]).strip().lower()
                if sys_type_str in ["rag", "agentic", "hybrid"]:
                    system_type = SystemType(sys_type_str)
            
            # Create evaluation input
            eval_input = EvaluationInput(
                query=query,
                answer=answer,
                system_type=system_type,
                source_snippets=snippets,
                metadata={"row_number": idx + 1, "source_file": str(file_path)}
            )
            
            evaluation_inputs.append(eval_input)
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing row {idx + 1}: {e}")
            error_count += 1
            continue
    
    logger.info(f"Successfully processed {processed_count} rows, "
               f"{error_count} errors encountered")
    
    if not evaluation_inputs:
        raise ValueError("No valid evaluation data found in Excel file")
    
    return evaluation_inputs


@handle_exceptions("Failed to validate Excel structure", raise_on_error=True)
def validate_excel_structure(file_path: str, 
                           question_col: str = "Question",
                           answer_col: str = "Answer", 
                           snippet_col: str = "Snippet") -> Dict[str, Any]:
    """
    Validate Excel file structure for RAG evaluation requirements.
    
    Args:
        file_path: Path to Excel file
        question_col: Expected question column name
        answer_col: Expected answer column name
        snippet_col: Expected snippet column name
    
    Returns:
        Dictionary with validation results and statistics
    """
    file_path = Path(file_path)
    
    validation_result = {
        "valid": False,
        "total_rows": 0,
        "valid_rows": 0,
        "missing_columns": [],
        "column_analysis": {},
        "snippet_statistics": {},
        "errors": []
    }
    
    try:
        df = pd.read_excel(file_path)
        validation_result["total_rows"] = len(df)
        
        # Check for required columns
        required_cols = [question_col, answer_col, snippet_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        validation_result["missing_columns"] = missing_cols
        
        if missing_cols:
            validation_result["errors"].append(f"Missing required columns: {missing_cols}")
            return validation_result
        
        # Analyze each column
        for col in required_cols:
            non_null_count = df[col].notna().sum()
            validation_result["column_analysis"][col] = {
                "non_null_count": int(non_null_count),
                "null_count": int(len(df) - non_null_count),
                "fill_rate": float(non_null_count / len(df))
            }
        
        # Analyze snippet data
        snippet_data = df[snippet_col].dropna()
        if len(snippet_data) > 0:
            # Count rows with numbered snippets
            numbered_snippets = snippet_data.str.contains(r'\d+\.', na=False).sum()
            avg_length = snippet_data.str.len().mean()
            
            validation_result["snippet_statistics"] = {
                "total_with_snippets": int(len(snippet_data)),
                "numbered_format_count": int(numbered_snippets),
                "average_length": float(avg_length),
                "has_html_tags": int(snippet_data.str.contains('<[^>]+>', na=False).sum())
            }
        
        # Count valid rows (all required fields present)
        valid_mask = df[required_cols].notna().all(axis=1)
        validation_result["valid_rows"] = int(valid_mask.sum())
        validation_result["valid"] = validation_result["valid_rows"] > 0
        
        logger.info(f"Excel validation complete: {validation_result['valid_rows']}/{validation_result['total_rows']} valid rows")
        
    except Exception as e:
        validation_result["errors"].append(f"Failed to read Excel file: {e}")
        logger.error(f"Excel validation failed: {e}")
    
    return validation_result


@handle_exceptions("Failed to create Excel sample", raise_on_error=True)
def create_sample_excel(output_path: str, num_samples: int = 3) -> None:
    """
    Create a sample Excel file showing the expected format for RAG evaluation.
    
    Args:
        output_path: Path where to save the sample Excel file
        num_samples: Number of sample rows to create
    """
    sample_data = []
    
    # Sample questions and answers
    samples = [
        {
            "Question": "Where can I park my car?",
            "Answer": "There are several parking options available to you at the new office building. * **Internal parking spaces:** There are 115 internal parking spaces that can be booked flexibly through the ParkHere app. * **External parking spaces:** Available at nearby Überseering 32a parking garage. * **Guest parking spaces:** Free on weekends for all employees.",
            "Snippet": "1. Several <b>cars</b> are <b>parked</b> on both sides of a central lane. The floor is marked with white lines indicating <b>parking</b> spaces.\n2. The image is likely related to <b>parking</b> management or allocation. The presence of the text suggests that the location has designated <b>parking</b> spaces.\n3. This booking is at \"Parkhaus A Ebene 1\" (<b>Parking</b> Garage A, Level 1) for the \"Ganzer Tag\" (Entire Day) and is associated with \"M-PH-223\"."
        },
        {
            "Question": "What are the office hours?",
            "Answer": "The office is open Monday through Friday from 8:00 AM to 6:00 PM. Access cards work 24/7 for employees who need after-hours access.",
            "Snippet": "1. Office hours are typically from 8 AM to 6 PM on weekdays.\n2. Employee access cards provide 24-hour building access for authorized personnel.\n3. Security personnel are on-site during business hours."
        },
        {
            "Question": "How do I book a meeting room?",
            "Answer": "Meeting rooms can be booked through Outlook calendar system. Simply create a new meeting and add the room as a resource. Rooms are available for booking up to 2 weeks in advance.",
            "Snippet": "1. Meeting room booking is done through the Outlook calendar system.\n2. Rooms can be reserved up to 14 days in advance.\n3. Each room has different capacity and equipment specifications."
        }
    ]
    
    # Take only the requested number of samples
    sample_data = samples[:num_samples]
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(sample_data)
    df.to_excel(output_path, index=False, engine='openpyxl')
    
    logger.info(f"Created sample Excel file with {len(sample_data)} rows at: {output_path}")


def export_evaluation_results_to_excel(evaluation_results: List[Dict[str, Any]], 
                                      output_path: str) -> None:
    """
    Export evaluation results back to Excel format with snippet analysis.
    
    Args:
        evaluation_results: List of evaluation result dictionaries
        output_path: Path where to save the results Excel file
    """
    # Flatten results for Excel export
    excel_data = []
    
    for result in evaluation_results:
        row_data = {
            "Question": result.get("question", ""),
            "Answer": result.get("answer", ""),
            "System_Type": result.get("system_type", ""),
            "Factual_Accuracy": result.get("scores", {}).get("factual_accuracy", ""),
            "Relevance": result.get("scores", {}).get("relevance", ""),
            "Completeness": result.get("scores", {}).get("completeness", ""),
            "Clarity": result.get("scores", {}).get("clarity", ""),
            "Citation_Quality": result.get("scores", {}).get("citation_quality", ""),
            "Weighted_Total": result.get("weighted_total", ""),
            "Raw_Total": result.get("raw_total", ""),
            "Snippet_Grounding_Score": result.get("snippet_grounding_score", ""),
            "Snippet_Count": len(result.get("source_snippets", [])),
            "Overall_Assessment": result.get("overall_assessment", "")
        }
        excel_data.append(row_data)
    
    # Create DataFrame and export
    df = pd.DataFrame(excel_data)
    df.to_excel(output_path, index=False, engine='openpyxl')
    
    logger.info(f"Exported {len(excel_data)} evaluation results to: {output_path}")
