"""
Utility functions for the RAG vs Agentic AI evaluation framework.

This module contains helper functions that support the main evaluation pipeline
but don't fit into the core logic modules.
"""

import logging
import json
import os
import functools
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING, Callable, TypeVar
from pathlib import Path

from .enums import EvaluationDimension
from .config import logger

# Type variable for generic function wrapping
T = TypeVar('T')

if TYPE_CHECKING:
    from .models import EvaluationRubric, SourceSnippet
    from .advanced_config import SemanticSimilarityMethod

logger = logging.getLogger(__name__)


def handle_exceptions(
    error_message: str, 
    return_value: Any = None, 
    raise_on_error: bool = False,
    log_traceback: bool = False
) -> Callable:
    """
    Decorator to handle common exception patterns with consistent logging.
    
    Args:
        error_message: Base error message to log (will be formatted with exception)
        return_value: Value to return on exception (if not raising)
        raise_on_error: Whether to re-raise the exception after logging
        log_traceback: Whether to include traceback in logs
        
    Returns:
        Decorated function with exception handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{error_message}: {e}", exc_info=log_traceback)
                if raise_on_error:
                    raise
                return return_value
        return wrapper
    return decorator


def log_operation(operation_name: str, log_start: bool = True, log_end: bool = True) -> Callable:
    """
    Decorator to log operation start and completion.
    
    Args:
        operation_name: Name of the operation for logging
        log_start: Whether to log operation start
        log_end: Whether to log operation completion
        
    Returns:
        Decorated function with operation logging
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if log_start:
                logger.info(f"Starting {operation_name}")
            try:
                result = func(*args, **kwargs)
                if log_end:
                    logger.info(f"Completed {operation_name}")
                return result
            except Exception as e:
                logger.error(f"Failed {operation_name}: {e}")
                raise
        return wrapper
    return decorator


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded JSON file: {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise


def save_json_file(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Target file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Successfully saved JSON file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        raise


def write_text_file(content: str, file_path: Union[str, Path], encoding: str = 'utf-8') -> None:
    """
    Write text content to a file with automatic directory creation.
    
    Args:
        content: Text content to write
        file_path: Target file path  
        encoding: File encoding (default: utf-8)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        logger.debug(f"Successfully wrote text file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to write text file {file_path}: {e}")
        raise


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default value if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division by zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length with optional suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of result
        suffix: Suffix to add if text is truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Account for suffix length
    truncate_length = max_length - len(suffix)
    if truncate_length <= 0:
        return suffix[:max_length]
    
    return text[:truncate_length] + suffix


def validate_environment_variables(required_vars: List[str]) -> Dict[str, Optional[str]]:
    """
    Validate that required environment variables are set.
    
    Args:
        required_vars: List of required environment variable names
        
    Returns:
        Dictionary mapping variable names to their values (or None if missing)
    """
    env_status = {}
    missing_vars = []
    
    for var_name in required_vars:
        value = os.getenv(var_name)
        env_status[var_name] = value
        
        if value is None:
            missing_vars.append(var_name)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
    else:
        logger.debug("All required environment variables are set")
    
    return env_status


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        dir_path: Directory path to ensure
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {dir_path}")
    return dir_path


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def calculate_percentage(part: float, total: float) -> float:
    """
    Calculate percentage with safe division.
    
    Args:
        part: Part value
        total: Total value
        
    Returns:
        Percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (part / total) * 100


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    # Characters that are invalid in filenames on most systems
    invalid_chars = '<>:"/\\|?*'
    
    # Replace invalid characters with underscores
    sanitized = filename
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "untitled"
    
    return sanitized


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    
    chunks = []
    for i in range(0, len(lst), chunk_size):
        chunks.append(lst[i:i + chunk_size])
    
    return chunks


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def calculate_weighted_score(scores: Dict[EvaluationDimension, int], 
                           rubrics: Dict[EvaluationDimension, 'EvaluationRubric']) -> float:
    """
    Calculate weighted total score based on academic rubric weights.
    
    Args:
        scores: Dictionary mapping dimensions to scores (1-5)
        rubrics: Dictionary mapping dimensions to rubric definitions
    
    Returns:
        Weighted average score as float
    """
    if not scores or not rubrics:
        return 0.0
    
    total_weighted = sum(scores[dim] * rubrics[dim].weight for dim in scores if dim in rubrics)
    total_weights = sum(rubrics[dim].weight for dim in scores if dim in rubrics)
    
    return safe_divide(total_weighted, total_weights, 0.0)


def validate_evaluation_response(data: Dict[str, Any], dimensions: List[EvaluationDimension]) -> bool:
    """
    Validate LLM response against expected academic evaluation format.
    
    Args:
        data: Parsed JSON response from LLM
        dimensions: Expected evaluation dimensions
    
    Returns:
        True if response is valid, False otherwise
    """
    import logging
    logger = logging.getLogger(__name__)
    
    required_keys = {"scores", "justifications", "confidence_scores"}
    if not all(key in data for key in required_keys):
        logger.error(f"Missing required keys. Expected: {required_keys}, Got: {set(data.keys())}")
        return False
    
    expected_dimensions = {dim.value for dim in dimensions}
    logger.error(f"DEBUG: Expected dimensions: {expected_dimensions}")
    logger.error(f"DEBUG: Config dimensions list: {[dim.value for dim in dimensions]}")
    
    for key in ["scores", "justifications", "confidence_scores"]:
        actual_dims = set(data[key].keys())
        logger.error(f"DEBUG: Actual dims in '{key}': {actual_dims}")
        if actual_dims != expected_dimensions:
            logger.error(f"Dimension mismatch in '{key}'. Expected: {expected_dimensions}, Got: {actual_dims}")
            missing = expected_dimensions - actual_dims
            extra = actual_dims - expected_dimensions
            logger.error(f"Missing dimensions: {missing}")
            logger.error(f"Extra dimensions: {extra}")
            return False
    
    # Validate score ranges
    for dim, score in data["scores"].items():
        if not isinstance(score, int) or not 1 <= score <= 5:
            logger.error(f"Invalid score for dimension '{dim}': {score} (must be int 1-5)")
            return False
    
    # Validate confidence scores
    for dim, conf in data["confidence_scores"].items():
        if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            logger.error(f"Invalid confidence score for dimension '{dim}': {conf} (must be float 0.0-1.0)")
            return False
    
    return True


def parse_file_content(file_path: Union[str, Path], 
                      expected_type: str = "auto") -> Union[List[str], Dict[str, Any]]:
    """
    Parse file content with automatic format detection.
    
    Args:
        file_path: Path to file to parse
        expected_type: Expected content type ("json", "text", "auto")
    
    Returns:
        Parsed content as list or dictionary
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If content format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try JSON first if auto-detection or explicitly requested
        if expected_type in ["json", "auto"]:
            try:
                parsed = json.loads(content)
                logger.debug(f"Successfully parsed JSON from: {file_path}")
                return parsed
            except json.JSONDecodeError:
                if expected_type == "json":
                    raise ValueError(f"Invalid JSON in file: {file_path}")
        
        # Fall back to line-by-line text parsing
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        logger.debug(f"Parsed {len(lines)} lines from: {file_path}")
        return lines
        
    except Exception as e:
        logger.error(f"Failed to parse file {file_path}: {e}")
        raise


def normalize_system_type_name(system_type: str) -> str:
    """
    Normalize system type names to standard format.
    
    Args:
        system_type: Raw system type string
    
    Returns:
        Normalized system type name
    """
    normalized = system_type.lower().strip()
    
    # Handle common variations
    type_mapping = {
        "retrieval-augmented generation": "rag",
        "retrieval_augmented_generation": "rag", 
        "retrieval augmented generation": "rag",
        "agentic ai": "agentic",
        "agentic_ai": "agentic",
        "agent": "agentic",
        "hybrid": "hybrid"
    }
    
    return type_mapping.get(normalized, normalized)


def parse_snippets_from_text(snippet_text: str) -> List['SourceSnippet']:
    """
    Parse snippet text from Excel into structured SourceSnippet objects.
    
    Based on academic standards for source snippet processing in RAG evaluation.
    Handles various snippet formats including numbered lists and HTML-marked text.
    
    Args:
        snippet_text: Raw snippet text from Excel (may contain HTML, numbering, etc.)
    
    Returns:
        List of SourceSnippet objects with parsed content
    """
    # Import here to avoid circular imports
    from .models import SourceSnippet
    
    if not snippet_text or snippet_text.strip() == "":
        return []
    
    snippets = []
    
    # Handle numbered snippet format (1. ... 2. ... etc.)
    if any(f"{i}. " in snippet_text for i in range(1, 20)):
        # Split by numbered patterns
        import re
        pattern = r'(\d+)\.\s*'
        parts = re.split(pattern, snippet_text)
        
        # Group parts (number, content) pairs
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                snippet_id = parts[i]
                content = parts[i + 1].strip()
                
                # Clean up content - remove HTML tags and normalize
                content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
                content = re.sub(r'&[^;]+;', ' ', content)  # Remove HTML entities
                content = re.sub(r'\s+', ' ', content).strip()  # Normalize whitespace
                
                if content and content != "No snippet is available for this page.":
                    snippets.append(SourceSnippet(
                        content=content,
                        snippet_id=snippet_id,
                        position_in_source=int(snippet_id) if snippet_id.isdigit() else None
                    ))
    else:
        # Handle single snippet or unstructured format
        content = snippet_text.strip()
        if content and content != "No snippet is available for this page.":
            # Clean content
            import re
            content = re.sub(r'<[^>]+>', '', content)
            content = re.sub(r'&[^;]+;', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()
            
            snippets.append(SourceSnippet(
                content=content,
                snippet_id="1"
            ))
    
    logger.debug(f"Parsed {len(snippets)} snippets from text")
    return snippets


def calculate_snippet_grounding_score(
    answer: str, 
    snippets: List['SourceSnippet'], 
    use_enhanced_similarity: bool = True,
    similarity_method: Optional['SemanticSimilarityMethod'] = None
) -> float:
    """
    Calculate grounding score based on answer-snippet alignment.
    
    Enhanced implementation with multiple similarity methods.
    Falls back to token overlap if advanced methods are unavailable.
    
    Args:
        answer: The generated answer text
        snippets: List of source snippets
        use_enhanced_similarity: Whether to use enhanced semantic similarity
        similarity_method: Specific similarity method to use (optional)
    
    Returns:
        Grounding score between 0.0 and 1.0
    """
    if not snippets or not answer:
        return 0.0
    
    # Use enhanced semantic similarity if requested and available
    if use_enhanced_similarity:
        try:
            from .semantic_similarity import SemanticSimilarityCalculator
            from .advanced_config import SemanticSimilarityMethod
            
            # Use specified method or default to hybrid
            method = similarity_method or SemanticSimilarityMethod.HYBRID_WEIGHTED
            calculator = SemanticSimilarityCalculator(method)
            
            # Calculate weighted average similarity across all snippets
            total_score = 0.0
            total_weight = 0.0
            
            for snippet in snippets:
                similarity = calculator.calculate_similarity(answer, snippet.content)
                weight = getattr(snippet, 'relevance_score', 1.0)
                total_score += similarity * weight
                total_weight += weight
            
            result = total_score / total_weight if total_weight > 0 else 0.0
            logger.debug(f"Enhanced grounding score: {result:.3f} (method: {method.value})")
            return result
            
        except ImportError as e:
            logger.debug(f"Enhanced similarity not available: {e}, using fallback")
        except Exception as e:
            logger.warning(f"Error in enhanced similarity calculation: {e}, using fallback")
    
    # Fallback to enhanced token overlap method
    logger.debug("Using fallback enhanced token overlap method")
    return _calculate_enhanced_token_overlap(answer, snippets)


def _calculate_enhanced_token_overlap(answer: str, snippets: List['SourceSnippet']) -> float:
    """
    Enhanced token overlap calculation with academic weighting.
    
    This is an improved version of the original token overlap method with:
    - Better tokenization (concepts, bigrams)
    - Precision/Recall/F1 metrics
    - Per-snippet quality assessment
    
    Args:
        answer: The generated answer text
        snippets: List of source snippets
    
    Returns:
        Enhanced grounding score between 0.0 and 1.0
    """
    import re
    
    def enhanced_tokenize(text: str) -> set:
        """Enhanced tokenization with concept extraction."""
        # Basic cleaning
        text = re.sub(r'[^\w\s-]', ' ', text.lower())
        tokens = set()
        
        # Word tokens (filter short words)
        words = [word.strip() for word in text.split() if len(word.strip()) > 2]
        tokens.update(words)
        
        # Bigrams for concept matching
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            tokens.add(bigram)
        
        # Domain-specific concepts
        concept_patterns = [
            r'\b\d+\s*(?:spaces?|minutes?|percent|%|EUR|euro|dollars?)\b',
            r'\b(?:app|application|system|platform|service|booking)\b',
            r'\b(?:parking|garage|building|floor|level|space)\b'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text)
            tokens.update(f"concept_{match.replace(' ', '_')}" for match in matches)
        
        return tokens
    
    answer_tokens = enhanced_tokenize(answer)
    total_overlap = 0
    total_snippet_tokens = 0
    snippet_scores = []
    
    for snippet in snippets:
        snippet_tokens = enhanced_tokenize(snippet.content)
        overlap = len(answer_tokens.intersection(snippet_tokens))
        total_overlap += overlap
        total_snippet_tokens += len(snippet_tokens)
        
        # Per-snippet score
        snippet_score = overlap / max(len(snippet_tokens), 1) if snippet_tokens else 0
        snippet_scores.append(snippet_score)
    
    if total_snippet_tokens == 0:
        return 0.0
    
    # Calculate precision, recall, and F1
    precision = total_overlap / max(len(answer_tokens), 1)
    recall = total_overlap / max(total_snippet_tokens, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-6)
    
    # Snippet quality metrics
    max_snippet_score = max(snippet_scores) if snippet_scores else 0
    avg_snippet_score = sum(snippet_scores) / len(snippet_scores) if snippet_scores else 0
    
    # Combined score with academic weighting
    final_score = (f1_score * 0.6) + (max_snippet_score * 0.25) + (avg_snippet_score * 0.15)
    
    logger.debug(f"Enhanced token overlap - F1: {f1_score:.3f}, Max snippet: {max_snippet_score:.3f}, Final: {final_score:.3f}")
    return min(1.0, final_score)


def validate_snippet_data(data: Dict[str, Any]) -> bool:
    """
    Validate snippet data structure for academic compliance.
    
    Args:
        data: Dictionary containing snippet data
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    # Required fields for snippet evaluation
    required_fields = ["query", "answer"]
    if not all(field in data for field in required_fields):
        return False
    
    # Optional but recommended for RAG evaluation
    if "snippets" in data:
        snippets = data["snippets"]
        if isinstance(snippets, str):
            # Allow string format (will be parsed)
            return True
        elif isinstance(snippets, list):
            # Validate snippet objects
            for snippet in snippets:
                if not isinstance(snippet, dict) or "content" not in snippet:
                    return False
    
    return True


def calculate_weighted_score(scores: Dict[Any, int], rubrics: Dict[Any, Any]) -> float:
    """
    Calculate weighted score based on dimension weights.
    
    Args:
        scores: Dictionary of dimension scores
        rubrics: Dictionary of evaluation rubrics with weights
    
    Returns:
        Weighted total score
    """
    weighted_total = 0.0
    total_weight = 0.0
    
    for dimension, score in scores.items():
        weight = getattr(rubrics.get(dimension, type('obj', (object,), {'weight': 1.0})()), 'weight', 1.0)
        weighted_total += score * weight
        total_weight += weight
    
    return weighted_total / total_weight if total_weight > 0 else 0.0


def validate_evaluation_response(data: Dict[str, Any], dimensions: List[Any]) -> bool:
    """
    Validate LLM evaluation response structure.
    
    Args:
        data: Response data dictionary
        dimensions: Expected evaluation dimensions
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["scores", "justifications", "confidence_scores"]
    if not all(key in data for key in required_keys):
        return False
    
    expected_dimensions = {dim.value if hasattr(dim, 'value') else str(dim) for dim in dimensions}
    
    for key in required_keys:
        if set(data[key].keys()) != expected_dimensions:
            return False
    
    # Validate score ranges
    for dim, score in data["scores"].items():
        if not isinstance(score, int) or not 1 <= score <= 5:
            return False
    
    # Validate confidence scores
    for dim, conf in data["confidence_scores"].items():
        if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            return False
    
    return True
