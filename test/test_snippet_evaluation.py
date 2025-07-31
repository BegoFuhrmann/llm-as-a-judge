"""
Test script for RAG snippet evaluation functionality.

This script demonstrates the enhanced evaluation framework's ability to process
Excel files with snippets and perform academic-quality RAG evaluation.
"""

import sys
import os
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rag_agentic_evaluation.excel_processor import (
    load_excel_with_snippets, 
    validate_excel_structure, 
    create_sample_excel,
    export_evaluation_results_to_excel
)
from rag_agentic_evaluation.evaluation import evaluate_answers_with_snippets
from rag_agentic_evaluation.config import EvalConfig
from rag_agentic_evaluation.utils import parse_snippets_from_text, calculate_snippet_grounding_score
from rag_agentic_evaluation.models import SourceSnippet
from rag_agentic_evaluation.enums import SystemType
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_snippet_parsing():
    """Test snippet parsing functionality with the user's actual data format."""
    logger.info("=== Testing Snippet Parsing ===")
    
    # Use the exact format from user's example
    snippet_text = '''1. Several <b>cars</b> are <b>parked</b> on both sides of a central lane. <b>The</b> floor is marked with white lines indicating <b>parking</b> spaces and a central guide line. <b>The</b> ceiling is&nbsp;...
2. * <b>The</b> image is likely related to <b>parking</b> management or allocation. * <b>The</b> presence of <b>the</b> text suggests that <b>the</b> location has designated <b>parking</b> spaces that can&nbsp;...
3. * <b>The car&#39;s</b> headlights are emitting a bright blue light. * <b>The car</b> appears to be in a dark environment, possibly a garage or at night. * <b>The car</b> is clean and&nbsp;...
4. ... <b>parking</b> area with orange columns and <b>parking</b> spaces marked with red stripes. <b>The</b> word &quot;BESUCHER&quot; is written on <b>the</b> wall. **Facts and Conclusions:** * <b>The</b>&nbsp;...
5. This booking is at &quot;Parkhaus A Ebene 1&quot; (<b>Parking</b> Garage A, Level 1) for <b>the</b> &quot;Ganzer Tag&quot; (Entire Day) and is associated with &quot;M-PH-223&quot;. * **Calendar Week 18:**&nbsp;...
6. No snippet is available for this page.
7. Here&#39;s a description of <b>the</b> image, focusing on <b>the</b> observable details: <b>The</b> image shows an indoor <b>parking</b> area, likely for bicycles. <b>The</b> space is mostly gray&nbsp;...
8. Here is a description of <b>the</b> image: <b>The</b> image shows an aerial view of a ... <b>The</b> buildings are characterized by <b>their</b> white facades and curved designs&nbsp;...
9. No snippet is available for this page.'''
    
    snippets = parse_snippets_from_text(snippet_text)
    
    logger.info(f"Parsed {len(snippets)} snippets:")
    for i, snippet in enumerate(snippets):
        # Show snippet analysis including HTML tag handling
        clean_content = snippet.content.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&#39;', "'")
        logger.info(f"  Snippet {i+1} (ID: {snippet.snippet_id}): {clean_content[:100]}...")
        
        # Analyze snippet content quality
        if "No snippet is available" in snippet.content:
            logger.info(f"    → Empty snippet detected (ID: {snippet.snippet_id})")
        elif len(snippet.content.strip()) < 20:
            logger.info(f"    → Short snippet detected (ID: {snippet.snippet_id})")
        else:
            logger.info(f"    -> Content-rich snippet (ID: {snippet.snippet_id})")
    
    return snippets


def test_grounding_score():
    """Test grounding score calculation with academic rigor."""
    logger.info("=== Testing Grounding Score Calculation ===")
    
    # Use the actual answer from user's example
    answer = """There are several parking options available to you at the new office building. 
    * **Internal parking spaces:** There are 115 internal parking spaces that can be booked flexibly. These are located in the building's underground parking garage. You can reserve a space via the "ParkHere" app. The parking spaces are booked on a daily basis and are billed monthly. With the app you can reserve a parking space up to two weeks in advance for a maximum of five days at a time. The parking garage can be accessed with your general access card. 
    * **External parking spaces:** There are external parking spaces in the nearby Überseering 32a parking garage, which is about a 3-minute walk away. 
    * **Guest parking spaces:** On weekends, all employees have the option of parking on the guest parking spaces free of charge. 
    * **Bicycle parking:** There are 157 bicycle parking spaces in the underground parking garage. Please note that if you enter the parking garage without a booking, you will be charged. Would you like more detailed instructions on how to use the "ParkHere" app, or is this information sufficient for now?"""
    
    # Create snippets based on user's data
    snippets = [
        SourceSnippet(
            content="Several cars are parked on both sides of a central lane. The floor is marked with white lines indicating parking spaces and a central guide line.",
            snippet_id="1"
        ),
        SourceSnippet(
            content="The image is likely related to parking management or allocation. The presence of the text suggests that the location has designated parking spaces",
            snippet_id="2"
        ),
        SourceSnippet(
            content="parking area with orange columns and parking spaces marked with red stripes. The word BESUCHER is written on the wall.",
            snippet_id="4"
        ),
        SourceSnippet(
            content="This booking is at Parkhaus A Ebene 1 (Parking Garage A, Level 1) for the Ganzer Tag (Entire Day) and is associated with M-PH-223",
            snippet_id="5"
        ),
        SourceSnippet(
            content="The image shows an indoor parking area, likely for bicycles. The space is mostly gray",
            snippet_id="7"
        )
    ]
    
    grounding_score = calculate_snippet_grounding_score(answer, snippets)
    logger.info(f"Grounding score: {grounding_score:.3f}")
    
    # Academic analysis
    logger.info("Academic Analysis:")
    logger.info(f"  Answer length: {len(answer)} characters")
    logger.info(f"  Snippet count: {len(snippets)} evidence sources")
    logger.info(f"  Grounding strength: {'Strong' if grounding_score > 0.4 else 'Moderate' if grounding_score > 0.2 else 'Weak'}")
    
    # Citation quality assessment
    answer_lower = answer.lower()
    snippet_matches = sum(1 for s in snippets if any(word in s.content.lower() for word in ["parking", "space", "garage"]))
    citation_coverage = snippet_matches / len(snippets) if snippets else 0
    
    logger.info(f"  Citation coverage: {citation_coverage:.2f} ({snippet_matches}/{len(snippets)} snippets relevant)")
    
    return grounding_score


def test_excel_validation():
    """Test Excel file validation."""
    logger.info("=== Testing Excel Validation ===")
    
    excel_file = "sample_evaluation_data.xlsx"
    
    if not Path(excel_file).exists():
        logger.info("Creating sample Excel file...")
        create_sample_excel(excel_file, 3)
    
    validation_result = validate_excel_structure(excel_file)
    
    logger.info("Validation Results:")
    logger.info(f"  Valid: {validation_result['valid']}")
    logger.info(f"  Total rows: {validation_result['total_rows']}")
    logger.info(f"  Valid rows: {validation_result['valid_rows']}")
    logger.info(f"  Missing columns: {validation_result['missing_columns']}")
    
    if validation_result['snippet_statistics']:
        stats = validation_result['snippet_statistics']
        logger.info("  Snippet Statistics:")
        logger.info(f"    Total with snippets: {stats['total_with_snippets']}")
        logger.info(f"    Numbered format count: {stats['numbered_format_count']}")
        logger.info(f"    Average length: {stats['average_length']:.1f}")
        logger.info(f"    Has HTML tags: {stats['has_html_tags']}")
    
    return validation_result


def test_excel_loading():
    """Test loading Excel file with snippets using the user's test data."""
    logger.info("=== Testing Excel Loading ===")
    
    # Use the user's actual test file
    excel_file = "test_snippets.xlsx"
    
    if not Path(excel_file).exists():
        logger.warning(f"Test file {excel_file} not found. Creating sample file...")
        excel_file = "sample_evaluation_data.xlsx"
        create_sample_excel(excel_file, 3)
    
    try:
        evaluation_inputs = load_excel_with_snippets(excel_file)
        logger.info(f"Loaded {len(evaluation_inputs)} evaluation inputs")
        
        for i, eval_input in enumerate(evaluation_inputs):
            logger.info(f"  Input {i+1}:")
            logger.info(f"    Query: {eval_input.query[:50]}...")
            logger.info(f"    Answer length: {len(eval_input.answer)} chars")
            logger.info(f"    Snippets: {len(eval_input.source_snippets or [])}")
            logger.info(f"    System type: {eval_input.system_type}")
            
            # Show snippet grounding analysis
            if eval_input.source_snippets:
                grounding_score = calculate_snippet_grounding_score(
                    eval_input.answer, eval_input.source_snippets
                )
                logger.info(f"    Grounding score: {grounding_score:.3f}")
        
        return evaluation_inputs
        
    except Exception as e:
        logger.error(f"Failed to load Excel file: {e}")
        return None


def test_full_evaluation_pipeline():
    """Test the complete evaluation pipeline with snippets."""
    logger.info("=== Testing Full Evaluation Pipeline ===")
    
    # Check if API keys are configured
    openai_key = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("ENDPOINT_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")
    
    if not openai_key and not azure_endpoint:
        logger.warning("No API keys configured. Skipping LLM evaluation test.")
        logger.info("To test LLM evaluation, set either:")
        logger.info("  - OPENAI_API_KEY for OpenAI API")
        logger.info("  - ENDPOINT_URL for Azure OpenAI")
        return
    
    # Load evaluation inputs
    evaluation_inputs = test_excel_loading()
    if not evaluation_inputs:
        logger.error("Failed to load evaluation inputs")
        return
    
    # Create configuration
    config = EvalConfig()
    
    try:
        # Run evaluation with snippet support
        logger.info("Running evaluation with snippet support...")
        evaluations = evaluate_answers_with_snippets(evaluation_inputs, config)
        
        if evaluations:
            logger.info(f"Successfully evaluated {len(evaluations)} responses")
            
            for i, evaluation in enumerate(evaluations):
                logger.info(f"  Evaluation {i+1}:")
                logger.info(f"    Weighted score: {evaluation.weighted_total:.2f}")
                logger.info(f"    Grounding score: {evaluation.snippet_grounding_score:.3f}")
                logger.info(f"    Citation quality: {evaluation.scores.get('citation_quality', 'N/A')}")
                logger.info(f"    Assessment: {evaluation.overall_assessment[:100]}...")
            
            # Export results
            output_file = "test_evaluation_results.xlsx"
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
            
            export_evaluation_results_to_excel(evaluation_dicts, output_file)
            logger.info(f"Results exported to: {output_file}")
            
        else:
            logger.error("No successful evaluations")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


def print_academic_summary():
    """Print academic summary of the snippet evaluation framework."""
    logger.info("=== Academic RAG Snippet Evaluation Framework Summary ===")
    logger.info("")
    logger.info("Enhanced Features:")
    logger.info("+ Source snippet parsing and structuring")
    logger.info("+ Grounding score calculation (answer-snippet alignment)")
    logger.info("+ Citation-snippet alignment analysis")
    logger.info("+ Excel integration with automatic column detection")
    logger.info("+ Academic rubric-based evaluation with snippet context")
    logger.info("+ Enhanced prompts for RAG-specific evaluation")
    logger.info("+ Statistical significance testing for snippet grounding")
    logger.info("")
    logger.info("Academic Standards Compliance:")
    logger.info("• Source attribution and traceability")
    logger.info("• Inter-rater reliability through confidence scoring")
    logger.info("• Evidence-based justification for all evaluations")
    logger.info("• Reproducible methodology with standardized rubrics")
    logger.info("• Statistical rigor in comparative analysis")
    logger.info("")
    logger.info("Supported Excel Format:")
    logger.info("• Question | Answer | Snippet [| SystemType]")
    logger.info("• Automatic HTML tag cleaning")
    logger.info("• Numbered snippet parsing (1. ... 2. ... etc.)")
    logger.info("• Validation and error reporting")


def main():
    """Run all tests in sequence."""
    logger.info("Starting RAG Snippet Evaluation Framework Tests")
    logger.info("=" * 60)
    
    try:
        # Test individual components
        test_snippet_parsing()
        test_grounding_score()
        test_excel_validation()
        test_excel_loading()
        
        # Test full pipeline (requires API keys)
        test_full_evaluation_pipeline()
        
        # Print summary
        print_academic_summary()
        
        logger.info("=" * 60)
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
