#!/usr/bin/env python3
"""
Quick evaluation script with corrected dimensions.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rag_agentic_evaluation.excel_processor import load_excel_with_snippets
from rag_agentic_evaluation.llm_client import create_openai_client, get_model_name
from rag_agentic_evaluation.config import EvalConfig
from rag_agentic_evaluation.enums import EvaluationDimension, SystemType
from rag_agentic_evaluation.templates import render_template, SNIPPET_EVALUATION_PROMPT
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_with_corrected_dimensions(query: str, answer: str, snippets=None):
    """Evaluate directly with the correct dimensions that Azure OpenAI returns."""
    
    config = EvalConfig()
    
    # Create the prompt manually
    if snippets:
        prompt = render_template(SNIPPET_EVALUATION_PROMPT, query=query, answer=answer, snippets=snippets)
    else:
        prompt = f"""Evaluate this AI response on these dimensions: factual_accuracy, relevance, completeness, clarity, coherence, citation_quality, provenance.

Query: {query}
Answer: {answer}

Return your evaluation as JSON with scores (1-5), justifications, confidence_scores (0-1), and overall_assessment."""
    
    try:
        client = create_openai_client()
        model_name = get_model_name(config)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": prompt}],
            temperature=config.temperature,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        
        logger.info("Raw response from Azure OpenAI:")
        logger.info(json.dumps(data, indent=2))
        
        # Calculate scores
        scores = data.get("scores", {})
        weighted_total = sum(scores.values()) / len(scores) if scores else 0
        raw_total = sum(scores.values()) if scores else 0
        
        return {
            "weighted_total": weighted_total,
            "raw_total": raw_total,
            "scores": scores,
            "justifications": data.get("justifications", {}),
            "confidence_scores": data.get("confidence_scores", {}),
            "overall_assessment": data.get("overall_assessment", ""),
            "tokens_used": response.usage.total_tokens if response.usage else 0
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None

def main():
    """Main evaluation function."""
    
    logger.info("=== Azure OpenAI Evaluation with Corrected Dimensions ===")
    
    # Load Excel data
    evaluation_inputs = load_excel_with_snippets("test_snippets.xlsx")
    logger.info(f"Loaded {len(evaluation_inputs)} evaluation items")
    
    results = []
    
    for i, eval_input in enumerate(evaluation_inputs, 1):
        logger.info(f"Evaluating item {i}/{len(evaluation_inputs)}: {eval_input.query[:50]}...")
        
        result = evaluate_with_corrected_dimensions(
            query=eval_input.query,
            answer=eval_input.answer,
            snippets=eval_input.source_snippets
        )
        
        if result:
            results.append(result)
            logger.info(f"Evaluation {i} completed successfully")
            logger.info(f"  Weighted Score: {result['weighted_total']:.2f}")
            logger.info(f"  Raw Score: {result['raw_total']:.2f}")
        else:
            logger.error(f"Evaluation {i} failed")
    
    if results:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"corrected_evaluation_results_{timestamp}"
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save as JSON
        with open(f"{output_dir}/evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Create CSV summary
        import csv
        with open(f"{output_dir}/evaluation_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Weighted_Score", "Raw_Score", "Factual_Accuracy", "Relevance", "Completeness", "Clarity", "Coherence", "Citation_Quality", "Provenance"])
            
            for i, result in enumerate(results):
                question = evaluation_inputs[i].query[:100] + "..." if len(evaluation_inputs[i].query) > 100 else evaluation_inputs[i].query
                scores = result["scores"]
                writer.writerow([
                    question,
                    f"{result['weighted_total']:.2f}",
                    f"{result['raw_total']:.2f}",
                    scores.get("factual_accuracy", "N/A"),
                    scores.get("relevance", "N/A"),
                    scores.get("completeness", "N/A"),
                    scores.get("clarity", "N/A"),
                    scores.get("coherence", "N/A"),
                    scores.get("citation_quality", "N/A"),
                    scores.get("provenance", "N/A")
                ])
        
        # Summary statistics
        weighted_scores = [r['weighted_total'] for r in results]
        raw_scores = [r['raw_total'] for r in results]
        
        logger.info(f"=== Evaluation Complete ===")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Average Weighted Score: {sum(weighted_scores)/len(weighted_scores):.2f}")
        logger.info(f"Average Raw Score: {sum(raw_scores)/len(raw_scores):.2f}")
        
        # Print detailed results
        for i, result in enumerate(results, 1):
            logger.info(f"\n--- Result {i} ---")
            logger.info(f"Question: {evaluation_inputs[i-1].query}")
            logger.info(f"Weighted Score: {result['weighted_total']:.2f}")
            for dim, score in result['scores'].items():
                confidence = result['confidence_scores'].get(dim, 'N/A')
                logger.info(f"  {dim}: {score} (confidence: {confidence})")
            logger.info(f"Assessment: {result['overall_assessment'][:200]}...")
    
    else:
        logger.error("No successful evaluations completed")

if __name__ == "__main__":
    main()
