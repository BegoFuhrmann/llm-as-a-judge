"""
Configuration and logging setup for the RAG vs Agentic AI Evaluation Framework.

This module centralizes all configuration settings, default values, and logging
configuration for the evaluation framework.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict

from .enums import EvaluationDimension
from .models import EvaluationRubric


# ─── Logging Configuration ─────────────────────────────────────────────────────
def setup_logging(log_level: int = logging.INFO, log_file: str = "rag_agentic_evaluation.log"):
    """Configure logging for the evaluation framework."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s › %(message)s",
        level=log_level,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger("rag_agentic_evaluator")


# Initialize logger
logger = setup_logging()


# ─── Default System Prompt Template ────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT_TEMPLATE = """
You are a distinguished academic evaluator specializing in artificial intelligence systems evaluation, with expertise equivalent to a senior researcher at a top-tier institution (MIT, Stanford, CMU). Your task is to conduct a rigorous, unbiased evaluation of AI system responses following established academic standards.

## EVALUATION CONTEXT
You are comparing responses from different AI architectures:
- **RAG (Retrieval-Augmented Generation)**: Systems that retrieve relevant documents before generating responses
- **Agentic AI**: Systems that can plan, reason, and use tools autonomously to solve complex problems
- **Hybrid**: Systems combining both approaches

## ACADEMIC EVALUATION PROTOCOL

### Phase 1: Rubric Internalization
For each evaluation dimension, you must apply the following rigorous criteria:

{% for dimension, rubric in rubrics.items() %}
**{{ rubric.dimension.value.upper().replace('_', ' ') }}** (Weight: {{ rubric.weight }})
{{ rubric.description }}

Scoring Scale:
- 5 (Excellent): {{ rubric.criteria_5 }}
- 4 (Good): {{ rubric.criteria_4 }}
- 3 (Satisfactory): {{ rubric.criteria_3 }}
- 2 (Needs Improvement): {{ rubric.criteria_2 }}
- 1 (Inadequate): {{ rubric.criteria_1 }}

{% endfor %}

### Phase 2: Systematic Evaluation Process

1. **Initial Reading**: Read the response completely without scoring
2. **Dimensional Analysis**: Evaluate each dimension independently using the rubric
3. **Evidence Collection**: Identify specific textual evidence supporting each score
4. **Cross-Validation**: Ensure scores are consistent across dimensions
5. **Confidence Assessment**: Rate your confidence in each evaluation (0.0-1.0)

### Phase 3: Academic Rigor Requirements

- **Objectivity**: Evaluate content, not perceived system type
- **Consistency**: Apply identical standards across all responses
- **Evidence-Based**: Ground all judgments in observable textual features
- **Calibration**: Consider the complexity and difficulty of the query
- **Transparency**: Provide clear, specific justifications

### Phase 4: Output Format

Return ONLY a valid JSON object with this exact structure:
{
  "evaluation_metadata": {
    "evaluator_id": "academic_judge_v2.0",
    "evaluation_timestamp": "{{ timestamp }}",
    "query_complexity": "low|medium|high",
    "response_length_assessment": "too_short|appropriate|too_long"
  },
  "scores": {
    "{{ dimension.value }}": integer_score_1_to_5,
    ...
  },
  "justifications": {
    "{{ dimension.value }}": "Specific, evidence-based explanation citing textual examples",
    ...
  },
  "confidence_scores": {
    "{{ dimension.value }}": float_0_to_1,
    ...
  },
  "weighted_total": float,
  "overall_assessment": "Brief summary highlighting key strengths and weaknesses",
  "comparative_notes": "Observations about system capabilities relevant to RAG vs Agentic classification"
}

## CRITICAL INSTRUCTIONS
- Maintain the highest standards of academic integrity
- Do not reveal or guess the system type before evaluation
- Focus on response quality, not architectural assumptions
- Provide specific, actionable feedback in justifications
- Ensure all scores reflect the established rubric criteria
- Include NO text outside the JSON response
"""


# ─── Default Rubrics ───────────────────────────────────────────────────────────
def get_default_rubrics() -> Dict[EvaluationDimension, EvaluationRubric]:
    """Get the default academic rubrics for all evaluation dimensions."""
    return {
        EvaluationDimension.FACTUAL_ACCURACY: EvaluationRubric(
            dimension=EvaluationDimension.FACTUAL_ACCURACY,
            description="Accuracy of factual claims against ground truth or authoritative sources",
            criteria_5="All factual claims are accurate and verifiable (>95% accuracy)",
            criteria_4="Most factual claims accurate with minor inaccuracies (85-95% accuracy)", 
            criteria_3="Generally accurate with some notable errors (75-84% accuracy)",
            criteria_2="Multiple factual errors present (65-74% accuracy)",
            criteria_1="Significant factual inaccuracies or misinformation (<65% accuracy)",
            weight=0.25
        ),
        EvaluationDimension.RELEVANCE: EvaluationRubric(
            dimension=EvaluationDimension.RELEVANCE,
            description="Alignment between response content and query intent",
            criteria_5="Perfectly addresses query intent with no extraneous information",
            criteria_4="Directly addresses query with minimal off-topic content",
            criteria_3="Generally relevant with some tangential information",
            criteria_2="Partially relevant but includes significant off-topic content",
            criteria_1="Minimally relevant or addresses different question entirely",
            weight=0.2
        ),
        EvaluationDimension.COMPLETENESS: EvaluationRubric(
            dimension=EvaluationDimension.COMPLETENESS,
            description="Coverage of all necessary aspects to fully answer the query",
            criteria_5="Comprehensive coverage of all relevant aspects and sub-questions",
            criteria_4="Covers most important aspects with minor gaps",
            criteria_3="Adequate coverage of main points with some omissions",
            criteria_2="Partial coverage leaving important aspects unaddressed",
            criteria_1="Incomplete response missing critical information",
            weight=0.15
        ),
        EvaluationDimension.CLARITY: EvaluationRubric(
            dimension=EvaluationDimension.CLARITY,
            description="Readability, organization, and linguistic quality",
            criteria_5="Exceptionally clear, well-organized, and easily understood",
            criteria_4="Clear and well-structured with good flow",
            criteria_3="Generally clear with minor organizational issues",
            criteria_2="Somewhat unclear or poorly organized",
            criteria_1="Difficult to understand due to poor structure or language",
            weight=0.13
        ),
        EvaluationDimension.CITATION_QUALITY: EvaluationRubric(
            dimension=EvaluationDimension.CITATION_QUALITY,
            description="Quality and appropriateness of source citations and references",
            criteria_5="Excellent citations from authoritative sources, properly formatted",
            criteria_4="Good citations with minor formatting or relevance issues",
            criteria_3="Adequate citations with some quality concerns",
            criteria_2="Poor citation quality or inappropriate sources",
            criteria_1="Missing, invalid, or misleading citations",
            weight=0.07
        ),
    }


@dataclass
class EvalConfig:
    """Enhanced configuration with academic rigor and statistical considerations."""
    model_name: str = os.getenv("EVAL_MODEL", "gpt-4o")
    temperature: float = 0.1  # Low but non-zero for consistency
    max_retries: int = 3
    confidence_threshold: float = 0.7
    inter_rater_agreement_threshold: float = 0.8
    
    # Evaluation dimensions with academic weights
    dimensions: List[EvaluationDimension] = field(default_factory=lambda: [
        EvaluationDimension.FACTUAL_ACCURACY,
        EvaluationDimension.RELEVANCE,
        EvaluationDimension.COMPLETENESS,
        EvaluationDimension.CLARITY,
        EvaluationDimension.CITATION_QUALITY
    ])
    
    # Academic rubrics for each dimension
    rubrics: Dict[EvaluationDimension, EvaluationRubric] = field(default_factory=get_default_rubrics)
    
    system_prompt_template: str = field(default=DEFAULT_SYSTEM_PROMPT_TEMPLATE)


def get_default_config() -> EvalConfig:
    """Get a default evaluation configuration."""
    return EvalConfig()
