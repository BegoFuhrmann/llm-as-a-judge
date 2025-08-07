"""
Template management for the RAG vs Agentic AI Evaluation Framework.

This module handles Jinja2 template loading and rendering for evaluation prompts
and other templated content.
"""

from datetime import datetime
from jinja2 import Template

from .config import EvalConfig


def load_system_prompt(template_str: str, 
                      config: EvalConfig,
                      timestamp: str = None) -> str:
    """
    Load and render the academic evaluation prompt with rubrics.
    
    Args:
        template_str: The Jinja2 template string
        config: Evaluation configuration containing rubrics
        timestamp: Optional timestamp string (defaults to current time)
    
    Returns:
        Rendered prompt string ready for LLM consumption
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    return Template(template_str).render(
        rubrics=config.rubrics,
        dimensions=config.dimensions,
        timestamp=timestamp
    )


def render_template(template_str: str, **kwargs) -> str:
    """
    Generic template rendering function.
    
    Args:
        template_str: The Jinja2 template string
        **kwargs: Template variables
    
    Returns:
        Rendered template string
    """
    return Template(template_str).render(**kwargs)


# Enhanced academic evaluation prompt with snippet support
SNIPPET_EVALUATION_PROMPT = """
You are an expert academic evaluator specializing in Retrieval-Augmented Generation (RAG) systems assessment.

Your task is to evaluate an AI system's response using rigorous academic standards, with special attention to source snippet grounding and citation quality.

## EVALUATION FRAMEWORK

### Core Academic Dimensions (Score 1-5):
{% for dimension in dimensions %}{{ loop.index }}. **{{ dimension.value.replace('_', ' ').title() }}**: {{ rubrics[dimension].description }}
{% endfor %}

### Snippet-Specific Evaluation Criteria:
- **Grounding Score**: How well the answer is supported by provided snippets
- **Citation Alignment**: Accuracy of references to specific snippet content (if present)
- **Source Attribution**: Proper acknowledgment of information sources (if provided)
- **Content Synthesis**: Integration of multiple snippet sources

### Academic Rubric (1-5 scale):
- **5 (Excellent)**: Exceeds academic standards (90-100%)
- **4 (Good)**: Meets high academic standards (80-89%)  
- **3 (Satisfactory)**: Acceptable academic quality (70-79%)
- **2 (Needs Improvement)**: Below academic standards (60-69%)
- **1 (Inadequate)**: Significantly deficient (<60%)

### Special Considerations for Citation Quality:
- **Systems WITHOUT explicit citations**: Focus on content accuracy and implicit grounding in provided snippets
- **Systems WITH explicit citations**: Evaluate both citation presence and accuracy
- **Mixed approaches**: Assess based on system capabilities and context

## INPUT DATA

**User Query**: {{ query }}

**AI Response**: {{ answer }}

{% if snippets and snippets|length > 0 %}
**Source Snippets** ({{ snippets|length }} total):
{% for snippet in snippets %}
**Snippet {{ snippet.snippet_id or loop.index }}**:
{{ snippet.content }}
{% if snippet.source_document %}
*Source: {{ snippet.source_document }}*
{% endif %}

{% endfor %}

### SNIPPET ANALYSIS REQUIRED:
For RAG systems, you must evaluate:
1. How well the answer references and uses the provided snippets
2. Whether claims in the answer are supported by snippet content  
3. Quality of information synthesis across multiple snippets
4. **Citation Quality Assessment**:
   - If the system provides explicit citations/references: Evaluate accuracy and completeness
   - If the system does NOT provide explicit citations: Focus on implicit grounding and content accuracy
   - Rate based on the system's design capabilities, not the absence of features it wasn't designed to provide

**IMPORTANT**: Do not penalize systems for missing explicit citations if they demonstrate strong implicit grounding in the source material.

{% else %}
**Note**: No source snippets provided. Evaluate based on general knowledge and content quality.
{% endif %}

## EVALUATION INSTRUCTIONS

1. **Read Carefully**: Analyze the query, response{% if snippets %}, and all provided snippets{% endif %}
2. **Score Each Dimension**: Assign scores 1-5 based on the academic rubric
3. **Provide Evidence**: Include specific text excerpts supporting each score
4. **Calculate Confidence**: Rate your evaluation confidence (0.0-1.0)
5. **Academic Justification**: Explain reasoning with reference to academic standards

{% if snippets and snippets|length > 0 %}
6. **Snippet Grounding**: Assess how well the answer leverages provided source material
7. **Citation Evaluation**: 
   - For systems WITH explicit citations: Check accuracy of references to snippet content
   - For systems WITHOUT explicit citations: Evaluate implicit grounding and content derivation from snippets
   - Do not penalize systems for architectural design choices (e.g., no explicit citation mechanism)
{% endif %}

## OUTPUT FORMAT (JSON)

You must respond with a valid JSON object in the following format:

{
    "scores": {
        {% for dimension in dimensions %}"{{ dimension.value }}": <score 1-5>{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "justifications": {
        {% for dimension in dimensions %}"{{ dimension.value }}": "<evidence-based justification>"{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "confidence_scores": {
        {% for dimension in dimensions %}"{{ dimension.value }}": <confidence 0.0-1.0>{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "overall_assessment": "<comprehensive evaluation summary>",
    {% if snippets and snippets|length > 0 %}
    "snippet_analysis": {
        "grounding_quality": "<how well answer uses snippets>",
        "citation_approach": "<explicit citations present: true/false>",
        "citation_accuracy": "<if explicit citations: accuracy assessment; if no citations: content grounding assessment>",
        "synthesis_quality": "<quality of multi-snippet integration>"
    },
    {% endif %}
    "academic_recommendation": "<actionable feedback for improvement>"
}

Focus on academic rigor, evidence-based evaluation, and transparent methodology. All scores must be supported by specific textual evidence."""


# Standard evaluation prompt (backwards compatibility)
STANDARD_EVALUATION_PROMPT = """
You are an expert academic evaluator for AI system responses.

Your task is to evaluate the following AI response using rigorous academic standards.

## EVALUATION DIMENSIONS (Score 1-5):
{% for dimension in dimensions %}{{ loop.index }}. **{{ dimension.value.replace('_', ' ').title() }}**: {{ rubrics[dimension].description }}
{% endfor %}

## INPUT DATA

**User Query**: {{ query }}

**AI Response**: {{ answer }}

## EVALUATION RUBRIC (1-5 scale):
- **5 (Excellent)**: Exceeds academic standards (90-100%)
- **4 (Good)**: Meets high academic standards (80-89%)
- **3 (Satisfactory)**: Acceptable academic quality (70-79%)
- **2 (Needs Improvement)**: Below academic standards (60-69%)
- **1 (Inadequate)**: Significantly deficient (<60%)

## OUTPUT FORMAT (JSON)

You must respond with a valid JSON object in the following format:

{
    "scores": {
        {% for dimension in dimensions %}"{{ dimension.value }}": <score 1-5>{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "justifications": {
        {% for dimension in dimensions %}"{{ dimension.value }}": "<evidence-based justification>"{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "confidence_scores": {
        {% for dimension in dimensions %}"{{ dimension.value }}": <confidence 0.0-1.0>{% if not loop.last %},{% endif %}
        {% endfor %}
    },
    "overall_assessment": "<comprehensive evaluation summary>"
}

Focus on academic rigor, evidence-based evaluation, and transparent methodology."""
