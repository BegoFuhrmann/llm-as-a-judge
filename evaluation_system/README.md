# RAG vs Agentic AI Evaluation Framework

A rigorous, academically-grounded framework for evaluating and comparing Retrieval-Augmented Generation (RAG) and Agentic AI systems based on established information retrieval and AI evaluation metrics.

## Overview

This modular framework implements best practices from:
- TREC evaluation methodologies
- BLEU/ROUGE evaluation standards
- Academic peer review criteria
- Information retrieval evaluation metrics

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage

### Command Line Interface

The framework provides three main commands:

#### 1. Evaluate Individual Answers
```bash
python -m rag_agentic_evaluation evaluate "What is machine learning?" "Answer 1" "Answer 2"
```

With system type annotations:
```bash
python -m rag_agentic_evaluation evaluate "What is machine learning?" "Answer 1" "Answer 2" --system-types rag agentic
```

#### 2. Batch Evaluation from File
```bash
python -m rag_agentic_evaluation batch --input data.json --output results.json
```

Input file format:
```json
{
  "evaluations": [
    {
      "query": "What is machine learning?",
      "answers": ["Answer 1", "Answer 2"],
      "system_types": ["rag", "agentic"]
    }
  ]
}
```

#### 3. Compare Systems
```bash
python -m rag_agentic_evaluation compare --rag-file rag_results.json --agentic-file agentic_results.json
```

### Programmatic Usage

```python
from rag_agentic_evaluation import (
    EvalConfig, 
    evaluate_answers, 
    perform_statistical_comparison,
    generate_academic_citation_report
)

# Configure evaluation
config = EvalConfig()

# Evaluate answers
query = "What is machine learning?"
answers = ["ML is...", "Machine learning refers to..."]
evaluations = evaluate_answers(query, answers, config)

# Generate report
report = generate_academic_citation_report(evaluations)
print(report)
```

## Module Structure

- **config.py**: Configuration and setup (comprehensive academic rubrics and evaluation templates)
- **enums.py**: Type definitions and enumerations (SystemType, EvaluationDimension)
- **models.py**: Data models and structures (AnswerEvaluation, EvaluationRubric, ComparisonResult)
- **templates.py**: Template rendering utilities (Jinja2-based prompt generation)
- **llm_client.py**: LLM API interaction (OpenAI integration with retry logic)
- **evaluation.py**: Core evaluation logic (answer scoring, batch processing, validation utilities)
- **statistics.py**: Statistical analysis and comparison (Cohen's d, t-tests, confidence intervals)
- **report.py**: Report generation (Academic, summary, CSV, JSON, LaTeX formats)
- **utils.py**: Utility functions (DRY decorators, file operations, parsing, validation, scoring)
- **main.py**: CLI interface and orchestration (unified patterns, common argument helpers)

## Evaluation Dimensions

The framework evaluates responses across five academic dimensions with normalized weights:

1. **Factual Accuracy**: Accuracy of factual claims against ground truth (Weight: 0.25)
2. **Relevance**: Alignment between response content and query intent (Weight: 0.20)
3. **Completeness**: Coverage of all necessary aspects to fully answer the query (Weight: 0.15)
4. **Clarity**: Readability, organization, and linguistic quality (Weight: 0.13)
5. **Citation Quality**: Quality and appropriateness of source citations (Weight: 0.07)

Each dimension is scored on a 5-point scale with detailed rubrics.

## Statistical Analysis

The framework provides rigorous statistical comparison including:
- Effect size calculations (Cohen's d)
- Statistical significance testing
- Confidence intervals
- Inter-rater reliability metrics (Cohen's kappa, Krippendorff's alpha)

## Report Generation

Multiple report formats are supported:
- **Academic**: Publication-ready results with proper citations
- **Summary**: Quick overview for analysis
- **CSV**: Data export for external analysis
- **JSON**: Programmatic access to results
- **LaTeX**: Tables for academic papers

## Configuration

Create a configuration file to customize evaluation parameters:

```json
{
  "model": "gpt-4",
  "temperature": 0.1,
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
```

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in requirements.txt

## Academic Standards

This framework follows established academic evaluation standards:
- Inter-rater reliability validation
- Statistical significance testing
- Effect size reporting
- Proper confidence intervals
- Academic citation formatting

## License

Academic research use. Please cite appropriately in publications.

## Contributing

This is a research framework. For contributions, please follow academic standards and include proper documentation and testing.
