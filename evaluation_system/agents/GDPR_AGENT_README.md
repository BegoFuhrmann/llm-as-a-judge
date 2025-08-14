# GDPR Compliance Agent

A specialized RAG agent for assessing GDPR compliance within the RAG vs Agentic AI Evaluation Framework.

## Overview

The `GDPRComplianceAgent` extends the base `RAGAgent` to provide comprehensive assessment capabilities for GDPR compliance across key regulatory requirements and data protection principles.

## Key Features

### 🎯 Core GDPR Assessment Areas

1. **Lawfulness of Processing (Article 6)**
   - Validates legal basis for processing
   - Checks consent mechanisms
   - Evaluates legitimate interests

2. **Data Minimization (Article 5(1)(c))**
   - Assesses necessity of data collection
   - Evaluates relevance to purpose
   - Checks limitation principles

3. **Purpose Limitation (Article 5(1)(b))**
   - Validates specified purposes
   - Ensures explicit communication
   - Confirms legitimate use

4. **Right to Explanation (Recital 71)**
   - Evaluates automated decision-making
   - Checks explanation provisions
   - Validates human intervention options

5. **Data Subject Rights (Articles 12-22)**
   - Assesses access rights implementation
   - Validates rectification procedures
   - Checks erasure mechanisms

6. **General Principle for Transfers (Article 44)**
   - Evaluates international transfer basis
   - Checks adequacy decisions
   - Validates safeguards

### 📊 Assessment Capabilities

- **Comprehensive Scoring**: 0-100% compliance scores per area
- **Risk Classification**: Low, Medium, High, Critical risk levels
- **Automated Recommendations**: Context-aware improvement suggestions
- **Risk Indicators**: Identification of compliance gaps and violations
- **Detailed Reporting**: Structured compliance reports

## Installation & Setup

### Prerequisites

```python
# Required dependencies
pip install langchain-openai
pip install langchain-huggingface  
pip install langchain-chroma
pip install chromadb
pip install python-dotenv
pip install azure-identity
```

### Environment Configuration

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# Optional: Proxy settings
NO_PROXY=localhost,127.0.0.1,::1
```

## Usage Examples

### Basic Usage

```python
from evaluation_system.agents import GDPRComplianceAgent

# Initialize agent
agent = GDPRComplianceAgent(db_path="path/to/gdpr/knowledge/base")

if agent.initialize():
    # Assess compliance for a scenario
    scenario = """
    Company processes customer emails for marketing with pre-checked consent.
    Data is shared with US partners without adequate safeguards.
    """
    
    results = agent.assess_gdpr_compliance(scenario)
    print(f"Compliance Score: {results['compliance_score']}%")
    print(f"Risk Level: {results['risk_level']}")
```

### Specific Area Assessment

```python
# Assess specific GDPR areas
lawfulness_result = agent.assess_lawfulness_of_processing(scenario)
minimization_result = agent.assess_data_minimization(scenario)
rights_result = agent.assess_data_subject_rights(scenario)

# Focus on specific area
focused_assessment = agent.assess_gdpr_compliance(
    scenario, 
    focus_area="right_to_explanation"
)
```

### Generate Compliance Report

```python
# Generate formatted compliance report
assessment = agent.assess_gdpr_compliance(scenario)
report = agent.generate_compliance_report(assessment)
print(report)

# Save report to file
with open("gdpr_compliance_report.md", "w") as f:
    f.write(report)
```

### Get GDPR Guidance

```python
# Get specific GDPR guidance
guidance = agent.get_gdpr_guidance("data subject consent requirements")
print(guidance['answer'])
```

## API Reference

### Core Methods

#### `assess_gdpr_compliance(context: str, focus_area: str = None) -> Dict[str, Any]`

Performs comprehensive GDPR compliance assessment.

**Parameters:**
- `context`: Description of data processing scenario
- `focus_area`: Optional specific area to focus on

**Returns:**
- Dictionary with assessment results, scores, and recommendations

#### `assess_lawfulness_of_processing(context: str) -> Dict[str, Any]`

Specific assessment for Article 6 compliance.

#### `assess_data_minimization(context: str) -> Dict[str, Any]`

Specific assessment for Article 5(1)(c) compliance.

#### `assess_purpose_limitation(context: str) -> Dict[str, Any]`

Specific assessment for Article 5(1)(b) compliance.

#### `assess_right_to_explanation(context: str) -> Dict[str, Any]`

Specific assessment for Recital 71 compliance.

#### `assess_data_subject_rights(context: str) -> Dict[str, Any]`

Specific assessment for Articles 12-22 compliance.

#### `assess_transfer_principles(context: str) -> Dict[str, Any]`

Specific assessment for Article 44 compliance.

#### `generate_compliance_report(results: Dict[str, Any]) -> str`

Generates formatted compliance report from assessment results.

#### `get_gdpr_guidance(topic: str) -> Dict[str, Any]`

Retrieves specific GDPR guidance on a topic.

### Assessment Result Structure

```python
{
    "overall_assessment": {
        "compliance_status": "Moderate compliance - several areas need attention",
        "key_strengths": ["lawfulness: Article 6 - Lawfulness of processing"],
        "critical_gaps": ["data_minimization: Article 5(1)(c) - Data minimisation"],
        "priority_actions": ["Implement proper consent mechanism", ...],
        "regulatory_risk": "medium"
    },
    "detailed_assessments": {
        "lawfulness": {
            "area": "lawfulness",
            "gdpr_reference": "Article 6 - Lawfulness of processing",
            "assessment_criteria": ["legal_basis_identified", "basis_appropriate", "documented_properly"],
            "detailed_analysis": "...",
            "sources": [...],
            "compliance_score": 75,
            "recommendations": [...],
            "risk_indicators": [...]
        },
        ...
    },
    "recommendations": ["Implement clear consent mechanisms", ...],
    "compliance_score": 68.5,
    "risk_level": "medium"
}
```

## Testing

Run the test suite to validate functionality:

```bash
python test_gdpr_agent.py
```

The test suite includes:
- Basic functionality tests
- Unit tests for all core methods
- Score calculation validation
- Risk assessment verification
- Report generation testing

## Configuration

### Agent Parameters

```python
agent = GDPRComplianceAgent(
    db_path="path/to/gdpr/db",           # Path to knowledge base
)

# Customize scoring and retrieval
agent.max_retrieval_results = 5          # Number of documents to retrieve
agent.temperature = 0                    # LLM temperature for consistency
```

### Assessment Customization

The agent uses predefined assessment templates that can be customized:

```python
# Access assessment templates
print(agent.assessment_templates['lawfulness'])

# Modify scoring parameters
agent.max_retrieval_results = 10         # Retrieve more documents
```

## Integration with Evaluation System

### With Excel Processing

```python
from evaluation_system import ExcelProcessor
from evaluation_system.agents import GDPRComplianceAgent

# Process evaluation questions
processor = ExcelProcessor("evaluation_questions.xlsx")
agent = GDPRComplianceAgent(db_path="gdpr_kb")

for question in processor.get_questions():
    assessment = agent.assess_gdpr_compliance(question['context'])
    # Process results...
```

### With Agentic System

```python
from agentic_system import RAGTopicIntegration
from evaluation_system.agents import GDPRComplianceAgent

# Integrate with topic-based system
integration = RAGTopicIntegration()
gdpr_agent = GDPRComplianceAgent(db_path="gdpr_kb")

# Use in evaluation pipeline
results = integration.evaluate_with_agent(
    questions=evaluation_questions,
    agent=gdpr_agent
)
```

## Knowledge Base Setup

The agent requires a ChromaDB knowledge base with GDPR-related documents:

```python
# Create GDPR knowledge base (example)
from create_regulatory_chroma_dbs import create_gdpr_db

create_gdpr_db(
    source_documents="gdpr_documents/",
    output_path="data/chroma_dbs/gdpr_compliance"
)
```

## Best Practices

### 1. Knowledge Base Quality
- Use official GDPR text and authoritative guidance
- Include case studies and practical examples
- Regular updates with new interpretations

### 2. Assessment Context
- Provide detailed scenario descriptions
- Include technical and organizational measures
- Specify data types and processing purposes

### 3. Results Interpretation
- Consider scores in context of risk tolerance
- Review recommendations for practical applicability
- Validate findings against legal requirements

### 4. Continuous Improvement
- Monitor assessment accuracy
- Update templates based on new requirements
- Incorporate feedback from legal experts

## Limitations

- **Knowledge Base Dependency**: Accuracy depends on quality of underlying knowledge base
- **Context Sensitivity**: Requires detailed scenario descriptions for accurate assessment
- **Legal Interpretation**: Results should be validated by legal professionals
- **Jurisdiction Specificity**: Focused on EU GDPR requirements

## Contributing

To extend the GDPR compliance agent:

1. Add new assessment areas in `gdpr_principles`
2. Create corresponding assessment templates
3. Implement specific assessment methods
4. Add unit tests for new functionality
5. Update documentation

## License

This module is part of the RAG vs Agentic AI Evaluation Framework and follows the same licensing terms.

## Support

For issues or questions:
1. Check the test suite for examples
2. Review the base `RAGAgent` documentation
3. Examine the evaluation system integration guides
4. Consult the main project README
