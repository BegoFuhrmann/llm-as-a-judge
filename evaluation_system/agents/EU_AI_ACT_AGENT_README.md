# EU AI Act Compliance Agent

A specialized RAG agent for assessing EU AI Act compliance within the RAG vs Agentic AI Evaluation Framework.

## Overview

The `EUAIActAgent` extends the base `RAGAgent` to provide comprehensive assessment capabilities for EU AI Act compliance, focusing on risk categorization, prohibited practices, and requirements for high-risk AI systems.

## Key Features

### 🎯 Core EU AI Act Assessment Areas

1.  **Risk Categorization (Articles 6-15)**
    -   Assesses if an AI system is classified as prohibited, high-risk, limited-risk, or minimal-risk.
    -   Validates justification for the selected risk category.

2.  **Prohibited AI Practices (Article 5)**
    -   Checks for practices like manipulative techniques, exploitation of vulnerabilities, and social scoring.

3.  **High-Risk AI System Requirements (Articles 16-51)**
    -   Evaluates compliance with risk management, data governance, technical documentation, transparency, human oversight, and cybersecurity.

4.  **Transparency Obligations (Article 52)**
    -   Assesses if AI interactions and synthetic content are properly disclosed to users.

5.  **CE Marking (Article 48)**
    -   Checks for conformity assessment and proper CE marking for high-risk systems.

### 📊 Assessment Capabilities

-   **Comprehensive Scoring**: 0-100% compliance scores per area.
-   **Risk Classification**: Low, Medium, High, Critical risk levels.
-   **Automated Recommendations**: Context-aware improvement suggestions.
-   **Risk Indicators**: Identification of compliance gaps and potential violations.
-   **Detailed Reporting**: Structured compliance reports.

## Installation & Setup

### Prerequisites

```bash
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
from evaluation_system.agents import EUAIActAgent

# Initialize agent
agent = EUAIActAgent(db_path="path/to/eu-ai-act/knowledge/base")

if agent.initialize():
    # Define a scenario for a high-risk AI system
    scenario = """
    An AI system used for credit scoring, making decisions on loan applications.
    The model is a complex neural network, and its training data includes demographic information.
    """
    
    results = agent.assess_eu_ai_act_compliance(scenario)
    print(f"Compliance Score: {results['compliance_score']}%")
    print(f"Risk Level: {results['risk_level']}")
```

### Specific Area Assessment

```python
# Assess specific EU AI Act areas
risk_cat_result = agent.assess_risk_categorization(scenario)
high_risk_req_result = agent.assess_high_risk_requirements(scenario)
transparency_result = agent.assess_transparency_obligations(scenario)

# Focus on a specific area
focused_assessment = agent.assess_eu_ai_act_compliance(
    scenario, 
    focus_area="high_risk_requirements"
)
```

### Generate Compliance Report

```python
# Generate a formatted compliance report
assessment = agent.assess_eu_ai_act_compliance(scenario)
report = agent.generate_compliance_report(assessment)
print(report)

# Save report to a file
with open("eu_ai_act_compliance_report.md", "w") as f:
    f.write(report)
```

### Get EU AI Act Guidance

```python
# Get specific guidance on a topic
guidance = agent.get_ai_act_guidance("technical documentation for high-risk AI systems")
print(guidance['answer'])
```

## API Reference

### Core Methods

#### `assess_eu_ai_act_compliance(context: str, focus_area: str = None) -> Dict[str, Any]`
Performs a comprehensive compliance assessment against the EU AI Act.

#### `assess_risk_categorization(context: str) -> Dict[str, Any]`
Specific assessment for risk categorization under Articles 6-15.

#### `assess_prohibited_practices(context: str) -> Dict[str, Any]`
Specific assessment for prohibited AI practices under Article 5.

#### `assess_high_risk_requirements(context: str) -> Dict[str, Any]`
Specific assessment for high-risk AI system requirements under Articles 16-51.

#### `assess_transparency_obligations(context: str) -> Dict[str, Any]`
Specific assessment for transparency obligations under Article 52.

#### `assess_ce_marking_requirements(context: str) -> Dict[str, Any]`
Specific assessment for CE marking requirements under Article 48.

#### `generate_compliance_report(results: Dict[str, Any]) -> str`
Generates a formatted compliance report from assessment results.

#### `get_ai_act_guidance(topic: str) -> Dict[str, Any]`
Retrieves specific guidance on an EU AI Act topic.

## Testing

Run the test suite to validate functionality:

```bash
python test_eu_ai_act_agent.py 
```

## Knowledge Base Setup

The agent requires a ChromaDB knowledge base with EU AI Act-related documents.

```python
# Example of creating the knowledge base
from create_regulatory_chroma_dbs import create_eu_ai_act_db

create_eu_ai_act_db(
    source_documents="eu_ai_act_documents/",
    output_path="data/chroma_dbs/eu_ai_act_compliance"
)
```

## Limitations

-   **Knowledge Base Dependency**: The accuracy of assessments depends heavily on the quality and completeness of the underlying knowledge base.
-   **Legal Interpretation**: The agent's outputs are not a substitute for professional legal advice. Results should be reviewed by qualified legal experts.
-   **Dynamic Regulation**: The AI Act and its interpretations will evolve. The knowledge base requires continuous updates.
