# Audit Trail Agent

A specialized agent for evaluating audit trails based on completeness, quality, and compliance with best practices within the RAG vs Agentic AI Evaluation Framework.

## Overview

The `AuditTrailAgent` is designed to analyze audit logs to assess their effectiveness in providing a clear and comprehensive record of system activities. Unlike other agents, it directly processes log files rather than querying a vector database, using its LLM capabilities to evaluate the content.

## Key Features

### 🎯 Core Assessment Areas

1.  **Completeness**
    -   Verifies that all critical system events are logged.
    -   Checks for the presence of essential data fields (e.g., timestamp, user, action).
    -   Identifies potential gaps in the audit trail.

2.  **Quality and Consistency**
    -   Assesses the clarity and detail of log messages.
    -   Evaluates the consistency of the log format.
    -   Checks for excessive noise or irrelevant entries.

3.  **Security and Integrity**
    -   Looks for signs of log tampering or unauthorized access.
    -   Assesses if logs are stored securely.

### 📊 Assessment Capabilities

-   **Direct Log Analysis**: Reads and evaluates raw log files from a specified directory.
-   **Scoring System**: Provides a 0-100 score for each assessment area and an overall score.
-   **Automated Recommendations**: Generates actionable suggestions for improving the audit trail.
-   **Structured Reporting**: Creates detailed assessment reports in a clear, readable format.

## Installation & Setup

### Prerequisites

```bash
# Required dependencies
pip install langchain-openai
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
from evaluation_system.agents import AuditTrailAgent

# Initialize the agent with the path to your audit logs
audit_log_directory = "path/to/your/audit_logs"
agent = AuditTrailAgent(audit_log_path=audit_log_directory)

# Initialize the agent's LLM
if agent.initialize():
    # Perform a comprehensive assessment
    assessment_results = agent.assess_audit_trail()
    
    print(f"Overall Audit Trail Score: {assessment_results['overall_score']}/100")
    
    # Generate and print the report
    report = agent.generate_assessment_report(assessment_results)
    print(report)
```

### Specific Area Assessment

```python
# Focus on a specific area, like 'completeness' or 'quality'
quality_assessment = agent.assess_audit_trail(focus_area="quality")

# Print the detailed analysis for the focused area
print(quality_assessment['detailed_assessments']['quality']['analysis'])
```

## API Reference

### Core Methods

#### `__init__(self, audit_log_path: str, agent_name: str = "Audit Trail Agent")`
Initializes the agent with the path to the audit logs.

#### `initialize(self) -> bool`
Initializes the agent's LLM. Must be called before performing assessments.

#### `assess_audit_trail(self, focus_area: str = None) -> Dict[str, Any]`
Performs a comprehensive assessment of the audit trail. Can be focused on a specific area.

#### `generate_assessment_report(self, assessment_results: Dict[str, Any]) -> str`
Generates a formatted text report from the assessment results.

## Assessment Result Structure

```json
{
    "overall_assessment": {
        "summary_statement": "The audit trail is adequate but has several areas that require attention...",
        "key_strengths": ["Completeness: Strong performance."],
        "critical_weaknesses": ["Security: Needs significant improvement."],
        "priority_actions": ["Implement log integrity checks.", ...]
    },
    "detailed_assessments": {
        "completeness": {
            "area": "completeness",
            "assessment_criteria": ["all_events_logged", ...],
            "analysis": "The logs cover most critical events...",
            "score": 85,
            "recommendations": ["Add logging for user role changes."]
        },
        ...
    },
    "recommendations": ["Implement log integrity checks.", ...],
    "overall_score": 65.0
}
```

## Limitations

-   **LLM Dependency**: The quality of the assessment is dependent on the analytical capabilities of the configured LLM.
-   **Context Sensitivity**: The agent's assessment is based solely on the provided log files and may lack broader system context.
-   **Parsing Accuracy**: The agent expects a JSON-formatted response from the LLM. Failures in parsing can affect results.
-   **Scalability**: For very large volumes of logs, the content sent to the LLM is truncated, which may affect the completeness of the assessment.
