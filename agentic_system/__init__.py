"""
Enhanced Agentic System for Regulatory-Compliant Document Processing
===================================================================

An advanced multi-agent system designed for comparative analysis against RAG systems
in regulated financial services environments, built with Azure OpenAI integration.

This system implements the three-agent architecture specified in your thesis:
- Document Collection Agent: Multi-modal document parsing with LangChain + Chroma
- Information Extraction Agent: Domain-specific entity recognition for financial services  
- Decision Coordination Agent: Multi-agent orchestration with comprehensive audit trails

Key Features:
- Direct integration with existing RAG evaluation framework
- Azure OpenAI with managed identity for enterprise security
- Comprehensive audit trails for regulatory compliance (GDPR, EU AI Act)
- Multi-modal processing (text, tables, images via pdfplumber/PyMuPDF)
- LangChain + Chroma vector indexing for semantic search
- Structured output for academic comparison with RAG systems

Architecture Philosophy:
This system is designed to enable rigorous academic comparison with RAG approaches
by implementing parallel processing capabilities and standardized output formats
compatible with your existing evaluation framework.
"""

__version__ = "2.0.0"
__author__ = "Academic Research Team - Thesis Implementation"

from .core.agentic_config import AgenticConfig, load_config
from .agents.document_collection import DocumentCollectionAgent
from .agents.information_extraction import InformationExtractionAgent  
from .agents.decision_coordination import DecisionCoordinatorAgent
from .core.audit_trail import AuditTrailManager

__all__ = [
    "AgenticConfig",
    "EvaluationConfig", 
    "DocumentCollectionAgent",
    "InformationExtractionAgent",
    "DecisionCoordinationAgent",
    "AuditTrailManager",
    "EvaluationBridge"
]
