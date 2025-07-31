"""
Agentic System Package for Academic RAG vs Agentic AI Evaluation Framework

This package implements a multi-agent system architecture using Azure OpenAI
for document processing, information extraction, and decision coordination
in regulated financial services environments.

Components:
- Document Collection Agent: PDF/DOCX parsing with LangChain/Chroma indexing
- Information Extraction Agent: Domain-specific entity recognition and relationship mapping
- Decision Coordination Agent: Multi-agent orchestration with audit trails

Academic Integration:
- Extends the existing rag_agentic_evaluation framework
- Implements evaluation metrics for agentic systems
- Provides comparative analysis capabilities
"""

from .core.config import AgenticConfig
from .core.models import (
    AgentMessage,
    DocumentMetadata,
    ExtractionResult,
    CoordinationDecision
)
from .agents.document_collection import DocumentCollectionAgent
from .agents.information_extraction import InformationExtractionAgent
from .agents.decision_coordination import DecisionCoordinationAgent

__version__ = "1.0.0"
__author__ = "Academic Research Framework"

__all__ = [
    "AgenticConfig",
    "AgentMessage",
    "DocumentMetadata", 
    "ExtractionResult",
    "CoordinationDecision",
    "DocumentCollectionAgent",
    "InformationExtractionAgent",
    "DecisionCoordinationAgent"
]
