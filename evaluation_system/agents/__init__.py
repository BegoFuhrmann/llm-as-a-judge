"""
Agents module for the RAG vs Agentic AI Evaluation Framework.

This module contains specialized agents for evaluating compliance
and regulatory aspects of AI systems.
"""

from .rag_agent import RAGAgent
from .gdpr_compliance_agent import GDPRComplianceAgent
from .eu_ai_act_agent import EUAIActAgent
from .audit_trail_agent import AuditTrailAgent

__all__ = ['RAGAgent', 'GDPRComplianceAgent', 'EUAIActAgent', 'AuditTrailAgent']
