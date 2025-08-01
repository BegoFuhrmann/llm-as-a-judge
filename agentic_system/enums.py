"""
Core enums and constants for the agentic system architecture.
"""
from enum import Enum
from typing import Dict, Any


class AgentType(Enum):
    """Types of agents in the system."""
    TOPIC_IDENTIFICATION = "topic_identification"
    TASK_ORCHESTRATION = "task_orchestration"
    RAG_BASED = "rag_based"
    AUTONOMOUS = "autonomous"
    COMPLIANCE_MONITORING = "compliance_monitoring"
    AUDIT_OBSERVABILITY = "audit_observability"


class TaskStatus(Enum):
    """Status of tasks in the system."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(Enum):
    """Priority levels for tasks and operations."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ComplianceLevel(Enum):
    """Compliance levels for monitoring."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL_VIOLATION = "critical_violation"


class LogLevel(Enum):
    """Log levels for audit and observability."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SearchScope(Enum):
    """Search scope for autonomous agents."""
    LOCAL = "local"
    INTERNAL_DOCS = "internal_docs"
    WEB_SEARCH = "web_search"
    COMBINED = "combined"


class ResponseFormat(Enum):
    """Response format types."""
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"
    MARKDOWN = "markdown"


# System-wide constants
SYSTEM_CONFIG = {
    "max_retry_attempts": 3,
    "timeout_seconds": 300,
    "audit_retention_days": 90,
    "max_concurrent_agents": 10,
    "chunk_size": 1000,
    "overlap_size": 200
}

# Agent-specific configurations
AGENT_CONFIGS: Dict[AgentType, Dict[str, Any]] = {
    AgentType.TOPIC_IDENTIFICATION: {
        "confidence_threshold": 0.8,
        "max_topics": 5,
        "classification_model": "azure-openai"
    },
    AgentType.TASK_ORCHESTRATION: {
        "max_parallel_tasks": 5,
        "task_timeout": 600,
        "retry_failed_tasks": True
    },
    AgentType.RAG_BASED: {
        "similarity_threshold": 0.7,
        "max_documents": 10,
        "rerank_results": True
    },
    AgentType.AUTONOMOUS: {
        "search_depth": 3,
        "validation_required": True,
        "source_verification": True
    },
    AgentType.COMPLIANCE_MONITORING: {
        "real_time_monitoring": True,
        "alert_threshold": ComplianceLevel.WARNING,
        "auto_remediation": False
    }
}
