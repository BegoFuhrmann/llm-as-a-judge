"""
Data models for the Agentic System.

This module defines the core data structures used for communication
between agents and data representation throughout the system.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import json


class AgentType(Enum):
    """Types of agents in the system."""
    DOCUMENT_COLLECTION = "document_collection"
    INFORMATION_EXTRACTION = "information_extraction"
    DECISION_COORDINATION = "decision_coordination"


class MessageType(Enum):
    """Types of messages between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class ExtractionStatus(Enum):
    """Status of information extraction."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    sender: AgentType
    recipient: AgentType
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "sender": self.sender.value,
            "recipient": self.recipient.value,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            sender=AgentType(data["sender"]),
            recipient=AgentType(data["recipient"]),
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data["message_id"],
            correlation_id=data.get("correlation_id")
        )


@dataclass
class DocumentMetadata:
    """Metadata for processed documents."""
    file_path: Path
    document_type: DocumentType
    file_size: int
    created_at: datetime
    processed_at: Optional[datetime] = None
    page_count: Optional[int] = None
    text_length: Optional[int] = None
    language: Optional[str] = None
    
    # Processing flags
    has_tables: bool = False
    has_images: bool = False
    extraction_successful: bool = False
    
    # Content checksums for change detection
    content_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "file_path": str(self.file_path),
            "document_type": self.document_type.value,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "page_count": self.page_count,
            "text_length": self.text_length,
            "language": self.language,
            "has_tables": self.has_tables,
            "has_images": self.has_images,
            "extraction_successful": self.extraction_successful,
            "content_hash": self.content_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Create metadata from dictionary."""
        return cls(
            file_path=Path(data["file_path"]),
            document_type=DocumentType(data["document_type"]),
            file_size=data["file_size"],
            created_at=datetime.fromisoformat(data["created_at"]),
            processed_at=datetime.fromisoformat(data["processed_at"]) if data.get("processed_at") else None,
            page_count=data.get("page_count"),
            text_length=data.get("text_length"),
            language=data.get("language"),
            has_tables=data.get("has_tables", False),
            has_images=data.get("has_images", False),
            extraction_successful=data.get("extraction_successful", False),
            content_hash=data.get("content_hash")
        )


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from document analysis."""
    entity_type: str
    text: str
    confidence: float
    start_position: int
    end_position: int
    context: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "entity_type": self.entity_type,
            "text": self.text,
            "confidence": self.confidence,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "context": self.context,
            "attributes": self.attributes
        }


@dataclass
class ExtractedRelationship:
    """Represents a relationship between extracted entities."""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relationship_type": self.relationship_type,
            "confidence": self.confidence,
            "context": self.context
        }


@dataclass
class ExtractionResult:
    """Result of information extraction from a document."""
    document_id: str
    extraction_status: ExtractionStatus
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    summary: Optional[str] = None
    key_findings: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert extraction result to dictionary."""
        return {
            "document_id": self.document_id,
            "extraction_status": self.extraction_status.value,
            "entities": [entity.to_dict() for entity in self.entities],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "summary": self.summary,
            "key_findings": self.key_findings,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "error_message": self.error_message
        }


@dataclass
class CoordinationDecision:
    """Decision made by the coordination agent."""
    decision_id: str
    query: str
    participating_agents: List[AgentType]
    agent_responses: Dict[str, Any] = field(default_factory=dict)
    final_decision: Optional[str] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None
    conflicts_detected: List[str] = field(default_factory=list)
    resolution_strategy: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "decision_id": self.decision_id,
            "query": self.query,
            "participating_agents": [agent.value for agent in self.participating_agents],
            "agent_responses": self.agent_responses,
            "final_decision": self.final_decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "conflicts_detected": self.conflicts_detected,
            "resolution_strategy": self.resolution_strategy,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AuditEntry:
    """Audit trail entry for system monitoring."""
    entry_id: str
    agent_type: AgentType
    action: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary."""
        return {
            "entry_id": self.entry_id,
            "agent_type": self.agent_type.value,
            "action": self.action,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id
        }
    
    def to_json(self) -> str:
        """Convert audit entry to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class ProcessingStats:
    """Statistics for document processing and system performance."""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_entities_extracted: int = 0
    total_relationships_extracted: int = 0
    average_processing_time: float = 0.0
    average_confidence_score: float = 0.0
    
    def calculate_success_rate(self) -> float:
        """Calculate processing success rate."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_documents": self.total_documents,
            "processed_documents": self.processed_documents,
            "failed_documents": self.failed_documents,
            "total_entities_extracted": self.total_entities_extracted,
            "total_relationships_extracted": self.total_relationships_extracted,
            "average_processing_time": self.average_processing_time,
            "average_confidence_score": self.average_confidence_score,
            "success_rate": self.calculate_success_rate()
        }
