"""
Audit Trail Management for Regulatory Compliance
===============================================

Comprehensive audit trail system for:
- GDPR compliance (Article 30 - Records of processing activities)
- EU AI Act compliance (Article 12 - Record keeping)
- Financial services regulatory requirements
- Academic research transparency and reproducibility
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    AGENT_ACTION = "agent_action"
    DATA_PROCESSING = "data_processing"
    DECISION_MADE = "decision_made"
    ERROR_OCCURRED = "error_occurred"
    COMPLIANCE_CHECK = "compliance_check"
    USER_INTERACTION = "user_interaction"


@dataclass
class AuditEvent:
    """
    Individual audit event with comprehensive metadata.
    
    Designed to meet regulatory requirements for financial services
    and provide transparency for academic research.
    """
    event_id: str
    agent_id: str
    event_type: AuditEventType
    action: str
    timestamp: datetime
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    compliance_flags: List[str] = None
    user_context: Optional[str] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = self._generate_event_id()
        if self.compliance_flags is None:
            self.compliance_flags = []
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        content = f"{self.agent_id}_{self.action}_{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['timestamp'] = self.timestamp.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


class AuditTrailManager:
    """
    Manages audit trails for the entire agentic system.
    
    Provides:
    - Centralized audit event logging
    - Compliance reporting
    - Data retention management
    - Academic research data export
    """
    
    def __init__(self, storage_path: str = "./audit_trails"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.events: List[AuditEvent] = []
        self.logger = logging.getLogger("AuditTrailManager")
        
        # Initialize audit trail
        self.logger.info(f"Audit trail manager initialized. Storage: {self.storage_path}")
    
    async def log_event(self, event: AuditEvent) -> str:
        """
        Log an audit event.
        
        Args:
            event: AuditEvent to log
            
        Returns:
            Event ID
        """
        try:
            # Add to in-memory storage
            self.events.append(event)
            
            # Persist to disk
            await self._persist_event(event)
            
            self.logger.debug(f"Audit event logged: {event.event_id}")
            return event.event_id
            
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
            raise
    
    async def _persist_event(self, event: AuditEvent) -> None:
        """Persist audit event to disk."""
        try:
            # Create daily audit file
            date_str = event.timestamp.strftime("%Y-%m-%d")
            audit_file = self.storage_path / f"audit_{date_str}.jsonl"
            
            # Append event to file
            with open(audit_file, 'a', encoding='utf-8') as f:
                json.dump(event.to_dict(), f, ensure_ascii=False)
                f.write('\n')
                
        except Exception as e:
            self.logger.error(f"Failed to persist audit event: {e}")
            raise
    
    async def get_events_by_agent(
        self,
        agent_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Get audit events for a specific agent."""
        filtered_events = [
            event for event in self.events
            if event.agent_id == agent_id
        ]
        
        if start_time:
            filtered_events = [
                event for event in filtered_events
                if event.timestamp >= start_time
            ]
        
        if end_time:
            filtered_events = [
                event for event in filtered_events
                if event.timestamp <= end_time
            ]
        
        return filtered_events
    
    async def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate compliance report for regulatory requirements.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report data
        """
        period_events = [
            event for event in self.events
            if start_date <= event.timestamp <= end_date
        ]
        
        # Analyze compliance indicators
        total_events = len(period_events)
        error_events = len([e for e in period_events if e.error])
        compliance_issues = []
        
        for event in period_events:
            if event.compliance_flags:
                compliance_issues.extend(event.compliance_flags)
        
        # Generate report
        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_events": total_events,
                "error_events": error_events,
                "error_rate": error_events / total_events if total_events > 0 else 0,
                "unique_agents": len(set(e.agent_id for e in period_events)),
                "compliance_issues": len(compliance_issues)
            },
            "compliance_flags": list(set(compliance_issues)),
            "agents_activity": {
                agent_id: len([e for e in period_events if e.agent_id == agent_id])
                for agent_id in set(e.agent_id for e in period_events)
            }
        }
        
        return report
    
    async def export_academic_data(
        self,
        output_path: str,
        anonymize: bool = True
    ) -> str:
        """
        Export audit data for academic research.
        
        Args:
            output_path: Export file path
            anonymize: Whether to anonymize personal data
            
        Returns:
            Path to exported file
        """
        try:
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_events": len(self.events),
                    "anonymized": anonymize
                },
                "events": []
            }
            
            for event in self.events:
                event_data = event.to_dict()
                
                if anonymize:
                    # Remove or hash potentially identifying information
                    if 'user_context' in event_data:
                        event_data['user_context'] = "ANONYMIZED"
                    if event_data.get('input_data'):
                        event_data['input_data'] = self._anonymize_data(event_data['input_data'])
                    if event_data.get('output_data'):
                        event_data['output_data'] = self._anonymize_data(event_data['output_data'])
                
                export_data["events"].append(event_data)
            
            # Write to file
            output_file = Path(output_path)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Academic data exported to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to export academic data: {e}")
            raise
    
    def _anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive data for academic export."""
        if not isinstance(data, dict):
            return data
        
        anonymized = {}
        sensitive_keys = ['name', 'email', 'phone', 'address', 'ssn', 'id']
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                anonymized[key] = "ANONYMIZED"
            elif isinstance(value, dict):
                anonymized[key] = self._anonymize_data(value)
            elif isinstance(value, list):
                anonymized[key] = [
                    self._anonymize_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                anonymized[key] = value
        
        return anonymized
