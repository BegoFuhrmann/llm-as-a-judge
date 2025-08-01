"""
Comprehensive audit logging and observability system for the agentic framework.
"""
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import uuid

# Handle imports for both module and direct execution
try:
    # Try relative imports first (when used as a module)
    from ..enums import AgentType, LogLevel, ComplianceLevel
    from ..core.base import Task, AgentResponse
except ImportError:
    # Fallback to absolute imports (when run directly)
    import sys
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    
    from enums import AgentType, LogLevel, ComplianceLevel
    from core.base import Task, AgentResponse


@dataclass
class AuditEvent:
    """Comprehensive audit event structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: str = ""
    agent_type: AgentType = AgentType.TASK_ORCHESTRATION
    event_type: str = ""
    log_level: LogLevel = LogLevel.INFO
    message: str = ""
    task_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_details: Optional[str] = None
    compliance_level: ComplianceLevel = ComplianceLevel.COMPLIANT
    metadata: Dict[str, Any] = field(default_factory=dict)
    sensitive_data_masked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['agent_type'] = self.agent_type.value
        data['log_level'] = self.log_level.value
        data['compliance_level'] = self.compliance_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create audit event from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['agent_type'] = AgentType(data['agent_type'])
        data['log_level'] = LogLevel(data['log_level'])
        data['compliance_level'] = ComplianceLevel(data['compliance_level'])
        return cls(**data)


class AuditLogger:
    """Advanced audit logging system with compliance monitoring."""
    
    def __init__(self, log_dir: str = "audit_logs", retention_days: int = 90):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.retention_days = retention_days
        self.events: List[AuditEvent] = []
        self._setup_file_logging()
        
    def _setup_file_logging(self):
        """Set up file-based logging."""
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AuditLogger')
    
    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive information in data."""
        sensitive_keys = ['password', 'token', 'key', 'secret', 'api_key', 'auth']
        masked_data = data.copy()
        
        def mask_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        obj[key] = "***MASKED***"
                    elif isinstance(value, (dict, list)):
                        mask_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        mask_recursive(item)
        
        mask_recursive(masked_data)
        return masked_data
    
    async def log_event(self, event: AuditEvent):
        """Log an audit event."""
        # Mask sensitive data
        event.input_data = self._mask_sensitive_data(event.input_data)
        event.output_data = self._mask_sensitive_data(event.output_data)
        event.sensitive_data_masked = True
        
        # Store in memory
        self.events.append(event)
        
        # Log to file
        self.logger.log(
            getattr(logging, event.log_level.value.upper()),
            f"[{event.agent_type.value}] {event.message} | Task: {event.task_id} | "
            f"Execution: {event.execution_time:.2f}s | Compliance: {event.compliance_level.value}"
        )
        
        # Write detailed event to JSON file
        await self._write_event_to_file(event)
    
    async def _write_event_to_file(self, event: AuditEvent):
        """Write detailed event data to JSON file."""
        event_file = self.log_dir / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(event_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event.to_dict()) + '\n')
    
    async def log_agent_action(self, 
                             agent_id: str,
                             agent_type: AgentType,
                             action: str,
                             task: Optional[Task] = None,
                             response: Optional[AgentResponse] = None,
                             log_level: LogLevel = LogLevel.INFO,
                             **kwargs):
        """Log an agent action with comprehensive details."""
        event = AuditEvent(
            agent_id=agent_id,
            agent_type=agent_type,
            event_type=f"agent_{action}",
            log_level=log_level,
            message=f"Agent {action}: {agent_id}",
            task_id=task.id if task else None,
            input_data=task.input_data if task else {},
            output_data=response.data if response else {},
            execution_time=response.execution_time if response else 0.0,
            error_details=response.error_message if response and not response.success else None,
            compliance_level=self._assess_compliance(response) if response else ComplianceLevel.COMPLIANT,
            metadata=kwargs
        )
        await self.log_event(event)
    
    def _assess_compliance(self, response: AgentResponse) -> ComplianceLevel:
        """Assess compliance level based on response."""
        if not response.success:
            return ComplianceLevel.VIOLATION
        elif response.confidence_score < 0.5:
            return ComplianceLevel.WARNING
        else:
            return ComplianceLevel.COMPLIANT
    
    async def log_system_event(self, 
                             event_type: str,
                             message: str,
                             log_level: LogLevel = LogLevel.INFO,
                             **kwargs):
        """Log a system-level event."""
        event = AuditEvent(
            agent_id="SYSTEM",
            agent_type=AgentType.AUDIT_OBSERVABILITY,
            event_type=event_type,
            log_level=log_level,
            message=message,
            metadata=kwargs
        )
        await self.log_event(event)
    
    def get_events(self, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   agent_id: Optional[str] = None,
                   agent_type: Optional[AgentType] = None,
                   log_level: Optional[LogLevel] = None,
                   compliance_level: Optional[ComplianceLevel] = None) -> List[AuditEvent]:
        """Query audit events with filters."""
        filtered_events = self.events
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        if agent_id:
            filtered_events = [e for e in filtered_events if e.agent_id == agent_id]
        if agent_type:
            filtered_events = [e for e in filtered_events if e.agent_type == agent_type]
        if log_level:
            filtered_events = [e for e in filtered_events if e.log_level == log_level]
        if compliance_level:
            filtered_events = [e for e in filtered_events if e.compliance_level == compliance_level]
        
        return filtered_events
    
    async def generate_compliance_report(self, 
                                       start_time: Optional[datetime] = None,
                                       end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a comprehensive compliance report."""
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        
        events = self.get_events(start_time=start_time, end_time=end_time)
        
        # Compliance statistics
        compliance_counts = {}
        for level in ComplianceLevel:
            compliance_counts[level.value] = len([e for e in events if e.compliance_level == level])
        
        # Agent performance
        agent_stats = {}
        for event in events:
            if event.agent_id not in agent_stats:
                agent_stats[event.agent_id] = {
                    'total_events': 0,
                    'errors': 0,
                    'avg_execution_time': 0.0,
                    'compliance_violations': 0
                }
            
            stats = agent_stats[event.agent_id]
            stats['total_events'] += 1
            if event.error_details:
                stats['errors'] += 1
            if event.compliance_level in [ComplianceLevel.VIOLATION, ComplianceLevel.CRITICAL_VIOLATION]:
                stats['compliance_violations'] += 1
            stats['avg_execution_time'] = (stats['avg_execution_time'] + event.execution_time) / 2
        
        return {
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_events': len(events),
            'compliance_summary': compliance_counts,
            'agent_performance': agent_stats,
            'top_errors': self._get_top_errors(events),
            'performance_metrics': self._calculate_performance_metrics(events)
        }
    
    def _get_top_errors(self, events: List[AuditEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most common errors."""
        error_counts = {}
        for event in events:
            if event.error_details:
                error_counts[event.error_details] = error_counts.get(event.error_details, 0) + 1
        
        return [
            {'error': error, 'count': count}
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    def _calculate_performance_metrics(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Calculate system performance metrics."""
        if not events:
            return {}
        
        execution_times = [e.execution_time for e in events if e.execution_time > 0]
        
        return {
            'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
            'max_execution_time': max(execution_times) if execution_times else 0,
            'min_execution_time': min(execution_times) if execution_times else 0,
            'total_processing_time': sum(execution_times),
            'success_rate': len([e for e in events if not e.error_details]) / len(events) * 100
        }
    
    async def cleanup_old_logs(self):
        """Clean up old log files based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()
        
        for event_file in self.log_dir.glob("*.jsonl"):
            if event_file.stat().st_mtime < cutoff_date.timestamp():
                event_file.unlink()


# Global audit logger instance
audit_logger = AuditLogger()
