"""
Base Agent Class for Enhanced Agentic System
==========================================

Abstract base class providing common functionality for all agents:
- Azure OpenAI integration with managed identity
- Comprehensive audit trail logging
- Structured output for academic evaluation
- Error handling and resilience patterns
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .agentic_config import AgenticConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Current state of an agent."""
    agent_id: str
    status: str  # "idle", "processing", "error", "completed"
    current_task: Optional[str]
    last_activity: datetime
    performance_metrics: Dict[str, Any]


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the enhanced agentic system.
    
    Provides common functionality including:
    - Configuration management
    - Audit trail integration
    - Performance monitoring
    - Error handling
    """
    
    def __init__(
        self,
        agent_id: str,
        role: str,
        config: AgenticConfig,
        audit_manager: Optional['AuditTrailManager'] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.config = config
        self.audit_manager = audit_manager
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{agent_id}")
        
        # Initialize agent state
        self.state = AgentState(
            agent_id=agent_id,
            status="idle",
            current_task=None,
            last_activity=datetime.now(),
            performance_metrics={}
        )
        
        self.logger.info(f"Agent {agent_id} initialized with role: {role}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the agent.
        
        Returns:
            Health status information
        """
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "status": self.state.status,
            "last_activity": self.state.last_activity.isoformat(),
            "is_healthy": self.state.status != "error"
        }
    
    def update_status(self, status: str, task: Optional[str] = None) -> None:
        """Update agent status and current task."""
        self.state.status = status
        self.state.current_task = task
        self.state.last_activity = datetime.now()
        
        self.logger.debug(f"Agent {self.agent_id} status updated to: {status}")
    
    async def log_performance_metric(self, metric_name: str, value: Any) -> None:
        """Log a performance metric for this agent."""
        self.state.performance_metrics[metric_name] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method that each agent must implement.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processing results
        """
        pass
