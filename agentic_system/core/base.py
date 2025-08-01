"""
Core base classes and interfaces for the agentic system.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import uuid

from ..enums import AgentType, TaskStatus, Priority, LogLevel


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    agent_type: AgentType = AgentType.TASK_ORCHESTRATION
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: Priority = Priority.MEDIUM


@dataclass
class Task:
    """Task representation for the agentic system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    retry_count: int = 0


@dataclass
class AgentResponse:
    """Standard response format from agents."""
    success: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    confidence_score: float = 0.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.is_active = False
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
    @abstractmethod
    async def process(self, task: Task) -> AgentResponse:
        """Process a task and return a response."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the agent gracefully."""
        pass
    
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "config": self.config
        }


class AgentRegistry:
    """Registry for managing agents in the system."""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_types: Dict[AgentType, List[str]] = {}
    
    def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent in the system."""
        try:
            self._agents[agent.agent_id] = agent
            
            if agent.agent_type not in self._agent_types:
                self._agent_types[agent.agent_type] = []
            self._agent_types[agent.agent_type].append(agent.agent_id)
            
            return True
        except Exception:
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the system."""
        try:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                del self._agents[agent_id]
                
                if agent.agent_type in self._agent_types:
                    self._agent_types[agent.agent_type].remove(agent_id)
                    if not self._agent_types[agent.agent_type]:
                        del self._agent_types[agent.agent_type]
                
                return True
            return False
        except Exception:
            return False
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type."""
        agent_ids = self._agent_types.get(agent_type, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def get_active_agents(self) -> List[BaseAgent]:
        """Get all active agents."""
        return [agent for agent in self._agents.values() if agent.is_active]


# Global agent registry instance
agent_registry = AgentRegistry()
