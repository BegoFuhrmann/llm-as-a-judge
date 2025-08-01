"""
Task Orchestration Agent (TOA) - Routes tasks to appropriate agents based on analysis.
"""
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.base import BaseAgent, Task, AgentResponse, agent_registry
from ..enums import AgentType, TaskStatus, Priority, LogLevel
from ..audit.audit_log import audit_logger


class TaskOrchestrationAgent(BaseAgent):
    """
    Central orchestrator that routes tasks to appropriate agents based on topic analysis
    and system requirements.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, AgentType.TASK_ORCHESTRATION, config)
        self.routing_rules = config.get('routing_rules', {}) if config else {}
        self.load_balancing_enabled = config.get('load_balancing', True) if config else True
        self.max_concurrent_tasks = config.get('max_concurrent_tasks', 10) if config else 10
        self.current_tasks: Dict[str, Task] = {}
        
        # Default routing rules
        self._setup_default_routing_rules()
    
    def _setup_default_routing_rules(self):
        """Setup default routing rules for different query types."""
        default_rules = {
            'compliance_keywords': [
                'regulation', 'compliance', 'policy', 'legal', 'audit',
                'gdpr', 'hipaa', 'sox', 'standard', 'requirement'
            ],
            'autonomous_keywords': [
                'research', 'explore', 'find', 'search', 'discover',
                'latest', 'current', 'news', 'trend', 'market'
            ],
            'rag_preferred_domains': [
                'documentation', 'knowledge_base', 'internal',
                'structured', 'database', 'repository'
            ],
            'complexity_thresholds': {
                'simple': 0.3,
                'moderate': 0.6,
                'complex': 0.8
            }
        }
        
        self.routing_rules.update(default_rules)
    
    async def initialize(self) -> bool:
        """Initialize the task orchestrator."""
        try:
            self.is_active = True
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "initialize",
                log_level=LogLevel.INFO
            )
            return True
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "initialize_failed",
                log_level=LogLevel.ERROR,
                error=str(e)
            )
            return False
    
    async def process(self, task: Task) -> AgentResponse:
        """
        Route the task to the appropriate agent based on topic analysis.
        
        Args:
            task: Task containing topic identification results
            
        Returns:
            AgentResponse with routing decision and execution results
        """
        start_time = time.time()
        
        try:
            # Extract topic identification results
            topic_data = task.input_data.get('identification_result', {})
            if not topic_data:
                return AgentResponse(
                    success=False,
                    error_message="No topic identification data provided",
                    execution_time=time.time() - start_time
                )
            
            # Determine routing strategy
            routing_decision = await self._make_routing_decision(topic_data, task)
            
            # Route to appropriate agent
            execution_result = await self._route_to_agent(routing_decision, task)
            
            execution_time = time.time() - start_time
            
            response = AgentResponse(
                success=True,
                data={
                    'routing_decision': routing_decision,
                    'execution_result': execution_result,
                    'orchestration_metadata': {
                        'total_execution_time': execution_time,
                        'routing_confidence': routing_decision.get('confidence', 0.0)
                    }
                },
                execution_time=execution_time,
                confidence_score=routing_decision.get('confidence', 0.0)
            )
            
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "orchestration_completed",
                task=task, response=response, log_level=LogLevel.INFO
            )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_response = AgentResponse(
                success=False,
                error_message=f"Task orchestration failed: {str(e)}",
                execution_time=execution_time
            )
            
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "orchestration_failed",
                task=task, response=error_response, log_level=LogLevel.ERROR
            )
            
            return error_response
    
    async def _make_routing_decision(self, topic_data: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """Make routing decision based on topic analysis."""
        query = task.input_data.get('query', '').lower()
        topics = topic_data.get('topics', [])
        intent = topic_data.get('intent', '')
        complexity = topic_data.get('complexity', 'moderate')
        capabilities_needed = topic_data.get('capabilities_needed', [])
        
        routing_score = {
            'rag_agent': 0.0,
            'autonomous_agent': 0.0
        }
        
        # Check for compliance-related content
        compliance_score = sum(1 for keyword in self.routing_rules['compliance_keywords'] 
                             if keyword in query or any(keyword in topic.lower() for topic in topics))
        
        # Check for autonomous/exploratory content
        autonomous_score = sum(1 for keyword in self.routing_rules['autonomous_keywords']
                             if keyword in query or any(keyword in topic.lower() for topic in topics))
        
        # Capability-based scoring
        if 'rag_search' in capabilities_needed or 'compliance_verification' in capabilities_needed:
            routing_score['rag_agent'] += 0.4
        
        if 'web_search' in capabilities_needed or 'real_time_data' in capabilities_needed:
            routing_score['autonomous_agent'] += 0.4
        
        # Compliance bias
        if compliance_score > 0:
            routing_score['rag_agent'] += 0.3 + (compliance_score * 0.1)
        
        # Exploratory bias
        if autonomous_score > 0:
            routing_score['autonomous_agent'] += 0.3 + (autonomous_score * 0.1)
        
        # Complexity consideration
        complexity_factor = self.routing_rules['complexity_thresholds'].get(complexity, 0.5)
        if complexity_factor > 0.7:  # Complex queries prefer RAG for accuracy
            routing_score['rag_agent'] += 0.2
        elif complexity_factor < 0.4:  # Simple queries can go autonomous
            routing_score['autonomous_agent'] += 0.2
        
        # Determine winner
        selected_agent = max(routing_score, key=routing_score.get)
        confidence = max(routing_score.values())
        
        # Fallback to RAG if confidence is too low
        if confidence < 0.3:
            selected_agent = 'rag_agent'
            confidence = 0.5
        
        return {
            'selected_agent': selected_agent,
            'confidence': min(confidence, 1.0),
            'routing_scores': routing_score,
            'decision_factors': {
                'compliance_score': compliance_score,
                'autonomous_score': autonomous_score,
                'complexity_factor': complexity_factor,
                'capabilities_needed': capabilities_needed
            },
            'reasoning': self._generate_routing_reasoning(selected_agent, routing_score, topic_data)
        }
    
    def _generate_routing_reasoning(self, selected_agent: str, scores: Dict[str, float], 
                                  topic_data: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for routing decision."""
        reasoning_parts = []
        
        if selected_agent == 'rag_agent':
            reasoning_parts.append("Routed to RAG Agent because:")
            if scores['rag_agent'] > scores['autonomous_agent']:
                reasoning_parts.append(f"- RAG score ({scores['rag_agent']:.2f}) > Autonomous score ({scores['autonomous_agent']:.2f})")
            
            capabilities = topic_data.get('capabilities_needed', [])
            if 'compliance_verification' in capabilities:
                reasoning_parts.append("- Compliance verification required")
            if 'rag_search' in capabilities:
                reasoning_parts.append("- Structured knowledge retrieval needed")
        else:
            reasoning_parts.append("Routed to Autonomous Agent because:")
            if scores['autonomous_agent'] > scores['rag_agent']:
                reasoning_parts.append(f"- Autonomous score ({scores['autonomous_agent']:.2f}) > RAG score ({scores['rag_agent']:.2f})")
            
            capabilities = topic_data.get('capabilities_needed', [])
            if 'web_search' in capabilities:
                reasoning_parts.append("- Web search capabilities required")
            if 'real_time_data' in capabilities:
                reasoning_parts.append("- Real-time data access needed")
        
        return "\n".join(reasoning_parts)
    
    async def _route_to_agent(self, routing_decision: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """Route task to the selected agent and get results."""
        selected_agent_type = routing_decision['selected_agent']
        
        # Map agent names to types
        agent_type_mapping = {
            'rag_agent': AgentType.RAG_BASED,
            'autonomous_agent': AgentType.AUTONOMOUS
        }
        
        target_agent_type = agent_type_mapping.get(selected_agent_type)
        if not target_agent_type:
            raise ValueError(f"Unknown agent type: {selected_agent_type}")
        
        # Get available agents of the target type
        available_agents = agent_registry.get_agents_by_type(target_agent_type)
        
        if not available_agents:
            raise ValueError(f"No available agents of type: {target_agent_type.value}")
        
        # Select agent (simple round-robin for now)
        selected_agent = available_agents[0]  # TODO: Implement proper load balancing
        
        # Update task assignment
        task.assigned_agent = selected_agent.agent_id
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()
        
        # Store task for tracking
        self.current_tasks[task.id] = task
        
        try:
            # Execute task on selected agent
            result = await selected_agent.process(task)
            
            # Update task status
            if result.success:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.output_data = result.data
            else:
                task.status = TaskStatus.FAILED
                task.error_info = result.error_message
            
            task.updated_at = datetime.now()
            
            return {
                'agent_id': selected_agent.agent_id,
                'agent_type': target_agent_type.value,
                'execution_result': result.data if result.success else None,
                'execution_success': result.success,
                'execution_time': result.execution_time,
                'error_message': result.error_message,
                'confidence_score': result.confidence_score
            }
            
        finally:
            # Clean up completed task
            if task.id in self.current_tasks:
                del self.current_tasks[task.id]
    
    async def shutdown(self) -> bool:
        """Shutdown the orchestrator gracefully."""
        try:
            # Wait for current tasks to complete
            if self.current_tasks:
                await audit_logger.log_agent_action(
                    self.agent_id, self.agent_type, "waiting_for_tasks",
                    log_level=LogLevel.INFO,
                    active_tasks=len(self.current_tasks)
                )
                
                # Wait up to 30 seconds for tasks to complete
                max_wait = 30
                waited = 0
                while self.current_tasks and waited < max_wait:
                    await asyncio.sleep(1)
                    waited += 1
            
            self.is_active = False
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "shutdown",
                log_level=LogLevel.INFO
            )
            return True
            
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "shutdown_failed",
                log_level=LogLevel.ERROR,
                error=str(e)
            )
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status including current tasks."""
        base_status = super().get_status()
        base_status.update({
            'active_tasks': len(self.current_tasks),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'load_balancing_enabled': self.load_balancing_enabled,
            'current_task_ids': list(self.current_tasks.keys())
        })
        return base_status