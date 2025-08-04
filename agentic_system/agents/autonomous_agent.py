"""
Autonomous Agent - Orchestrates complex tasks using reasoning, planning, and tool execution.
"""
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Handle imports for both module and direct execution
try:
    # Try relative imports first (when used as a module)
    from ..core.base import BaseAgent, Task, AgentResponse
    from ..enums import AgentType, LogLevel, Priority, TaskStatus
    from ..audit.audit_log import audit_logger
    from ..tools.web_search import WebSearchTool, SearchQuery
    from ..tools.content_synthesizer import ContentSynthesizer
    from ..tools.source_validator import SourceValidator
except ImportError:
    # Fallback to absolute imports (when run directly)
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir.parent))
    
    from agentic_system.core.base import BaseAgent, Task, AgentResponse
    from agentic_system.enums import AgentType, LogLevel, Priority, TaskStatus
    from agentic_system.audit.audit_log import audit_logger
    from agentic_system.tools.web_search import WebSearchTool, SearchQuery
    from agentic_system.tools.content_synthesizer import ContentSynthesizer
    from agentic_system.tools.source_validator import SourceValidator


class AutonomousAgent(BaseAgent):
    """
    An autonomous agent capable of reasoning, planning, and executing complex tasks.
    
    This agent uses a set of tools to perform actions like web searches,
    content synthesis, and source validation to fulfill user requests.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, AgentType.AUTONOMOUS, config)
        self.config = config or {}
        
        # Agent state and planning
        self.plan: List[Dict[str, Any]] = []
        self.reasoning_history: List[str] = []
        self.current_subtask_id: Optional[str] = None
        
        # Tool integration
        self.web_search_tool = WebSearchTool()
        self.content_synthesizer = ContentSynthesizer()
        self.source_validator = SourceValidator()
        
        # Agent parameters
        self.max_iterations = self.config.get('max_iterations', 10)
        self.max_reasoning_steps = self.config.get('max_reasoning_steps', 5)
        
    async def initialize(self) -> bool:
        """Initialize the autonomous agent and its tools."""
        await audit_logger.log_system_event(
            event_type="agent_initialization",
            message=f"AutonomousAgent {self.agent_id} initializing.",
            log_level=LogLevel.INFO,
            agent_id=self.agent_id
        )
        self.is_active = True
        return True

    async def shutdown(self) -> bool:
        """Shutdown the autonomous agent."""
        await audit_logger.log_system_event(
            event_type="agent_shutdown",
            message=f"AutonomousAgent {self.agent_id} shutting down.",
            log_level=LogLevel.INFO,
            agent_id=self.agent_id
        )
        self.is_active = False
        return True

    async def process(self, task: Task) -> AgentResponse:
        """
        Process a complex task by breaking it down, planning, and executing steps.
        """
        start_time = time.time()
        
        await audit_logger.log_agent_action(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            action="task_received",
            task=task,
            log_level=LogLevel.INFO
        )
        
        try:
            # 1. Deconstruct and understand the task
            understanding = await self._understand_task(task)
            self.reasoning_history.append(f"Task understanding: {understanding}")
            
            # 2. Create a plan
            self.plan = await self._create_plan(task, understanding)
            self.reasoning_history.append(f"Initial plan: {self.plan}")
            
            # 3. Execute the plan
            final_result = await self._execute_plan(task)
            
            # 4. Final response synthesis
            response_data = {
                "result": final_result,
                "reasoning": self.reasoning_history,
                "plan": self.plan
            }
            
            execution_time = time.time() - start_time
            response = AgentResponse(
                success=True,
                data=response_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            response = AgentResponse(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
            await audit_logger.log_agent_action(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                action="task_failed",
                task=task,
                response=response,
                log_level=LogLevel.ERROR
            )
            
        await audit_logger.log_agent_action(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            action="task_completed",
            task=task,
            response=response,
            log_level=LogLevel.INFO
        )
        
        return response

    async def _understand_task(self, task: Task) -> str:
        """Deconstruct and analyze the task requirements."""
        # Placeholder: In a real scenario, this would involve LLM-based analysis
        understanding = f"The user wants to accomplish: '{task.description}'. This requires information gathering and synthesis."
        return understanding

    async def _create_plan(self, task: Task, understanding: str) -> List[Dict[str, Any]]:
        """Create a step-by-step plan to address the task."""
        # Placeholder: Simple plan based on keywords
        plan = []
        if "search" in task.description.lower() or "find" in task.description.lower():
            plan.append({
                "step": 1,
                "action": "web_search",
                "query": task.input_data.get("query", task.description),
                "status": "pending"
            })
        
        # Add placeholder steps for future tools
        plan.append({
            "step": len(plan) + 1,
            "action": "synthesize_content",
            "status": "pending"
        })
        plan.append({
            "step": len(plan) + 1,
            "action": "validate_sources",
            "status": "pending"
        })
        
        return plan

    async def _execute_plan(self, task: Task) -> Any:
        """Execute the plan step-by-step, adapting as needed."""
        results_context = {}
        
        for i, step in enumerate(self.plan):
            self.current_subtask_id = f"{task.id}_step_{step['step']}"
            self.reasoning_history.append(f"Executing step {step['step']}: {step['action']}")
            
            try:
                if step["action"] == "web_search":
                    query = step["query"]
                    search_results = await self.web_search_tool.search(query)
                    results_context["web_search"] = search_results
                    self.reasoning_history.append(f"Web search found {len(search_results)} results.")
                
                elif step["action"] == "validate_sources":
                    if "web_search" in results_context:
                        validation_response = await self.source_validator.validate(results_context["web_search"])
                        if validation_response.success:
                            validated_sources = validation_response.data["validated_sources"]
                            results_context["validated_sources"] = validated_sources
                            self.reasoning_history.append(f"Source validation completed. {len(validated_sources)} trusted sources found.")
                        else:
                            self.reasoning_history.append("Source validation failed.")
                    else:
                        self.reasoning_history.append("No sources to validate.")

                elif step["action"] == "synthesize_content":
                    sources_to_synthesize = results_context.get("validated_sources", results_context.get("web_search"))
                    if sources_to_synthesize:
                        synthesis_response = await self.content_synthesizer.synthesize(task.description, sources_to_synthesize)
                        if synthesis_response.success:
                            results_context["synthesis"] = synthesis_response.data["synthesized_text"]
                            self.reasoning_history.append("Content synthesis successful.")
                        else:
                            self.reasoning_history.append("Content synthesis failed.")
                    else:
                        results_context["synthesis"] = "No content to synthesize."
                        self.reasoning_history.append("Content synthesis skipped: no sources available.")
                
                self.plan[i]["status"] = "completed"
                
            except Exception as e:
                self.plan[i]["status"] = "failed"
                self.reasoning_history.append(f"Step {step['step']} failed: {e}")
                raise
        
        return results_context

# Example of how to create and use the agent
async def main():
    agent = AutonomousAgent(agent_id="agent-007")
    await agent.initialize()
    
    task = Task(
        id="task-123",
        name="Research Python",
        description="Search for information about the Python programming language.",
        input_data={"query": "What is Python programming language?"}
    )
    
    response = await agent.process(task)
    
    if response.success:
        print("Task completed successfully!")
        print("Final Result:", response.data.get("result"))
        print("\nReasoning History:")
        for entry in response.data.get("reasoning", []):
            print(f"- {entry}")
    else:
        print(f"Task failed: {response.error_message}")
        
    await agent.shutdown()

if __name__ == "__main__":
    # This allows running the agent file directly for testing
    # Note: Ensure the parent directory is in PYTHONPATH
    # `export PYTHONPATH=$PYTHONPATH:./`
    asyncio.run(main())
