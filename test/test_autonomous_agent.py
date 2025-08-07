import sys, os
import asyncio
import pytest
from dotenv import load_dotenv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agentic_system.agents.autonomous_agent import AutonomousAgent
from agentic_system.core.base import Task

@pytest.mark.asyncio
async def test_autonomous_agent_full_pipeline():
    agent = AutonomousAgent(agent_id="test-agent-002")
    await agent.initialize()
    task = Task(
        id="test-task-002",
        name="Research Python",
        description="Search for information about Python programming.",
        input_data={"query": "What is Python programming?"}
    )
    response = await agent.process(task)
    await agent.shutdown()
    assert response.success, f"Agent failed: {response.error_message}"
    data = response.data
    # Check that web search results exist
    assert "result" in data, "Response should contain 'result'"
    result = data["result"]
    assert "web_search" in result, "Web search results missing"
    # Check that source validation ran
    assert "validated_sources" in result, "Validated sources missing"
    # Check that synthesis ran
    assert "synthesis" in result, "Synthesis missing"
    # Check reasoning history for tool usage
    reasoning = data.get("reasoning", [])
    assert any("Web search found" in r for r in reasoning), "Web search step not recorded in reasoning"
    assert any("Source validation completed" in r for r in reasoning), "Source validation step not recorded in reasoning"
    assert any("Content synthesis" in r for r in reasoning), "Content synthesis step not recorded in reasoning"

@pytest.mark.asyncio
async def test_autonomous_agent_full_pipeline_env():
    # Load environment variables
    load_dotenv()
    # Check that required env vars are set
    required_vars = [
        "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_CLIENT_SECRET"
    ]
    for var in required_vars:
        assert os.getenv(var), f"Environment variable {var} must be set for synthesis"
    agent = AutonomousAgent(agent_id="test-agent-003")
    await agent.initialize()
    task = Task(
        id="test-task-003",
        name="Research Python",
        description="Search for information about Python programming.",
        input_data={"query": "What is Python programming?"}
    )
    response = await agent.process(task)
    await agent.shutdown()
    assert response.success, f"Agent failed: {response.error_message}"
    data = response.data
    # Check that web search results exist
    assert "result" in data, "Response should contain 'result'"
    result = data["result"]
    assert "web_search" in result, "Web search results missing"
    # Check that source validation ran
    assert "validated_sources" in result, "Validated sources missing"
    # Check that synthesis ran and is non-empty
    assert "synthesis" in result, "Synthesis missing"
    assert isinstance(result["synthesis"], str), "Synthesis should be a string"
    assert len(result["synthesis"]) > 10, "Synthesized content should not be empty"
    # Check reasoning history for tool usage
    reasoning = data.get("reasoning", [])
    assert any("Web search found" in r for r in reasoning), "Web search step not recorded in reasoning"
    assert any("Source validation completed" in r for r in reasoning), "Source validation step not recorded in reasoning"
    assert any("Content synthesis" in r for r in reasoning), "Content synthesis step not recorded in reasoning"
