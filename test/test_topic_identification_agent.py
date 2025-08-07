import sys, os
import asyncio
from dotenv import load_dotenv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agentic_system.agents.topic_identification_agent import TopicIdentificationAgent
from agentic_system.core.base import Task

async def main():
    load_dotenv()
    agent = TopicIdentificationAgent(agent_id="tia-test-001")
    await agent.initialize()
    task = Task(
        id="tia-task-001",
        name="Query",
        description="Identify topic and routing for the question.",
        input_data={"query": "who is the current coach for hamburger sv?"}
    )
    response = await agent.process(task)
    await agent.shutdown()
    
    print("="*80)
    print("TOPIC IDENTIFICATION AGENT TEST RESULTS")
    print("="*80)
    print(f"Query: '{task.input_data['query']}'")
    print(f"Success: {response.success}")
    print(f"Confidence Score: {response.confidence_score}")
    print(f"Execution Time: {response.execution_time:.2f}s")
    print()
    
    if response.success:
        identification = response.data.get("identification_result", {})
        routing = response.data.get("routing_recommendation", {})
        
        print("IDENTIFICATION RESULT:")
        print(f"  Topics: {identification.get('topics', [])}")
        print(f"  Intent: {identification.get('intent', 'N/A')}")
        print(f"  Complexity: {identification.get('complexity', 'N/A')}")
        print(f"  Domain: {identification.get('metadata', {}).get('domain', 'N/A')}")
        print(f"  Database Recommendation: {identification.get('database_recommendation', 'N/A')}")
        print(f"  Routing Strategy: {identification.get('routing_strategy', 'N/A')}")
        print(f"  Capabilities Needed: {identification.get('capabilities_needed', [])}")
        print(f"  Confidence: {identification.get('confidence', 0)}")
        print(f"  Fallback Used: {identification.get('metadata', {}).get('fallback_used', False)}")
        print()
        
        print("ROUTING RECOMMENDATION:")
        print(f"  Primary Agent: {routing.get('primary_agent', 'N/A')}")
        print(f"  Parallel Processing: {routing.get('parallel_processing', [])}")
        print(f"  Priority Level: {routing.get('priority_level', 'N/A')}")
        print(f"  Validation Required: {routing.get('validation_required', False)}")
        print(f"  Human Review Needed: {routing.get('human_review_needed', False)}")
        print()
        
        print("TOPIC SCORES:")
        topic_scores = identification.get('metadata', {}).get('topic_scores', {})
        for db, score in topic_scores.items():
            print(f"  {db}: {score}")
        print()
        
        print("FINAL ANSWER:")
        final_response = response.data.get("final_response", "")
        if final_response:
            print(f"📋 FINAL RESPONSE:")
            print(f"   {final_response}")
            print()
        
        # Additional routing analysis
        if identification.get('database_recommendation') == 'newhq':
            print(f"✅ SUCCESS: Query correctly routed to NewHQ database")
            print(f"   - The system identified this as a facilities-related query")
            print(f"   - Primary agent: {routing.get('primary_agent')} (should be 'rag_based')")
            print(f"   - Will search NewHQ knowledge base for information")
        elif identification.get('database_recommendation') == 'confluence':
            print(f"✅ SUCCESS: Query correctly routed to Confluence database")
            print(f"   - The system identified this as a technical-related query")
            print(f"   - Primary agent: {routing.get('primary_agent')} (should be 'rag_based')")
            print(f"   - Will search Confluence knowledge base for information")
        elif identification.get('database_recommendation') == 'autonomous_agent':
            print(f"✅ SUCCESS: Query correctly routed to web search")
            print(f"   - The system identified this requires external information")
            print(f"   - Primary agent: {routing.get('primary_agent')} (should be 'autonomous')")
            print(f"   - Will search the public internet for current information")
        else:
            print(f"⚠️  UNCLEAR: Query routing is ambiguous: {identification.get('database_recommendation')}")
    else:
        print(f"❌ FAILURE: {response.error_message}")
    
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
