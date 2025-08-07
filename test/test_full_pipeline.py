import sys, os
import asyncio
import time
from datetime import datetime
from dotenv import load_dotenv

# Set up environment and proxy settings
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agentic_system.agents.topic_identification_agent import TopicIdentificationAgent
from agentic_system.agents.autonomous_agent import AutonomousAgent
from agentic_system.core.base import Task

async def test_with_timeout(coro, timeout_seconds, operation_name):
    """Helper function to run async operations with timeout and progress indication"""
    print(f"⏱️  Starting {operation_name} (timeout: {timeout_seconds}s)")
    start_time = time.time()
    
    try:
        # Run with timeout
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        elapsed = time.time() - start_time
        print(f"✅ {operation_name} completed in {elapsed:.2f}s")
        return result, True
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"❌ {operation_name} timed out after {elapsed:.2f}s")
        return None, False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ {operation_name} failed after {elapsed:.2f}s: {e}")
        return None, False

async def main():
    # Query to test
    query = "who is the current coach for hamburger sv?"
    
    print("="*80)
    print("FULL PIPELINE TEST: TOPIC IDENTIFICATION + AUTONOMOUS EXECUTION (WITH TIMEOUT)")
    print("="*80)
    print(f"Query: '{query}'")
    print(f"Test started at: {datetime.now()}")
    print()
    
    # Step 1: Topic Identification with timeout
    print("STEP 1: TOPIC IDENTIFICATION")
    print("-" * 40)
    
    # Create agent
    tia = TopicIdentificationAgent(agent_id="tia-test-001")
    
    # Initialize with timeout
    _, init_success = await test_with_timeout(
        tia.initialize(), 
        30, 
        "Agent Initialization"
    )
    
    if not init_success:
        print("❌ Agent initialization failed or timed out")
        print("💡 This is usually caused by:")
        print("   - Network connectivity issues")
        print("   - Azure authentication problems")
        print("   - Slow Azure token acquisition")
        print("   - Firewall/proxy blocking Azure endpoints")
        return
    
    # Create task
    tia_task = Task(
        id="tia-task-001",
        name="Query Analysis",
        description="Identify topic and routing for the question.",
        input_data={"query": query}
    )
    
    # Process with timeout
    tia_response, process_success = await test_with_timeout(
        tia.process(tia_task),
        20,
        "Topic Processing"
    )
    
    # Shutdown agent
    try:
        await asyncio.wait_for(tia.shutdown(), timeout=10)
        print("✅ Agent shutdown completed")
    except Exception as e:
        print(f"⚠️  Agent shutdown warning: {e}")
    
    if not process_success or not tia_response or not tia_response.success:
        error_msg = tia_response.error_message if tia_response else "No response received"
        print(f"❌ Topic identification failed: {error_msg}")
        return
    
    identification = tia_response.data.get("identification_result", {})
    routing = tia_response.data.get("routing_recommendation", {})
    
    print(f"✅ Topic Identification Complete:")
    print(f"   - Database Recommendation: {identification.get('database_recommendation')}")
    print(f"   - Primary Agent: {routing.get('primary_agent')}")
    print(f"   - Confidence: {identification.get('confidence', 0)}")
    print(f"   - Topics: {identification.get('topics', [])}")
    print()
    
    # Step 2: Execute based on routing recommendation with timeout
    print("STEP 2: AUTONOMOUS AGENT EXECUTION")
    print("-" * 40)
    
    if routing.get('primary_agent') == 'autonomous':
        # Initialize autonomous agent with timeout
        autonomous_agent = AutonomousAgent(agent_id="auto-test-001")
        
        _, auto_init_success = await test_with_timeout(
            autonomous_agent.initialize(),
            30,
            "Autonomous Agent Initialization"
        )
        
        if not auto_init_success:
            print("❌ Autonomous agent initialization failed or timed out")
            return
        
        # Create task for autonomous agent
        autonomous_task = Task(
            id="auto-task-001",
            name="Web Search Query",
            description="Find current information about Hamburger SV coach",
            input_data={
                "query": query,
                "search_type": "current_information",
                "topic_context": identification.get('topics', [])
            }
        )
        
        print(f"🔍 Executing web search via autonomous agent...")
        
        # Process with timeout (web search can take longer)
        autonomous_response, auto_process_success = await test_with_timeout(
            autonomous_agent.process(autonomous_task),
            60,  # 60 seconds for web search
            "Autonomous Agent Processing"
        )
        
        # Shutdown autonomous agent
        try:
            await asyncio.wait_for(autonomous_agent.shutdown(), timeout=10)
            print("✅ Autonomous agent shutdown completed")
        except Exception as e:
            print(f"⚠️  Autonomous agent shutdown warning: {e}")
        
        if auto_process_success and autonomous_response and autonomous_response.success:
            print(f"✅ Autonomous Agent Execution Complete:")
            print(f"   - Execution Time: {autonomous_response.execution_time:.2f}s")
            print(f"   - Confidence: {autonomous_response.confidence_score}")
            print()
            
            # Extract the actual answer
            answer = autonomous_response.data.get('final_answer', '')
            sources = autonomous_response.data.get('sources', [])
            
            print("🎯 FINAL ANSWER:")
            print("=" * 60)
            if answer:
                print(answer)
            else:
                print("No specific answer found in response data")
                print("Raw response data:", autonomous_response.data)
            print()
            
            if sources:
                print("📚 SOURCES:")
                for i, source in enumerate(sources[:3], 1):
                    if isinstance(source, dict):
                        print(f"   {i}. {source.get('title', 'No title')} - {source.get('url', 'No URL')}")
                    else:
                        print(f"   {i}. {source}")
            
        else:
            error_msg = autonomous_response.error_message if autonomous_response else "No response received"
            print(f"❌ Autonomous agent execution failed: {error_msg}")
            print("💡 This could be due to:")
            print("   - Web search API issues (Tavily)")
            print("   - Network connectivity problems")
            print("   - Agent processing timeout")
            print("   - Missing or invalid API keys")
    
    elif routing.get('primary_agent') == 'rag_based':
        print(f"📖 This query should be handled by RAG-based search in {identification.get('database_recommendation')} database")
        print("   (RAG execution not implemented in this test)")
    
    else:
        print(f"❓ Unknown routing recommendation: {routing.get('primary_agent')}")
    
    print("\n" + "="*80)
    print("PIPELINE TEST COMPLETE")
    print(f"Test completed at: {datetime.now()}")
    print("="*80)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Test crashed with error: {e}")
        import traceback
        traceback.print_exc()
