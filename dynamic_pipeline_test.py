"""
Fully Dynamic End-to-End Test: User Input -> Topic Identification -> Agent Routing -> Final Response
This script provides an interactive command-line interface to test the complete agentic pipeline.
"""
import asyncio
import sys
import traceback
from pathlib import Path

# Add the agentic_system directory to the path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent if "test" in str(current_dir) else current_dir
sys.path.insert(0, str(parent_dir))

# Import necessary components after setting up the path
try:
    from agentic_system.agents.topic_identification_agent import TopicIdentificationAgent
    from agentic_system.agents.rag_agent import RAGAgent
    from agentic_system.agents.autonomous_agent import AutonomousAgent
    from agentic_system.core.base import Task
    from agentic_system.enums import Priority
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the script is run from the correct directory and all dependencies are installed.")
    sys.exit(1)

async def dynamic_query_pipeline():
    """
    Execute the complete query pipeline from user input to final answer in a loop.
    """
    print("🚀 DYNAMIC QUERY PIPELINE - INTERACTIVE TEST 🚀")
    print("=" * 70)
    print("Enter your query to see the full agentic routing in action.")
    print("Type 'exit' or 'quit' to terminate the session.")
    print("=" * 70)

    while True:
        try:
            user_query = input("\nEnter your query: ")
            if user_query.lower() in ['exit', 'quit']:
                print("👋 Exiting dynamic pipeline test. Goodbye!")
                break

            print("-" * 70)
            print(f"📝 Original Query: '{user_query}'")
            print("-" * 70)

            # STEP 1: Topic Identification and Routing
            print("🔍 STEP 1: Topic Identification & Routing Analysis")
            tia = TopicIdentificationAgent(
                agent_id="dynamic_pipeline_tia",
                config={'confidence_threshold': 0.6, 'max_topics': 5}
            )
            routing_info = await tia.get_database_recommendation(user_query)
            
            print(f"   🎯 Identified Topics: {routing_info.get('topics', [])}")
            print(f"   🗂️  Recommended Database: {routing_info.get('database', 'unknown')}")
            print(f"   📊 Confidence: {routing_info.get('confidence', 0.0):.1%}")
            print(f"   💭 Reasoning: {routing_info.get('reasoning', 'No reasoning')}")
            
            await tia.shutdown()

            # STEP 2: Route to Appropriate Agent
            print("\n🔄 STEP 2: Routing to Appropriate Agent")
            recommended_db = routing_info.get('database', 'unknown')
            agent = None
            final_response = None

            if recommended_db in ['newhq', 'confluence']:
                print(f"✅ Routing to RAG Agent with '{recommended_db}' database...")
                agent = RAGAgent(
                    agent_id="dynamic_pipeline_rag",
                    config={
                        'confidence_threshold': 0.7,
                        'max_retrieval_results': 3,
                        'temperature': 0.1,
                    }
                )
                init_success = await agent.initialize()
                if not init_success:
                    print("❌ RAG Agent initialization failed.")
                    continue

                task = Task(
                    name="dynamic_user_query",
                    description=f"Query for {recommended_db}",
                    priority=Priority.HIGH,
                    input_data={
                        'query': user_query,
                        'database': recommended_db,
                        'topics': routing_info.get('topics', [])
                    }
                )
                rag_response = await agent.process(task)
                if rag_response.success:
                    final_response = rag_response.data.get('answer', 'No answer found.')
                else:
                    final_response = f"Error processing RAG task: {rag_response.error_message}"

            elif recommended_db == 'autonomous_agent':
                print("✅ Routing to Autonomous Agent for web search...")
                agent = AutonomousAgent(agent_id="dynamic_pipeline_autonomous")
                task = Task(
                    name="autonomous_web_search",
                    description="Perform a web search for the user query.",
                    priority=Priority.HIGH,
                    input_data={'query': user_query}
                )
                # Note: This assumes AutonomousAgent has a `process` method that returns a response.
                # You may need to adjust this based on the actual implementation.
                auto_response = await agent.process(task)
                if auto_response.success:
                    final_response = auto_response.data.get('result', 'No result from web search.')
                else:
                    final_response = f"Error processing autonomous task: {auto_response.error_message}"

            else:
                print(f"⚠️  Unclear routing for '{recommended_db}'. No agent assigned.")
                final_response = "Could not determine an appropriate agent for this query."

            # STEP 3: Display Final Results
            print("\n📋 STEP 3: Final Result")
            print("=" * 70)
            print(final_response)
            print("=" * 70)

            if agent:
                await agent.shutdown()
                print(f"🔄 {agent.agent_type.value} Agent shutdown completed.")

        except Exception as e:
            print(f"❌ An unexpected error occurred in the pipeline: {str(e)}")
            traceback.print_exc()

async def main():
    """Main execution function."""
    await dynamic_query_pipeline()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Shutting down.")
