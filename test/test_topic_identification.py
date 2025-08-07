"""
Test script for Topic Identification Agent with the query "where can you park at the newHQ"
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add the agentic_system directory to the path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent  # Go up one level from test/ to code/
agentic_system_dir = parent_dir / "agentic_system"
sys.path.insert(0, str(parent_dir))

from agentic_system.agents.topic_identification_agent import TopicIdentificationAgent
from agentic_system.core.base import Task
from agentic_system.enums import Priority


async def test_topic_identification():
    """Test the Topic Identification Agent with a parking query."""
    
    print("🧪 Testing Topic Identification Agent")
    print("=" * 50)
    
    # Create the agent
    config = {
        'confidence_threshold': 0.6,
        'max_topics': 5
    }
    
    agent = TopicIdentificationAgent(
        agent_id="test_tia_001",
        config=config
    )
    
    # Test query about parking at newHQ
    test_query = "where can you park at the newHQ"
    
    print(f"📝 Test Query: '{test_query}'")
    print("-" * 50)
    
    # Create a task with the query
    task = Task(
        name="query_test",
        description="Test query",
        priority=Priority.MEDIUM,
        input_data={'query': test_query}
    )
    
    try:
        # Initialize the agent (this will try Azure OpenAI, but will gracefully fallback)
        print("🔄 Initializing Topic Identification Agent...")
        initialized = await agent.initialize()
        
        if initialized:
            print("✅ Agent initialized successfully with Azure OpenAI")
        else:
            print("⚠️  Agent initialization failed, will use fallback analysis")
        
        # Process the query
        print(f"🔄 Processing query: '{test_query}'")
        response = await agent.process(task)
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 ANALYSIS RESULTS")
        print("=" * 50)
        
        if response.success:
            data = response.data
            identification_result = data.get('identification_result', {})
            
            print(f"🎯 Topics Identified: {identification_result.get('topics', [])}")
            print(f"🔍 Intent: {identification_result.get('intent', 'unknown')}")
            print(f"⚡ Complexity: {identification_result.get('complexity', 'unknown')}")
            print(f"🛠️  Capabilities Needed: {identification_result.get('capabilities_needed', [])}")
            print(f"📊 Confidence Score: {identification_result.get('confidence', 0.0):.2%}")
            print(f"🗂️  Database Recommendation: {identification_result.get('database_recommendation', 'unknown')}")
            print(f"🔄 Routing Strategy: {identification_result.get('routing_strategy', 'unknown')}")
            
            # Metadata details
            metadata = identification_result.get('metadata', {})
            print(f"\n📋 Metadata:")
            print(f"   Domain: {metadata.get('domain', 'unknown')}")
            print(f"   Urgency: {metadata.get('urgency', 'unknown')}")
            print(f"   Estimated Processing Time: {metadata.get('estimated_processing_time', 'unknown')}")
            print(f"   Fallback Used: {metadata.get('fallback_used', False)}")
            
            # Topic scores
            topic_scores = metadata.get('topic_scores', {})
            if topic_scores:
                print(f"\n🎯 Topic Scores:")
                print(f"   Confluence Relevance: {topic_scores.get('confluence_relevance', 0.0):.2f}")
                print(f"   NewHQ Relevance: {topic_scores.get('newhq_relevance', 0.0):.2f}")
                print(f"   Web Search Relevance: {topic_scores.get('web_search_relevance', 0.0):.2f}")
            
            # Routing recommendation
            routing_rec = data.get('routing_recommendation', {})
            if routing_rec:
                print(f"\n🔄 Routing Recommendation:")
                print(f"   Primary Agent: {routing_rec.get('primary_agent', 'unknown')}")
                print(f"   Priority Level: {routing_rec.get('priority_level', 'unknown')}")
                print(f"   Validation Required: {routing_rec.get('validation_required', False)}")
                print(f"   Human Review Needed: {routing_rec.get('human_review_needed', False)}")
            
            # Final response
            final_response = data.get('final_response', '')
            if final_response:
                print(f"\n💬 Final Response:")
                print(f"   {final_response}")
            
            print(f"\n⏱️  Execution Time: {response.execution_time:.3f} seconds")
            print(f"📄 Sources: {response.sources}")
            
        else:
            print(f"❌ Processing failed: {response.error_message}")
        
        # Test the direct database recommendation method
        print("\n" + "=" * 50)
        print("🎯 DIRECT DATABASE RECOMMENDATION TEST")
        print("=" * 50)
        
        db_recommendation = await agent.get_database_recommendation(test_query)
        print(f"📍 Database: {db_recommendation.get('database', 'unknown')}")
        print(f"📊 Confidence: {db_recommendation.get('confidence', 0.0):.2%}")
        print(f"🏷️  Topics: {db_recommendation.get('topics', [])}")
        print(f"🏢 Domain: {db_recommendation.get('domain', 'unknown')}")
        print(f"💭 Reasoning: {db_recommendation.get('reasoning', 'No reasoning provided')}")
        
        if 'topic_scores' in db_recommendation:
            scores = db_recommendation['topic_scores']
            print(f"\n🎯 Topic Scores:")
            for score_type, value in scores.items():
                print(f"   {score_type}: {value:.2f}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Shutdown the agent
        try:
            await agent.shutdown()
            print("\n🔄 Agent shutdown completed")
        except Exception as e:
            print(f"⚠️  Agent shutdown error: {str(e)}")


def test_fallback_analysis():
    """Test the fallback analysis method directly (no Azure required)."""
    print("\n" + "=" * 50)
    print("🔧 FALLBACK ANALYSIS TEST (No Azure Required)")
    print("=" * 50)
    
    # Create agent without initialization
    agent = TopicIdentificationAgent(agent_id="test_fallback", config={'max_topics': 5})
    
    test_query = "where can you park at the newHQ"
    print(f"📝 Test Query: '{test_query}'")
    
    # Test fallback analysis directly
    fallback_result = agent._fallback_analysis(test_query)
    
    print(f"\n🎯 Fallback Analysis Results:")
    print(f"   Topics: {fallback_result.get('topics', [])}")
    print(f"   Intent: {fallback_result.get('intent', 'unknown')}")
    print(f"   Complexity: {fallback_result.get('complexity', 'unknown')}")
    print(f"   Capabilities: {fallback_result.get('capabilities_needed', [])}")
    print(f"   Confidence: {fallback_result.get('confidence', 0.0):.2%}")
    print(f"   Database Rec: {fallback_result.get('database_recommendation', 'unknown')}")
    print(f"   Routing Strategy: {fallback_result.get('routing_strategy', 'unknown')}")
    
    metadata = fallback_result.get('metadata', {})
    print(f"\n📋 Metadata:")
    print(f"   Domain: {metadata.get('domain', 'unknown')}")
    print(f"   Fallback Used: {metadata.get('fallback_used', False)}")
    
    topic_scores = metadata.get('topic_scores', {})
    if topic_scores:
        print(f"\n🎯 Topic Scores:")
        for score_type, value in topic_scores.items():
            print(f"   {score_type}: {value:.2f}")


async def main():
    """Main test function."""
    print("🚀 Starting Topic Identification Agent Tests")
    print("=" * 60)
    
    # Test 1: Fallback analysis (always works)
    test_fallback_analysis()
    
    # Test 2: Full agent processing (requires Azure setup)
    await test_topic_identification()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
