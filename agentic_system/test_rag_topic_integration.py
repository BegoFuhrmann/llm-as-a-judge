"""
Test script to demonstrate the integration between RAG Agent and Topic Identification Agent.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the agentic_system to Python path
sys.path.append(str(Path(__file__).parent))

from agents.rag_agent import RAGAgent
from agents.topic_identification_agent import TopicIdentificationAgent
from core.base import Task
from enums import AgentType


async def test_rag_topic_integration():
    """Test the integration between RAG Agent and Topic Identification Agent."""
    
    print("🔄 Initializing RAG Agent with Topic Identification Integration...")
    
    # Initialize RAG Agent (it will automatically initialize the Topic Identification Agent)
    rag_agent = RAGAgent(
        agent_id="test_rag_agent",
        config={
            'confidence_threshold': 0.7,
            'max_retrieval_results': 3,
            'temperature': 0
        }
    )
    
    # Initialize the agent
    init_success = await rag_agent.initialize()
    if not init_success:
        print("❌ Failed to initialize RAG Agent")
        return
    
    print("✅ RAG Agent initialized successfully")
    print(f"📊 Knowledge Base Stats: {rag_agent.get_knowledge_base_stats()}")
    
    # Test queries for different databases
    test_queries = [
        {
            'query': 'Where can I park at the new headquarters?',
            'expected_db': 'newhq',
            'description': 'Office facility query'
        },
        {
            'query': 'What is the BAIA project about?',
            'expected_db': 'confluence',
            'description': 'Project documentation query'
        },
        {
            'query': 'How do I access the building facilities?',
            'expected_db': 'newhq',
            'description': 'Building access query'
        },
        {
            'query': 'Tell me about RPA automation workflows',
            'expected_db': 'confluence',
            'description': 'Technical documentation query'
        },
        {
            'query': 'What time does the office close?',
            'expected_db': 'newhq',
            'description': 'Office hours query'
        }
    ]
    
    print("\n🧪 Testing Topic Identification and Database Routing...")
    print("=" * 80)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        expected_db = test_case['expected_db']
        description = test_case['description']
        
        print(f"\n📝 Test {i}: {description}")
        print(f"Query: '{query}'")
        print(f"Expected Database: {expected_db}")
        
        # Test direct topic identification first
        if rag_agent.topic_agent:
            print("\n🔍 Topic Identification Results:")
            try:
                topic_recommendation = await rag_agent.topic_agent.get_database_recommendation(query)
                print(f"  📍 Recommended Database: {topic_recommendation['database']}")
                print(f"  🎯 Confidence: {topic_recommendation['confidence']:.2f}")
                print(f"  🏷️  Topics: {topic_recommendation['topics']}")
                print(f"  🏢 Domain: {topic_recommendation['domain']}")
                print(f"  💭 Reasoning: {topic_recommendation['reasoning']}")
            except Exception as e:
                print(f"  ❌ Topic identification error: {e}")
        
        # Test full RAG processing
        print("\n🔄 Processing with RAG Agent:")
        task = Task(
            task_id=f"test_task_{i}",
            input_data={'query': query},
            agent_type=AgentType.RAG_BASED,
            priority='medium'
        )
        
        try:
            response = await rag_agent.process(task)
            
            if response.success:
                print(f"  ✅ Success: {response.success}")
                print(f"  🗃️  Database Used: {response.data.get('database_used', 'Unknown')}")
                print(f"  📊 Confidence: {response.confidence_score:.2f}")
                print(f"  🔧 Method: {response.data.get('search_method', 'Unknown')}")
                print(f"  📄 Answer Preview: {response.data.get('answer', 'No answer')[:150]}...")
                
                # Check if routing was correct
                actual_db = response.data.get('database_used', '').lower()
                if expected_db in actual_db:
                    print(f"  ✅ Routing Correct: Expected {expected_db}, got {actual_db}")
                else:
                    print(f"  ⚠️  Routing Check: Expected {expected_db}, got {actual_db}")
            else:
                print(f"  ❌ Failed: {response.error_message}")
                
        except Exception as e:
            print(f"  ❌ Processing error: {e}")
        
        print("-" * 80)
    
    # Test knowledge base statistics
    print("\n📈 Final Knowledge Base Statistics:")
    stats = rag_agent.get_knowledge_base_stats()
    for db_name, db_stats in stats.items():
        if isinstance(db_stats, dict) and 'status' in db_stats:
            print(f"  {db_name}: {db_stats}")
    
    # Shutdown
    print("\n🔄 Shutting down agents...")
    shutdown_success = await rag_agent.shutdown()
    print(f"✅ Shutdown {'successful' if shutdown_success else 'failed'}")


async def test_topic_agent_standalone():
    """Test the Topic Identification Agent standalone."""
    print("\n🔬 Testing Topic Identification Agent Standalone...")
    
    topic_agent = TopicIdentificationAgent(
        agent_id="test_topic_agent",
        config={'confidence_threshold': 0.7}
    )
    
    init_success = await topic_agent.initialize()
    if not init_success:
        print("❌ Failed to initialize Topic Identification Agent")
        return
    
    test_queries = [
        "Where is the parking garage in the new building?",
        "How does the BAIA chatbot work?",
        "What are the office hours for the headquarters?",
        "Show me the RPA development guidelines"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Analyzing: '{query}'")
        try:
            recommendation = await topic_agent.get_database_recommendation(query)
            print(f"  Database: {recommendation['database']}")
            print(f"  Confidence: {recommendation['confidence']:.2f}")
            print(f"  Topics: {recommendation['topics']}")
            print(f"  Reasoning: {recommendation['reasoning']}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    await topic_agent.shutdown()


if __name__ == "__main__":
    print("🚀 Starting RAG-Topic Integration Tests")
    print("=" * 80)
    
    # Run the tests
    asyncio.run(test_rag_topic_integration())
    asyncio.run(test_topic_agent_standalone())
    
    print("\n✅ All tests completed!")
