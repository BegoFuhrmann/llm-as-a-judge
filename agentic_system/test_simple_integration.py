"""
Standalone test script for RAG-Topic Integration.
Run this script from the agentic_system directory.
"""
import asyncio
import sys
import os
from pathlib import Path

# Simple import fix - add current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Mock classes for testing if imports fail
class MockTask:
    def __init__(self, task_id, input_data, agent_type, priority='medium'):
        self.task_id = task_id
        self.input_data = input_data
        self.agent_type = agent_type
        self.priority = priority

class MockAgentType:
    RAG_BASED = "rag_based"
    TOPIC_IDENTIFICATION = "topic_identification"

# Try to import the real classes
try:
    from agents.rag_agent import RAGAgent
    from agents.topic_identification_agent import TopicIdentificationAgent
    from core.base import Task
    from enums import AgentType, Priority
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📋 Using mock classes for demonstration")
    # Use mock classes
    Task = MockTask
    AgentType = MockAgentType
    RAGAgent = None
    TopicIdentificationAgent = None

async def test_simple_functionality():
    """Test basic functionality without complex imports."""
    print("🚀 Starting Simple RAG-Topic Integration Test")
    print("=" * 60)
    
    if RAGAgent is None:
        print("❌ RAGAgent not available due to import issues")
        print("\n🔧 To fix imports, ensure you're running from the correct directory:")
        print("   cd agentic_system")
        print("   python test_simple_integration.py")
        return
    
    try:
        # Test 1: Initialize RAG Agent
        print("\n📝 Test 1: Initializing RAG Agent...")
        rag_agent = RAGAgent(
            agent_id="test_rag_agent",
            config={
                'confidence_threshold': 0.7,
                'max_retrieval_results': 3,
                'temperature': 0
            }
        )
        
        # Initialize
        print("🔄 Initializing agent...")
        init_success = await rag_agent.initialize()
        
        if init_success:
            print("✅ RAG Agent initialized successfully")
            
            # Get stats
            stats = rag_agent.get_knowledge_base_stats()
            print(f"📊 Knowledge Base Status:")
            for db_name, db_info in stats.items():
                if isinstance(db_info, dict):
                    status = db_info.get('status', 'unknown')
                    print(f"   {db_name}: {status}")
            
            # Test 2: Topic Identification
            print("\n📝 Test 2: Testing Topic Identification...")
            if rag_agent.topic_agent:
                test_queries = [
                    "Where can I park at the office?",
                    "What is the BAIA project?",
                    "How do I access building facilities?"
                ]
                
                for query in test_queries:
                    print(f"\n🔍 Query: '{query}'")
                    try:
                        recommendation = await rag_agent.topic_agent.get_database_recommendation(query)
                        print(f"   📍 Database: {recommendation['database']}")
                        print(f"   🎯 Confidence: {recommendation['confidence']:.2f}")
                        print(f"   🏷️  Topics: {recommendation['topics']}")
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
            else:
                print("⚠️  Topic agent not available")
            
            # Test 3: End-to-end processing
            print("\n📝 Test 3: End-to-end Query Processing...")
            test_task = Task(
                id="test_001",
                name="Test Query",
                description="Test parking query",
                input_data={'query': 'Where is parking available?'},
                priority=Priority.MEDIUM
            )
            
            try:
                response = await rag_agent.process(test_task)
                print(f"   ✅ Success: {response.success}")
                if response.success:
                    print(f"   🗃️  Database: {response.data.get('database_used', 'Unknown')}")
                    print(f"   📊 Confidence: {response.confidence_score:.2f}")
                    print(f"   📄 Answer Preview: {response.data.get('answer', 'No answer')[:100]}...")
                else:
                    print(f"   ❌ Error: {response.error_message}")
            except Exception as e:
                print(f"   ❌ Processing error: {e}")
            
            # Cleanup
            print("\n🔄 Shutting down...")
            await rag_agent.shutdown()
            print("✅ Test completed successfully")
            
        else:
            print("❌ Failed to initialize RAG Agent")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

async def test_topic_agent_only():
    """Test just the topic identification agent."""
    print("\n🔬 Testing Topic Identification Agent Only")
    print("=" * 50)
    
    if TopicIdentificationAgent is None:
        print("❌ TopicIdentificationAgent not available")
        return
    
    try:
        topic_agent = TopicIdentificationAgent(
            agent_id="test_topic_agent",
            config={'confidence_threshold': 0.7}
        )
        
        init_success = await topic_agent.initialize()
        if init_success:
            print("✅ Topic Agent initialized")
            
            test_queries = [
                "Where is the parking garage?",
                "How does BAIA work?",
                "Office facilities information",
                "RPA automation guide"
            ]
            
            for query in test_queries:
                print(f"\n🔍 '{query}'")
                try:
                    rec = await topic_agent.get_database_recommendation(query)
                    print(f"   Database: {rec['database']}")
                    print(f"   Confidence: {rec['confidence']:.2f}")
                    print(f"   Domain: {rec['domain']}")
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            
            await topic_agent.shutdown()
        else:
            print("❌ Failed to initialize Topic Agent")
            
    except Exception as e:
        print(f"❌ Topic agent test failed: {e}")

def main():
    """Main test function."""
    print("🎯 RAG-Topic Integration Test Suite")
    print("=" * 80)
    print(f"📁 Working Directory: {Path.cwd()}")
    print(f"🐍 Python Path: {sys.path[:3]}...")
    
    # Run tests
    try:
        asyncio.run(test_simple_functionality())
        asyncio.run(test_topic_agent_only())
        print("\n🎉 All tests completed!")
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
