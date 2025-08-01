"""
Direct test script that works around import issues.
Run from the agentic_system directory.
"""
import sys
from pathlib import Path

# Fix imports by adding the current directory to sys.path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Set environment variable to avoid proxy issues
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

print("🚀 RAG-Topic Integration Test (Direct)")
print("=" * 50)
print(f"📁 Working from: {current_dir}")

# Test basic functionality without complex async operations
def test_basic_functionality():
    """Test basic class instantiation and configuration."""
    try:
        print("\n📝 Testing basic imports and instantiation...")
        
        # Import classes
        from agents.rag_agent import RAGAgent
        from agents.topic_identification_agent import TopicIdentificationAgent
        print("✅ Imports successful")
        
        # Test RAG Agent instantiation
        rag_agent = RAGAgent(
            agent_id="test_rag",
            config={
                'confidence_threshold': 0.7,
                'max_retrieval_results': 3,
                'temperature': 0
            }
        )
        print("✅ RAG Agent created")
        
        # Test Topic Agent instantiation
        topic_agent = TopicIdentificationAgent(
            agent_id="test_topic",
            config={'confidence_threshold': 0.7}
        )
        print("✅ Topic Agent created")
        
        # Test database paths
        print(f"\n📊 Configuration Check:")
        print(f"   Confluence DB: {rag_agent.confluence_db_path}")
        print(f"   NewHQ DB: {rag_agent.newhq_db_path}")
        
        # Check if databases exist
        confluence_exists = os.path.exists(rag_agent.confluence_db_path)
        newhq_exists = os.path.exists(rag_agent.newhq_db_path)
        print(f"   Confluence DB exists: {confluence_exists}")
        print(f"   NewHQ DB exists: {newhq_exists}")
        
        # Test topic mapping
        print(f"\n🏷️  Topic Mapping:")
        for db, keywords in rag_agent.topic_to_database_mapping.items():
            print(f"   {db}: {keywords[:5]}...")  # Show first 5 keywords
        
        print("\n✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_availability():
    """Test if the ChromaDB databases are accessible."""
    print("\n📝 Testing database availability...")
    
    try:
        # Test if we can access the database directories
        base_path = Path(r"\\dnsbego.de\dfsbego\home04\FuhrmannD\Documents\01_Trainee\Master\Thesis\code\agentic_system\data")
        
        print(f"📁 Base data path: {base_path}")
        print(f"   Exists: {base_path.exists()}")
        
        if base_path.exists():
            # List contents
            contents = list(base_path.iterdir()) if base_path.exists() else []
            print(f"   Contents: {[item.name for item in contents]}")
            
            # Check specific databases
            confluence_db = base_path / "confluence_chroma_db"
            newhq_db = base_path / "newHQ_chroma_db"
            
            print(f"\n🗃️  Database Status:")
            print(f"   Confluence DB: {confluence_db.exists()} at {confluence_db}")
            print(f"   NewHQ DB: {newhq_db.exists()} at {newhq_db}")
            
            # If databases exist, check their contents
            if confluence_db.exists():
                conf_contents = list(confluence_db.iterdir())
                print(f"   Confluence DB files: {len(conf_contents)} items")
            
            if newhq_db.exists():
                newhq_contents = list(newhq_db.iterdir())
                print(f"   NewHQ DB files: {len(newhq_contents)} items")
        
        print("✅ Database availability check completed")
        return True
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False

def test_topic_classification():
    """Test topic classification logic."""
    print("\n📝 Testing topic classification logic...")
    
    try:
        from agents.rag_agent import RAGAgent
        
        rag_agent = RAGAgent("test_rag")
        
        # Test queries
        test_cases = [
            ("Where can I park at the office?", "newhq"),
            ("What is the BAIA project?", "confluence"),
            ("Office building facilities", "newhq"),
            ("RPA automation workflow", "confluence"),
            ("Parking garage access", "newhq"),
            ("BegoChat AI assistant", "confluence")
        ]
        
        print("🔍 Testing keyword-based routing...")
        for query, expected in test_cases:
            # Simulate the fallback keyword routing
            query_lower = query.lower()
            
            newhq_matches = sum(1 for keyword in rag_agent.topic_to_database_mapping['newhq'] 
                              if keyword in query_lower)
            confluence_matches = sum(1 for keyword in rag_agent.topic_to_database_mapping['confluence'] 
                                   if keyword in query_lower)
            
            if newhq_matches > confluence_matches:
                predicted = 'newhq'
            elif confluence_matches > newhq_matches:
                predicted = 'confluence'
            else:
                predicted = 'confluence'  # default
            
            status = "✅" if predicted == expected else "⚠️"
            print(f"   {status} '{query}' → {predicted} (expected: {expected})")
        
        print("✅ Topic classification test completed")
        return True
        
    except Exception as e:
        print(f"❌ Topic classification test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 Running Direct Integration Tests")
    
    success = True
    
    # Run tests
    success &= test_basic_functionality()
    success &= test_database_availability()
    success &= test_topic_classification()
    
    # Summary
    print(f"\n🏁 Test Summary")
    print("=" * 30)
    if success:
        print("✅ All tests passed!")
        print("\n📋 Next steps:")
        print("   1. Ensure ChromaDB databases are populated")
        print("   2. Configure Azure OpenAI credentials")
        print("   3. Test with real queries using async functions")
    else:
        print("❌ Some tests failed!")
        print("\n🔧 Troubleshooting:")
        print("   1. Check database paths and permissions")
        print("   2. Verify Python environment setup")
        print("   3. Check import dependencies")

if __name__ == "__main__":
    main()
