"""
Setup script to fix import issues and run tests.
"""
import sys
import os
from pathlib import Path

def setup_environment():
    """Set up the Python environment for proper imports."""
    # Get the agentic_system directory
    current_dir = Path(__file__).parent.absolute()
    
    # Add current directory to Python path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Add parent directory to Python path for package imports
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    print(f"✅ Environment set up")
    print(f"📁 Current directory: {current_dir}")
    print(f"🐍 Python path updated with: {current_dir}")
    
    return current_dir

def test_imports():
    """Test if all imports work correctly."""
    print("\n🔍 Testing imports...")
    
    try:
        from agents.rag_agent import RAGAgent
        print("✅ RAGAgent imported successfully")
    except ImportError as e:
        print(f"❌ RAGAgent import failed: {e}")
        return False
    
    try:
        from agents.topic_identification_agent import TopicIdentificationAgent
        print("✅ TopicIdentificationAgent imported successfully")
    except ImportError as e:
        print(f"❌ TopicIdentificationAgent import failed: {e}")
        return False
    
    try:
        from core.base import Task
        print("✅ Task imported successfully")
    except ImportError as e:
        print(f"❌ Task import failed: {e}")
        return False
    
    try:
        from enums import AgentType
        print("✅ AgentType imported successfully")
    except ImportError as e:
        print(f"❌ AgentType import failed: {e}")
        return False
    
    print("✅ All imports successful!")
    return True

def main():
    """Main setup and test function."""
    print("🚀 RAG-Topic Integration Setup and Test")
    print("=" * 50)
    
    # Setup environment
    current_dir = setup_environment()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check your environment.")
        return
    
    # Run the actual test
    print("\n🎯 Running integration test...")
    try:
        # Import and run the test
        import test_simple_integration
        print("✅ Test script imported successfully")
        
        # Run the main test
        test_simple_integration.main()
        
    except ImportError as e:
        print(f"❌ Failed to import test script: {e}")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
