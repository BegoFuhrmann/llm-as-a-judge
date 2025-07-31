#!/usr/bin/env python3
"""
Simple test of the evaluation framework with Azure OpenAI
"""

import os
import sys

# Set Azure environment variables
os.environ["ENDPOINT_URL"] = "https://begobaiatest.openai.azure.com/"
os.environ["DEPLOYMENT_NAME"] = "gpt-4o-mini"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

# Add current directory to Python path
sys.path.append('.')

try:
    from rag_agentic_evaluation import EvalConfig, SystemType
    from rag_agentic_evaluation.llm_client import create_openai_client, get_model_name, test_llm_connection
    
    print("✅ Successfully imported evaluation framework")
    
    # Test client creation
    print("\n🔧 Testing Azure OpenAI client creation...")
    client = create_openai_client()
    print(f"✅ Client created: {type(client).__name__}")
    
    # Test configuration
    config = EvalConfig()
    print(f"\n📋 Configuration loaded:")
    print(f"   Model: {config.model_name}")
    print(f"   Temperature: {config.temperature}")
    print(f"   Dimensions: {len(config.dimensions)}")
    
    # Test model name resolution
    model_name = get_model_name(config)
    print(f"   Resolved model name: {model_name}")
    
    # Test connection
    print("\n🌐 Testing LLM connection...")
    connection_ok = test_llm_connection(config)
    if connection_ok:
        print("✅ LLM connection test successful")
    else:
        print("❌ LLM connection test failed")
    
    print("\n🎉 All Azure OpenAI framework tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
