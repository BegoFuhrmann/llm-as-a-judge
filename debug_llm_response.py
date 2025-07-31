#!/usr/bin/env python3
"""
Debug script to test LLM response format
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    env_path = Path(__file__).parent / ".env"
    print(f"Loading environment variables from: {env_path}")
    load_dotenv(env_path)
    
    # Map your .env variables to the framework's expected names
    env_mapping = {
        'ENDPOINT_URL': 'AZURE_OPENAI_ENDPOINT',
        'DEPLOYMENT_NAME': 'AZURE_OPENAI_DEPLOYMENT',
        'AZURE_TENANT_ID': 'AZURE_TENANT_ID',
        'AZURE_CLIENT_ID': 'AZURE_CLIENT_ID', 
        'AZURE_CLIENT_SECRET': 'AZURE_CLIENT_SECRET'
    }
    
    # Set the mapped environment variables
    for framework_var, env_var in env_mapping.items():
        value = os.getenv(env_var)
        if value:
            os.environ[framework_var] = value
    
    print("Testing LLM response format...")
    
    # Import and test the LLM client
    try:
        from rag_agentic_evaluation.llm_client import call_llm_evaluator
        from rag_agentic_evaluation.models import SourceSnippet
        from rag_agentic_evaluation.config import EvalConfig
        
        # Create config
        config = EvalConfig()
        
        # Simple test data
        query = "Where can I park my car?"
        answer = "There are several parking options available including internal parking spaces."
        snippets = [
            SourceSnippet(content="Several cars are parked with parking spaces", snippet_id="1")
        ]
        
        print("Calling LLM evaluator...")
        result = call_llm_evaluator(query, answer, config, snippets=snippets)
        
        print("LLM Response received:")
        print(f"Type: {type(result)}")
        print(f"Content: {result}")
        
        if hasattr(result, '__dict__'):
            print("Result attributes:")
            for attr, value in result.__dict__.items():
                print(f"  {attr}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
