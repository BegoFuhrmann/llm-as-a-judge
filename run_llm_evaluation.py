#!/usr/bin/env python3
"""
Script to load .env file and run LLM-as-a-judge evaluation on test_snippets.xlsx
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    env_path = Path(__file__).parent / ".env"
    print(f"Loading environment variables from: {env_path}")
    
    if not env_path.exists():
        print(f"ERROR: .env file not found at {env_path}")
        return 1
    
    # Load the .env file
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
            print(f"Set {framework_var} from {env_var}")
        else:
            print(f"WARNING: {env_var} not found in .env file")
    
    # Show what's configured
    print("\nConfigured environment variables:")
    for var in ['ENDPOINT_URL', 'DEPLOYMENT_NAME', 'AZURE_TENANT_ID', 'AZURE_CLIENT_ID']:
        value = os.environ.get(var, 'NOT SET')
        masked_value = value if var != 'AZURE_CLIENT_SECRET' else ('*' * len(value) if value != 'NOT SET' else 'NOT SET')
        print(f"  {var}: {masked_value}")
    
    print(f"\nRunning LLM evaluation on test_snippets.xlsx...")
    
    # Import and run the evaluation
    try:
        from rag_agentic_evaluation.main import main as eval_main
        
        # Set up command line arguments for the excel command
        sys.argv = [
            'rag_agentic_evaluation',
            'excel',
            '--input', 'test_snippets.xlsx',
            '--format', 'json'
        ]
        
        # Run the evaluation
        return eval_main()
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
