#!/usr/bin/env python3
"""
Simple test to see what JSON response we get from Azure OpenAI
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

def main():
    # Load environment variables from .env file
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    
    # Set NO_PROXY for local development
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    
    # Get configuration from environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    
    print(f"Using endpoint: {endpoint}")
    print(f"Using deployment: {deployment}")
    
    # Use Azure CLI credentials for this test
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default"
    )
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )
    
    # Simple test prompt to return JSON
    test_prompt = """Please evaluate this query-answer pair and respond in JSON format:

Query: "Where can I park?"
Answer: "There are parking spaces available in the garage."

Return a JSON object with scores (1-5) for factual_accuracy and relevance."""
    
    try:
        print("Testing JSON response format...")
        
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON objects."},
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        print(f"\nRaw response content:")
        print("=" * 50)
        print(response_content)
        print("=" * 50)
        
        # Try to parse as JSON
        try:
            parsed_json = json.loads(response_content)
            print(f"\n✅ Successfully parsed JSON:")
            print(json.dumps(parsed_json, indent=2))
            return 0
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON parsing failed: {e}")
            print("This indicates the response format is not valid JSON")
            return 1
            
    except Exception as e:
        print(f"❌ Error during API call: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
