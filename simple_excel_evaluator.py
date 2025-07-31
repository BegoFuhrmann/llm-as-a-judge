#!/usr/bin/env python3
"""
Simple Excel evaluator using Azure OpenAI directly (bypassing template issues)
"""

import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any

# Set Azure environment variables
os.environ["ENDPOINT_URL"] = "https://begobaiatest.openai.azure.com/"
os.environ["DEPLOYMENT_NAME"] = "gpt-4o-mini"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

def evaluate_answer_with_azure_openai(question: str, answer: str) -> Dict[str, Any]:
    """Evaluate a single answer using Azure OpenAI directly."""
    
    from openai import AzureOpenAI
    from azure.identity import ClientSecretCredential, get_bearer_token_provider
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get configuration from environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT ", os.getenv("ENDPOINT_URL"))
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("DEPLOYMENT_NAME"))
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    
    # Service principal credentials
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    
    # Validate required environment variables
    if not all([tenant_id, client_id, client_secret]):
        raise ValueError("Missing required environment variables: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
    
    # Initialize Azure credential with service principal
    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret
    )
    
    # Create token provider for Azure OpenAI
    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default"
    )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )
    
    # Academic evaluation prompt
    system_prompt = """You are an expert academic evaluator specializing in AI system assessment. 
    
    Evaluate the provided answer based on these dimensions:
    - factual_accuracy: Is the information correct and verifiable? (1-5)
    - relevance: How well does the answer address the question? (1-5)
    - completeness: Is all necessary information included? (1-5)
    - clarity: Is the answer clear and understandable? (1-5)
    - coherence: Is the response logically structured? (1-5)
    - citation_quality: Quality of references or sources mentioned (1-5)
    - reasoning_depth: Depth of reasoning and analysis (1-5)
    
    Respond with a JSON object containing:
    {
        "scores": {
            "factual_accuracy": <score 1-5>,
            "relevance": <score 1-5>,
            "completeness": <score 1-5>,
            "clarity": <score 1-5>,
            "coherence": <score 1-5>,
            "citation_quality": <score 1-5>,
            "reasoning_depth": <score 1-5>
        },
        "justifications": {
            "factual_accuracy": "<brief justification>",
            "relevance": "<brief justification>",
            "completeness": "<brief justification>",
            "clarity": "<brief justification>",
            "coherence": "<brief justification>",
            "citation_quality": "<brief justification>",
            "reasoning_depth": "<brief justification>"
        },
        "confidence_scores": {
            "factual_accuracy": <confidence 0.0-1.0>,
            "relevance": <confidence 0.0-1.0>,
            "completeness": <confidence 0.0-1.0>,
            "clarity": <confidence 0.0-1.0>,
            "coherence": <confidence 0.0-1.0>,
            "citation_quality": <confidence 0.0-1.0>,
            "reasoning_depth": <confidence 0.0-1.0>
        },
        "overall_assessment": "<brief overall assessment>",
        "weighted_total": <calculated weighted score>
    }"""
    
    user_content = {
        "query": question,
        "answer": answer
    }
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_content, indent=2)}
    ]
    
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=1500,
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(completion.choices[0].message.content)
    
    # Calculate weighted total if not provided
    if "weighted_total" not in result:
        scores = result["scores"]
        # Academic weights
        weights = {
            "factual_accuracy": 0.20,
            "relevance": 0.20,
            "completeness": 0.15,
            "clarity": 0.15,
            "coherence": 0.10,
            "citation_quality": 0.10,
            "reasoning_depth": 0.10
        }
        weighted_total = sum(scores[dim] * weights.get(dim, 0.1) for dim in scores)
        result["weighted_total"] = weighted_total
    
    # Add metadata
    result["metadata"] = {
        "model_used": completion.model,
        "tokens_used": completion.usage.total_tokens,
        "system_type": "rag"
    }
    
    return result

def main():
    """Main function to evaluate Excel file."""
    
    # Load Excel file
    excel_file = "input_questions.xlsx"
    print(f"📊 Loading Excel file: {excel_file}")
    
    try:
        df = pd.read_excel(excel_file)
        print(f"✅ Loaded {len(df)} rows with columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return 1
    
    # Validate structure
    if 'Question' not in df.columns or 'Answer' not in df.columns:
        print("❌ Excel file must have 'Question' and 'Answer' columns")
        return 1
    
    # Evaluate each question-answer pair
    results = []
    
    for idx, row in df.iterrows():
        question = str(row['Question']).strip()
        answer = str(row['Answer']).strip()
        
        if not question or not answer or pd.isna(row['Question']) or pd.isna(row['Answer']):
            print(f"⚠️  Skipping row {idx + 1}: missing question or answer")
            continue
        
        print(f"\n🔍 Evaluating question {idx + 1}: {question[:60]}...")
        
        try:
            evaluation = evaluate_answer_with_azure_openai(question, answer)
            evaluation["question"] = question
            evaluation["answer"] = answer
            evaluation["row_number"] = idx + 1
            
            results.append(evaluation)
            
            print(f"✅ Completed - Score: {evaluation['weighted_total']:.2f}")
            
        except Exception as e:
            print(f"❌ Error evaluating row {idx + 1}: {e}")
            continue
    
        # Generate report
        if results:
            print(f"\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            
            scores = [r['weighted_total'] for r in results]
            print(f"Total evaluations: {len(results)}")
            print(f"Average score: {sum(scores)/len(scores):.2f}")
            print(f"Highest score: {max(scores):.2f}")
            print(f"Lowest score: {min(scores):.2f}")
            
            # Create results directory
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = f"evaluation_results_{timestamp}"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save detailed results
            output_file = os.path.join(results_dir, "excel_evaluation_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\nDetailed results saved to: {output_file}")
            
            # Generate CSV summary
            csv_data = []
            for r in results:
                row_data = {
                    "Question": r["question"],
                    "Answer": r["answer"],
                    "Weighted_Score": r["weighted_total"],
                    "Factual_Accuracy": r["scores"]["factual_accuracy"],
                    "Relevance": r["scores"]["relevance"],
                    "Completeness": r["scores"]["completeness"],
                    "Clarity": r["scores"]["clarity"],
                    "Coherence": r["scores"]["coherence"],
                    "Citation_Quality": r["scores"]["citation_quality"],
                    "Reasoning_Depth": r["scores"]["reasoning_depth"],
                    "Overall_Assessment": r["overall_assessment"]
                }
                csv_data.append(row_data)
            
            csv_df = pd.DataFrame(csv_data)
            csv_file = os.path.join(results_dir, "excel_evaluation_summary.csv")
            csv_df.to_csv(csv_file, index=False)
            print(f"CSV summary saved to: {csv_file}")
            
            print("="*60)
            
        else:
            print("❌ No evaluations completed successfully")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
