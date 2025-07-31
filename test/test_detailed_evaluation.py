#!/usr/bin/env python3
"""
Detailed test of the evaluation with template debugging
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
    from rag_agentic_evaluation import EvalConfig, SystemType, evaluate_answers
    
    print("✅ Successfully imported evaluation framework")
    
    # Test configuration and template
    config = EvalConfig()
    print(f"\n📋 Configuration details:")
    print(f"   Model: {config.model_name}")
    print(f"   Dimensions: {[dim.value for dim in config.dimensions]}")
    print(f"   Rubrics keys: {list(config.rubrics.keys())}")
    
    # Test template loading
    from rag_agentic_evaluation.templates import load_system_prompt
    try:
        prompt = load_system_prompt(config.system_prompt_template, config)
        print(f"✅ Template loaded successfully (length: {len(prompt)} chars)")
        print(f"   Template preview: {prompt[:200]}...")
    except Exception as template_error:
        print(f"❌ Template error: {template_error}")
        sys.exit(1)
    
    # Test simple evaluation
    print(f"\n🧪 Testing simple evaluation...")
    query = "When can I pick up the laptops?"
    answer = "Next week"
    
    try:
        evaluations = evaluate_answers(
            query=query,
            answers=[answer],
            config=config,
            system_types=[SystemType.RAG]
        )
        
        if evaluations:
            evaluation = evaluations[0]
            print("✅ Evaluation completed successfully!")
            print(f"   Weighted score: {evaluation.weighted_total:.2f}")
            print(f"   Scores: {[(dim.value, score) for dim, score in evaluation.scores.items()]}")
        else:
            print("❌ No evaluations returned")
            
    except Exception as eval_error:
        print(f"❌ Evaluation error: {eval_error}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
