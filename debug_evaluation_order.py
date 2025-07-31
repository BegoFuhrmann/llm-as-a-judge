#!/usr/bin/env python3
"""
Debug script to check the evaluation order and see if there's a mismatch
between the queries and the evaluation results.
"""

import pandas as pd
import json
from pathlib import Path

def debug_evaluation_order():
    """Check if evaluations are correctly matched to their inputs."""
    
    # Load Excel data
    print("=== Loading Excel Data ===")
    df = pd.read_excel('test_snippets.xlsx')
    print(f"Excel shape: {df.shape}")
    print("Excel rows:")
    for i, row in df.iterrows():
        print(f"  Row {i+1}: {row['Question']}")
    
    # Load evaluation results
    print("\n=== Loading Evaluation Results ===")
    result_file = 'snippet_evaluation_results_20250730_093237/detailed_results.json'
    if Path(result_file).exists():
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Results count: {len(data['detailed_results'])}")
        print("Result items:")
        for result in data['detailed_results']:
            item_num = result['item_number']
            query = result['query']
            factual_justification = result['justifications']['factual_accuracy'][:100]
            print(f"  Item {item_num}: {query}")
            print(f"    Factual justification: {factual_justification}...")
            
            # Check for mismatches
            if 'parking' in query.lower() and 'cafeteria' in factual_justification.lower():
                print(f"    ❌ MISMATCH: Parking query has cafeteria justification!")
            elif 'cafeteria' in query.lower() and 'parking' in factual_justification.lower():
                print(f"    ❌ MISMATCH: Cafeteria query has parking justification!")
            else:
                print(f"    ✅ Match appears correct")
            print()
    else:
        print(f"Result file not found: {result_file}")

if __name__ == "__main__":
    debug_evaluation_order()
