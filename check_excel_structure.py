#!/usr/bin/env python3
"""
Quick Excel file structure checker for the evaluation framework.
"""

import pandas as pd
import sys

def check_excel_structure(file_path: str):
    """Check the structure of the Excel file."""
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        print(f"Excel file: {file_path}")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print()
        
        # Show first few rows
        print("First 3 rows:")
        print(df.head(3))
        print()
        
        # Check for Question and Answer columns (case insensitive)
        columns_lower = [col.lower() for col in df.columns]
        has_question = any('question' in col or 'query' in col for col in columns_lower)
        has_answer = any('answer' in col or 'response' in col for col in columns_lower)
        
        print("Structure validation:")
        print(f"✓ Has question column: {has_question}")
        print(f"✓ Has answer column: {has_answer}")
        
        # Check for missing values
        if has_question and has_answer:
            question_col = None
            answer_col = None
            
            for i, col in enumerate(df.columns):
                col_lower = col.lower()
                if 'question' in col_lower or 'query' in col_lower:
                    question_col = col
                elif 'answer' in col_lower or 'response' in col_lower:
                    answer_col = col
            
            if question_col and answer_col:
                missing_questions = df[question_col].isna().sum()
                missing_answers = df[answer_col].isna().sum()
                
                print(f"Missing questions: {missing_questions}")
                print(f"Missing answers: {missing_answers}")
                print(f"Valid question-answer pairs: {len(df) - max(missing_questions, missing_answers)}")
        
        return True
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return False

if __name__ == "__main__":
    file_path = "input_questions.xlsx"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    check_excel_structure(file_path)
