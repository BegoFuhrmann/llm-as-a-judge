"""
Simple test to verify the modular structure and imports work correctly.
"""

try:
    # Test imports from the package
    from evaluation_system import (
        EvalConfig, 
        SystemType, 
        EvaluationDimension,
        AnswerEvaluation,
        evaluate_answers
    )
    
    print("Core imports successful")

    # Test configuration creation
    config = EvalConfig()
    print(f"Config created with {len(config.dimensions)} dimensions")

    # Test enum values
    print(f"SystemType.RAG = {SystemType.RAG.value}")
    print(f"First dimension = {config.dimensions[0].value}")

    # Test rubric access
    first_dim = config.dimensions[0]
    rubric = config.rubrics[first_dim]
    print(f"Rubric for {first_dim.value}: weight = {rubric.weight}")

    print("\nAll modular components loaded successfully!")
    print("\nPackage structure verification complete.")
    
except ImportError as e:
    print(f"Import error: {e}")
    
except Exception as e:
    print(f"Error: {e}")
