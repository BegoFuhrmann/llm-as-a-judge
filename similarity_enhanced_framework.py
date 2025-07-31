"""
Test script for enhanced framework using actual evaluation data.

This script tests the improved configuration management and semantic similarity
features with the user's actual test data from test_snippets.xlsx.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import os

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rag_agentic_evaluation.advanced_config import (
    AdvancedEvalConfig, 
    ConfigurationProfile, 
    SemanticSimilarityMethod
)
from rag_agentic_evaluation.semantic_similarity import SemanticSimilarityCalculator
from rag_agentic_evaluation.models import EvaluationInput, SourceSnippet
from rag_agentic_evaluation.evaluation import evaluate_answers_with_snippets
from rag_agentic_evaluation.utils import calculate_snippet_grounding_score
from rag_agentic_evaluation.excel_processor import load_excel_with_snippets


def load_test_data(excel_file: str) -> List[EvaluationInput]:
    """Load test data from Excel file and convert to EvaluationInput format."""
    print(f"📁 Loading test data from: {excel_file}")
    
    # Load Excel file
    df = pd.read_excel(excel_file)
    print(f"✅ Loaded {len(df)} test cases")
    print(f"📊 Columns: {list(df.columns)}")
    
    evaluation_inputs = []
    
    for idx, row in df.iterrows():
        question = str(row['Question'])
        answer = str(row['Answer'])
        snippet_text = str(row['Snippet'])
        
        # Parse snippet text into SourceSnippet objects
        # Assuming snippets are numbered like "1. content... 2. content..."
        snippets = []
        snippet_parts = snippet_text.split('\n') if '\n' in snippet_text else [snippet_text]
        
        for i, snippet_part in enumerate(snippet_parts):
            if snippet_part.strip():  # Skip empty lines
                snippet = SourceSnippet(
                    snippet_id=f"snippet_{idx}_{i+1}",
                    content=snippet_part.strip(),
                    source_document=f"Test Source {idx+1}",
                    relevance_score=0.9  # Default high relevance for test data
                )
                snippets.append(snippet)
        
        eval_input = EvaluationInput(
            query=question,
            answer=answer,
            source_snippets=snippets,
            metadata={
                "test_case_id": idx + 1,
                "original_row": idx
            }
        )
        
        evaluation_inputs.append(eval_input)
        print(f"  📝 Case {idx+1}: {question[:50]}... ({len(snippets)} snippets)")
    
    return evaluation_inputs


def test_similarity_methods(test_data: List[EvaluationInput]) -> Dict[str, List[float]]:
    """Test different semantic similarity methods on the test data."""
    print("\n🔍 Testing Semantic Similarity Methods")
    print("=" * 50)
    
    similarity_results = {}
    
    for method in SemanticSimilarityMethod:
        print(f"\n🎯 Testing {method.value}...")
        method_scores = []
        
        try:
            calculator = SemanticSimilarityCalculator(method)
            
            for i, eval_input in enumerate(test_data):
                # Test similarity calculation
                if eval_input.source_snippets:
                    # Calculate grounding score using this method
                    score = calculate_snippet_grounding_score(
                        eval_input.answer,
                        eval_input.source_snippets,
                        use_enhanced_similarity=True,
                        similarity_method=method
                    )
                    method_scores.append(score)
                    print(f"  📊 Case {i+1}: {score:.3f}")
                else:
                    print(f"  ⚠️  Case {i+1}: No snippets available")
            
            similarity_results[method.value] = method_scores
            avg_score = sum(method_scores) / len(method_scores) if method_scores else 0
            print(f"  📈 Average score: {avg_score:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error with {method.value}: {e}")
            similarity_results[method.value] = []
    
    return similarity_results


def test_configuration_profiles(test_data: List[EvaluationInput]) -> Dict[str, Any]:
    """Test different configuration profiles."""
    print("\n⚙️  Testing Configuration Profiles")
    print("=" * 50)
    
    profile_results = {}
    
    for profile in ConfigurationProfile:
        print(f"\n🏷️  Testing {profile.value} profile...")
        
        try:
            # Create configuration for this profile
            config = AdvancedEvalConfig.create_profile_config(profile)
            
            # Display key configuration settings
            print(f"  📝 Model: {config.llm_provider.model_name}")
            print(f"  🌡️  Temperature: {config.llm_provider.temperature}")
            print(f"  🔍 Similarity method: {config.semantic_similarity.method.value}")
            print(f"  📊 Confidence threshold: {config.evaluation.confidence_threshold}")
            
            # Test configuration validation
            is_valid, errors = config.validate()
            print(f"  ✅ Configuration valid: {is_valid}")
            if errors:
                print(f"  ⚠️  Validation errors: {errors}")
            
            profile_results[profile.value] = {
                "config": {
                    "model": config.llm_provider.model_name,
                    "temperature": config.llm_provider.temperature,
                    "similarity_method": config.semantic_similarity.method.value,
                    "confidence_threshold": config.evaluation.confidence_threshold
                },
                "valid": is_valid,
                "errors": errors
            }
            
        except Exception as e:
            print(f"  ❌ Error with {profile.value}: {e}")
            profile_results[profile.value] = {"error": str(e)}
    
    return profile_results


def run_enhanced_evaluation_demo(test_data: List[EvaluationInput]) -> Dict[str, Any]:
    """Run a demonstration evaluation using the enhanced framework."""
    print("\n🚀 Running Enhanced Evaluation Demo")
    print("=" * 50)
    
    # Create academic research configuration (most rigorous)
    config = AdvancedEvalConfig.create_profile_config(ConfigurationProfile.ACADEMIC_RESEARCH)
    config.semantic_similarity.method = SemanticSimilarityMethod.HYBRID_WEIGHTED
    
    print(f"📋 Using configuration:")
    print(f"  🏷️  Profile: {config.profile.value}")
    print(f"  🔍 Similarity: {config.semantic_similarity.method.value}")
    print(f"  🌡️  Temperature: {config.llm_provider.temperature}")
    
    demo_results = {
        "configuration": {
            "profile": config.profile.value,
            "similarity_method": config.semantic_similarity.method.value,
            "temperature": config.llm_provider.temperature
        },
        "evaluations": []
    }
    
    # Note: Since we don't have LLM access in this demo, we'll focus on
    # the enhanced semantic similarity and configuration features
    print("\n📊 Calculating Enhanced Grounding Scores:")
    
    for i, eval_input in enumerate(test_data):
        print(f"\n📝 Test Case {i+1}: {eval_input.query[:60]}...")
        
        if eval_input.source_snippets:
            # Calculate enhanced grounding score
            enhanced_score = calculate_snippet_grounding_score(
                eval_input.answer,
                eval_input.source_snippets,
                use_enhanced_similarity=True
            )
            
            # Calculate basic grounding score for comparison
            basic_score = calculate_snippet_grounding_score(
                eval_input.answer,
                eval_input.source_snippets,
                use_enhanced_similarity=False
            )
            
            print(f"  🔍 Enhanced grounding score: {enhanced_score:.3f}")
            print(f"  📊 Basic grounding score: {basic_score:.3f}")
            print(f"  📈 Improvement: {((enhanced_score - basic_score) / basic_score * 100):.1f}%")
            
            case_result = {
                "case_id": i + 1,
                "query": eval_input.query,
                "answer_length": len(eval_input.answer),
                "snippet_count": len(eval_input.source_snippets),
                "enhanced_grounding_score": enhanced_score,
                "basic_grounding_score": basic_score,
                "improvement_percentage": ((enhanced_score - basic_score) / basic_score * 100) if basic_score > 0 else 0
            }
            
            demo_results["evaluations"].append(case_result)
        else:
            print(f"  ⚠️  No snippets available for evaluation")
    
    return demo_results


def save_test_results(results: Dict[str, Any], output_dir: Path, timestamp: str):
    """Save test results to organized subfolder structure."""
    print(f"\n💾 Saving results to: {output_dir}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy float32 to regular Python float for JSON serialization
    def convert_floats(obj):
        if isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        return obj
    
    # Convert the results to ensure JSON compatibility
    json_safe_results = convert_floats(results)
    
    # Save main results file
    results_file = output_dir / "test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
    
    # Save individual component results for easy analysis
    if "similarity_method_testing" in results:
        similarity_file = output_dir / "similarity_method_results.json"
        with open(similarity_file, 'w', encoding='utf-8') as f:
            json.dump(json_safe_results["similarity_method_testing"], f, indent=2, ensure_ascii=False)
    
    if "configuration_profile_testing" in results:
        config_file = output_dir / "configuration_profile_results.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(json_safe_results["configuration_profile_testing"], f, indent=2, ensure_ascii=False)
    
    if "enhanced_evaluation_demo" in results:
        demo_file = output_dir / "enhanced_evaluation_demo.json"
        with open(demo_file, 'w', encoding='utf-8') as f:
            json.dump(json_safe_results["enhanced_evaluation_demo"], f, indent=2, ensure_ascii=False)
    
    # Create a summary report
    summary_file = output_dir / "test_summary.md"
    create_test_summary_report(results, summary_file, timestamp)
    
    print(f"✅ Results saved successfully to {output_dir}")
    print(f"📄 Files created:")
    print(f"  📊 test_results.json - Complete test results")
    print(f"  🔍 similarity_method_results.json - Similarity method comparison")
    print(f"  ⚙️  configuration_profile_results.json - Configuration validation")
    print(f"  🚀 enhanced_evaluation_demo.json - Enhanced vs basic comparison")
    print(f"  📋 test_summary.md - Human-readable summary report")


def create_test_summary_report(results: Dict[str, Any], output_file: Path, timestamp: str):
    """Create a human-readable summary report."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Enhanced Framework Test Report\n\n")
        f.write(f"**Test Date:** {timestamp}\n")
        f.write(f"**Framework Version:** {results.get('test_metadata', {}).get('framework_version', 'Unknown')}\n\n")
        
        # Test data summary
        summary = results.get('test_data_summary', {})
        f.write(f"## Test Data Summary\n\n")
        f.write(f"- **Total Test Cases:** {summary.get('total_cases', 0)}\n")
        f.write(f"- **Cases with Snippets:** {summary.get('cases_with_snippets', 0)}\n")
        f.write(f"- **Average Snippets per Case:** {summary.get('average_snippet_count', 0):.1f}\n\n")
        
        # Similarity method results
        similarity = results.get('similarity_method_testing', {})
        if similarity:
            f.write(f"## Semantic Similarity Method Performance\n\n")
            f.write(f"| Method | Average Score | Performance |\n")
            f.write(f"|--------|---------------|-------------|\n")
            
            for method, scores in similarity.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    f.write(f"| {method.replace('_', ' ').title()} | {avg_score:.3f} | ")
                    
                    # Add performance notes
                    if method == "token_overlap":
                        f.write("Fast baseline |\n")
                    elif method == "cosine_similarity":
                        f.write("Medium speed |\n")
                    elif method == "sentence_transformers":
                        f.write("Best accuracy |\n")
                    elif method == "hybrid_weighted":
                        f.write("Balanced approach |\n")
                    else:
                        f.write("Unknown |\n")
            f.write("\n")
        
        # Configuration profiles
        configs = results.get('configuration_profile_testing', {})
        if configs:
            f.write(f"## Configuration Profile Validation\n\n")
            valid_count = sum(1 for cfg in configs.values() if cfg.get('valid', False))
            f.write(f"**Validation Results:** {valid_count}/{len(configs)} profiles valid\n\n")
            
            for profile, config_data in configs.items():
                status = "✅" if config_data.get('valid', False) else "❌"
                f.write(f"- {status} **{profile.replace('_', ' ').title()}**")
                if config_data.get('config'):
                    cfg = config_data['config']
                    f.write(f" (Model: {cfg.get('model', 'Unknown')}, ")
                    f.write(f"Temperature: {cfg.get('temperature', 'Unknown')}, ")
                    f.write(f"Similarity: {cfg.get('similarity_method', 'Unknown')})")
                f.write("\n")
            f.write("\n")
        
        # Enhanced evaluation demo
        demo = results.get('enhanced_evaluation_demo', {})
        if demo and demo.get('evaluations'):
            f.write(f"## Enhanced vs Basic Framework Comparison\n\n")
            evaluations = demo['evaluations']
            
            enhanced_scores = [e['enhanced_grounding_score'] for e in evaluations]
            basic_scores = [e['basic_grounding_score'] for e in evaluations]
            
            avg_enhanced = sum(enhanced_scores) / len(enhanced_scores)
            avg_basic = sum(basic_scores) / len(basic_scores)
            avg_improvement = sum(e['improvement_percentage'] for e in evaluations) / len(evaluations)
            
            f.write(f"**Overall Performance:**\n")
            f.write(f"- Enhanced Average Score: {avg_enhanced:.3f}\n")
            f.write(f"- Basic Average Score: {avg_basic:.3f}\n")
            f.write(f"- Average Improvement: {avg_improvement:.1f}%\n\n")
            
            f.write(f"**Per Test Case Results:**\n\n")
            f.write(f"| Case | Enhanced Score | Basic Score | Improvement |\n")
            f.write(f"|------|----------------|-------------|-------------|\n")
            
            for i, eval_result in enumerate(evaluations):
                case_id = eval_result.get('case_id', i+1)
                enhanced = eval_result['enhanced_grounding_score']
                basic = eval_result['basic_grounding_score']
                improvement = eval_result['improvement_percentage']
                f.write(f"| Case {case_id} | {enhanced:.3f} | {basic:.3f} | {improvement:+.1f}% |\n")
        
        f.write(f"\n---\n")
        f.write(f"*Generated by Enhanced Framework Test Suite v2.0*\n")


def main():
    """Main test execution function."""
    # Create timestamp for this test run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamp_readable = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("🧪 Enhanced Framework Testing")
    print("=" * 60)
    print(f"📅 Test Date: {timestamp_readable}")
    
    # Create organized output directory
    output_base_dir = Path("enhanced_framework_test_results")
    output_dir = output_base_dir / f"test_run_{timestamp}"
    
    print(f"📁 Output Directory: {output_dir}")
    
    try:
        # Load test data
        test_data = load_test_data("test_snippets.xlsx")
        
        if not test_data:
            print("❌ No test data loaded. Exiting.")
            return
        
        # Test results container
        all_results = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_cases_count": len(test_data),
                "framework_version": "2.0 (Enhanced)",
                "output_directory": str(output_dir)
            },
            "test_data_summary": {
                "total_cases": len(test_data),
                "cases_with_snippets": sum(1 for td in test_data if td.source_snippets),
                "average_snippet_count": sum(len(td.source_snippets) for td in test_data) / len(test_data)
            }
        }
        
        # Test 1: Semantic Similarity Methods
        print("\n" + "="*60)
        similarity_results = test_similarity_methods(test_data)
        all_results["similarity_method_testing"] = similarity_results
        
        # Test 2: Configuration Profiles
        print("\n" + "="*60)
        profile_results = test_configuration_profiles(test_data)
        all_results["configuration_profile_testing"] = profile_results
        
        # Test 3: Enhanced Evaluation Demo
        print("\n" + "="*60)
        demo_results = run_enhanced_evaluation_demo(test_data)
        all_results["enhanced_evaluation_demo"] = demo_results
        
        # Save results to organized directory structure
        save_test_results(all_results, output_dir, timestamp_readable)
        
        # Print summary
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        print(f"✅ Test cases processed: {len(test_data)}")
        print(f"📊 Similarity methods tested: {len(similarity_results)}")
        print(f"⚙️  Configuration profiles tested: {len(profile_results)}")
        
        if demo_results.get("evaluations"):
            enhanced_scores = [e["enhanced_grounding_score"] for e in demo_results["evaluations"]]
            basic_scores = [e["basic_grounding_score"] for e in demo_results["evaluations"]]
            avg_improvement = sum(e["improvement_percentage"] for e in demo_results["evaluations"]) / len(demo_results["evaluations"])
            
            print(f"🔍 Average enhanced score: {sum(enhanced_scores)/len(enhanced_scores):.3f}")
            print(f"📊 Average basic score: {sum(basic_scores)/len(basic_scores):.3f}")
            print(f"📈 Average improvement: {avg_improvement:.1f}%")
        
        print(f"💾 Results saved to: {output_dir}")
        print(f"📄 Open {output_dir / 'test_summary.md'} for detailed analysis")
        print("\n🎉 Enhanced framework testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
