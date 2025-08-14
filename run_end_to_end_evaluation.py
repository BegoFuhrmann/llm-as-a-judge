"""
End-to-End Evaluation Pipeline Test Script with Specialized Regulatory Agents.

This script orchestrates the full evaluation pipeline:
1. Loads evaluation data from an Excel file.
2. Configures the evaluation system with an advanced profile.
3. Initializes the core evaluation engine AND specialized regulatory agents.
4. Runs the evaluation for each data row using appropriate agents.
5. Generates and saves a comprehensive report.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from evaluation_system.excel_processor import load_excel_with_snippets
from evaluation_system.evaluation import evaluate_answers_with_snippets
from evaluation_system.advanced_config import AdvancedEvalConfig, ConfigurationProfile
from evaluation_system.report import generate_academic_citation_report
from evaluation_system.enums import EvaluationDimension
from evaluation_system.agents import GDPRComplianceAgent, EUAIActAgent, AuditTrailAgent

def evaluate_regulatory_dimensions(query: str, answer: str, snippets: list = None) -> dict:
    """
    Evaluate regulatory dimensions using specialized agents.
    
    Args:
        query: The user query
        answer: The system response
        snippets: List of source snippets (if available)
    
    Returns:
        Dictionary with regulatory evaluation results from specialized agents
    """
    print("🏛️ Running specialized regulatory agent evaluations...")
    
    regulatory_results = {}
    
    # Prepare context for evaluation
    context = f"""
    User Query: {query}
    
    System Response: {answer}
    
    Source Information: {' '.join([s.content for s in snippets]) if snippets else 'No source snippets provided'}
    """
    
    try:
        # 1. GDPR Compliance Assessment
        print("   📋 Evaluating GDPR compliance...")
        gdpr_db_path = "agentic_system/data/gdpr_chroma_db"
        
        # Check if knowledge base exists
        if not os.path.exists(gdpr_db_path):
            print(f"     ⚠️ GDPR knowledge base not found at: {gdpr_db_path}")
            print(f"     Using LLM-based assessment without RAG knowledge base")
            gdpr_agent = GDPRComplianceAgent()  # Fallback to LLM-only mode
        else:
            print(f"     📚 Using GDPR knowledge base at: {gdpr_db_path}")
            gdpr_agent = GDPRComplianceAgent(db_path=gdpr_db_path)
            
        gdpr_results = gdpr_agent.assess_gdpr_compliance(context)
        regulatory_results['gdpr_compliance'] = {
            'agent_score': gdpr_results.get('compliance_score', 0) / 20,  # Convert to 1-5 scale
            'risk_level': gdpr_results.get('risk_level', 'unknown'),
            'detailed_assessment': gdpr_results.get('detailed_assessments', {}),
            'recommendations': gdpr_results.get('recommendations', []),
            'agent_used': 'GDPRComplianceAgent'
        }
        print(f"     ✅ GDPR score: {regulatory_results['gdpr_compliance']['agent_score']:.2f}/5")
        
    except Exception as e:
        print(f"     ❌ GDPR evaluation failed: {e}")
        regulatory_results['gdpr_compliance'] = {'agent_score': None, 'error': str(e)}
    
    try:
        # 2. EU AI Act Alignment Assessment  
        print("   🤖 Evaluating EU AI Act alignment...")
        eu_ai_db_path = "agentic_system/data/eu_ai_act_chroma_db"
        
        # Check if knowledge base exists
        if not os.path.exists(eu_ai_db_path):
            print(f"     ⚠️ EU AI Act knowledge base not found at: {eu_ai_db_path}")
            print(f"     Using LLM-based assessment without RAG knowledge base")
            eu_ai_agent = EUAIActAgent()  # Fallback to LLM-only mode
        else:
            print(f"     📚 Using EU AI Act knowledge base at: {eu_ai_db_path}")
            eu_ai_agent = EUAIActAgent(db_path=eu_ai_db_path)
            
        eu_ai_results = eu_ai_agent.assess_eu_ai_act_compliance(context)
        regulatory_results['eu_ai_act_alignment'] = {
            'agent_score': eu_ai_results.get('compliance_score', 0) / 20,  # Convert to 1-5 scale
            'risk_categorization': eu_ai_results.get('risk_categorization', 'unknown'),
            'detailed_assessment': eu_ai_results.get('detailed_assessments', {}),
            'recommendations': eu_ai_results.get('recommendations', []),
            'agent_used': 'EUAIActAgent'
        }
        print(f"     ✅ EU AI Act score: {regulatory_results['eu_ai_act_alignment']['agent_score']:.2f}/5")
        
    except Exception as e:
        print(f"     ❌ EU AI Act evaluation failed: {e}")
        regulatory_results['eu_ai_act_alignment'] = {'agent_score': None, 'error': str(e)}
    
    try:
        # 3. Audit Trail Quality Assessment
        print("   📊 Evaluating audit trail quality...")
        audit_logs_path = "audit_logs"  # Default audit logs directory
        if os.path.exists(audit_logs_path):
            audit_agent = AuditTrailAgent(audit_log_path=audit_logs_path)
            audit_results = audit_agent.assess_audit_trail()
            regulatory_results['audit_trail_quality'] = {
                'agent_score': audit_results.get('overall_score', 0) / 20,  # Convert to 1-5 scale
                'completeness_score': audit_results.get('completeness', {}).get('score', 0),
                'quality_score': audit_results.get('quality', {}).get('score', 0),
                'detailed_assessment': audit_results,
                'agent_used': 'AuditTrailAgent'
            }
            print(f"     ✅ Audit Trail score: {regulatory_results['audit_trail_quality']['agent_score']:.2f}/5")
        else:
            print(f"     ⚠️ Audit logs directory not found: {audit_logs_path}")
            regulatory_results['audit_trail_quality'] = {
                'agent_score': 3.0,  # Default score when audit logs unavailable
                'note': 'Audit logs directory not available, using default score'
            }
            
    except Exception as e:
        print(f"     ❌ Audit Trail evaluation failed: {e}")
        regulatory_results['audit_trail_quality'] = {'agent_score': None, 'error': str(e)}
    
    print("🏛️ Regulatory agent evaluations complete!")
    return regulatory_results

def run_evaluation_pipeline():
    """
    Executes the complete evaluation pipeline from data loading to report generation.
    """
    print("--- Starting End-to-End Evaluation Test ---")

    # --- 1. Configuration ---
    input_excel_path = "sample_evaluation_data.xlsx"
    
    # Create timestamped subfolder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path("test_evaluation_results") / f"run_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {run_folder}")

    if not os.path.exists(input_excel_path):
        print(f"Error: Input file not found at '{input_excel_path}'")
        return

    # --- 2. Load Evaluation Data ---
    print(f"Loading evaluation data from: {input_excel_path}")
    try:
        evaluation_data = load_excel_with_snippets(input_excel_path)
        print(f"Successfully loaded {len(evaluation_data)} records for evaluation.")
    except Exception as e:
        print(f"Failed to load or process Excel file: {e}")
        return

    # --- 3. Initialize Evaluation System ---
    print("Initializing evaluation system...")
    try:
        # Use advanced evaluation configuration with all dimensions including regulatory
        config = AdvancedEvalConfig(profile=ConfigurationProfile.ACADEMIC_RESEARCH)
        print("Evaluation system initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize evaluation system: {e}")
        return

    # --- 4. Run Evaluation ---
    print("\n--- Running Evaluation on Sample Data ---")
    try:
        # Run standard evaluation
        results = evaluate_answers_with_snippets(evaluation_data, config)
        print(f"Successfully evaluated {len(results)} records with standard evaluation.")
        
        # Enhance results with specialized regulatory agent evaluations
        print("\n--- Enhancing with Specialized Regulatory Agents ---")
        enhanced_results = []
        
        for i, (eval_input, result) in enumerate(zip(evaluation_data, results)):
            print(f"\n🔍 Processing record {i+1}/{len(results)}:")
            print(f"   Query: {eval_input.query[:60]}...")
            
            # Get regulatory assessments from specialized agents
            regulatory_assessments = evaluate_regulatory_dimensions(
                eval_input.query, 
                eval_input.answer, 
                eval_input.source_snippets
            )
            
            # Enhance the result with agent-specific scores
            enhanced_result = result
            
            # Override regulatory dimension scores with agent results
            if hasattr(enhanced_result, 'scores'):
                if regulatory_assessments.get('gdpr_compliance', {}).get('agent_score') is not None:
                    enhanced_result.scores[EvaluationDimension.GDPR_COMPLIANCE] = regulatory_assessments['gdpr_compliance']['agent_score']
                    enhanced_result.gdpr_compliance = regulatory_assessments['gdpr_compliance']['agent_score']
                
                if regulatory_assessments.get('eu_ai_act_alignment', {}).get('agent_score') is not None:
                    enhanced_result.scores[EvaluationDimension.EU_AI_ACT_ALIGNMENT] = regulatory_assessments['eu_ai_act_alignment']['agent_score']
                    enhanced_result.eu_ai_act_alignment = regulatory_assessments['eu_ai_act_alignment']['agent_score']
                    
                if regulatory_assessments.get('audit_trail_quality', {}).get('agent_score') is not None:
                    enhanced_result.scores[EvaluationDimension.AUDIT_TRAIL_QUALITY] = regulatory_assessments['audit_trail_quality']['agent_score']
                    enhanced_result.audit_trail_quality = regulatory_assessments['audit_trail_quality']['agent_score']
            
            # Add regulatory assessment details to metadata
            if hasattr(enhanced_result, 'metadata'):
                enhanced_result.metadata.update({'regulatory_agents_assessment': regulatory_assessments})
            else:
                enhanced_result.metadata = {'regulatory_agents_assessment': regulatory_assessments}
            
            enhanced_results.append(enhanced_result)
            
        results = enhanced_results  # Use enhanced results for reporting
        print(f"\n✅ Enhanced evaluation complete with specialized regulatory agents!")
        
    except Exception as e:
        print(f"Failed to run evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not results:
        print("No results were generated. Aborting report generation.")
        return
        
    print("\n--- Evaluation Complete ---")

    # --- 5. Generate Report ---
    print("Generating evaluation reports...")
    try:
        # Generate academic report
        academic_report = generate_academic_citation_report(results, config=config)
        
        # Save academic report to file
        academic_report_path = run_folder / "academic_evaluation_report.txt"
        with open(academic_report_path, 'w', encoding='utf-8') as f:
            f.write(academic_report)
        print(f"Academic report saved to: {academic_report_path}")

        # Create detailed JSON results
        detailed_results = []
        for i, (eval_input, result) in enumerate(zip(evaluation_data, results)):
            result_dict = {
                "row_number": i + 1,
                "question": eval_input.query,
                "answer": eval_input.answer,
                "system_type": eval_input.system_type.value if eval_input.system_type else None,
                "source_snippets": [
                    {
                        "content": snippet.content,
                        "metadata": snippet.metadata
                    } for snippet in eval_input.source_snippets
                ],
                "evaluation_scores": {
                    "factual_accuracy": getattr(result, 'scores', {}).get(EvaluationDimension.FACTUAL_ACCURACY) if hasattr(result, 'scores') else None,
                    "relevance": getattr(result, 'scores', {}).get(EvaluationDimension.RELEVANCE) if hasattr(result, 'scores') else None,
                    "completeness": getattr(result, 'scores', {}).get(EvaluationDimension.COMPLETENESS) if hasattr(result, 'scores') else None,
                    "clarity": getattr(result, 'scores', {}).get(EvaluationDimension.CLARITY) if hasattr(result, 'scores') else None,
                    "citation_quality": getattr(result, 'scores', {}).get(EvaluationDimension.CITATION_QUALITY) if hasattr(result, 'scores') else None,
                    # Regulatory dimensions (enhanced by specialized agents)
                    "gdpr_compliance": getattr(result, 'scores', {}).get(EvaluationDimension.GDPR_COMPLIANCE) if hasattr(result, 'scores') else None,
                    "eu_ai_act_alignment": getattr(result, 'scores', {}).get(EvaluationDimension.EU_AI_ACT_ALIGNMENT) if hasattr(result, 'scores') else None,
                    "audit_trail_quality": getattr(result, 'scores', {}).get(EvaluationDimension.AUDIT_TRAIL_QUALITY) if hasattr(result, 'scores') else None,
                    # Additional dimensions
                    "reasoning_depth": getattr(result, 'scores', {}).get(EvaluationDimension.REASONING_DEPTH) if hasattr(result, 'scores') else None,
                    "adaptability": getattr(result, 'scores', {}).get(EvaluationDimension.ADAPTABILITY) if hasattr(result, 'scores') else None,
                    "efficiency": getattr(result, 'scores', {}).get(EvaluationDimension.EFFICIENCY) if hasattr(result, 'scores') else None,
                },
                "confidence_scores": {
                    key.value if hasattr(key, 'value') else str(key): value 
                    for key, value in getattr(result, 'confidence_scores', {}).items()
                },
                "justifications": {
                    key.value if hasattr(key, 'value') else str(key): value 
                    for key, value in getattr(result, 'justifications', {}).items()
                },
                "weighted_total": result.weighted_total if hasattr(result, 'weighted_total') else None,
                "raw_total": result.raw_total if hasattr(result, 'raw_total') else None,
                "snippet_grounding_score": result.snippet_grounding_score if hasattr(result, 'snippet_grounding_score') else None,
                # Enhanced: Specialized regulatory agent results
                "regulatory_agents_assessment": getattr(result, 'metadata', {}).get('regulatory_agents_assessment', {}),
                "evaluation_metadata": {
                    "timestamp": result.timestamp if hasattr(result, 'timestamp') else None,
                    "model_used": result.model_used if hasattr(result, 'model_used') else None,
                    "evaluation_duration": result.evaluation_duration if hasattr(result, 'evaluation_duration') else None,
                    "specialized_agents_used": ["GDPRComplianceAgent", "EUAIActAgent", "AuditTrailAgent"]
                }
            }
            detailed_results.append(result_dict)

        # Save detailed JSON results
        json_results_path = run_folder / "detailed_evaluation_results.json"
        with open(json_results_path, 'w', encoding='utf-8') as f:
            json.dump({
                "run_metadata": {
                    "timestamp": timestamp,
                    "input_file": input_excel_path,
                    "total_evaluations": len(results),
                    "evaluation_config": "AdvancedEvalConfig (academic_research)"
                },
                "results": detailed_results
            }, f, indent=2, ensure_ascii=False)
        print(f"Detailed JSON results saved to: {json_results_path}")

        # Save run summary
        summary_data = {
            "run_timestamp": timestamp,
            "input_file": input_excel_path,
            "total_records": len(evaluation_data),
            "successful_evaluations": len(results),
            "evaluation_approach": "Enhanced with Specialized Regulatory Agents",
            "specialized_agents": {
                "gdpr_compliance": "GDPRComplianceAgent",
                "eu_ai_act_alignment": "EUAIActAgent", 
                "audit_trail_quality": "AuditTrailAgent"
            },
            "average_scores": {},
            "files_generated": [
                str(academic_report_path.name),
                str(json_results_path.name)
            ]
        }

        # Calculate average scores if available
        if results:
            score_attributes = [
                'factual_accuracy', 'relevance', 'completeness', 'clarity', 'citation_quality',
                'gdpr_compliance', 'eu_ai_act_alignment', 'audit_trail_quality',
                'reasoning_depth', 'adaptability', 'efficiency', 'weighted_total'
            ]
            for attr in score_attributes:
                scores = [getattr(result, attr) for result in results if hasattr(result, attr) and getattr(result, attr) is not None]
                if scores:
                    summary_data["average_scores"][attr] = {
                        "mean": sum(scores) / len(scores),
                        "count": len(scores)
                    }

        summary_path = run_folder / "run_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Run summary saved to: {summary_path}")

        # Print a brief summary to console
        print("\n--- Evaluation Summary ---")
        if results:
            print(f"Total evaluations: {len(results)}")
            print(f"Results saved in folder: {run_folder}")
            print(f"🏛️ Regulatory Evaluation Method: Specialized Agents")
            print(f"   📋 GDPR Compliance: GDPRComplianceAgent")
            print(f"   🤖 EU AI Act Alignment: EUAIActAgent") 
            print(f"   📊 Audit Trail Quality: AuditTrailAgent")
            print("\n📊 Average Scores:")
            for attr, data in summary_data["average_scores"].items():
                print(f"   {attr}: {data['mean']:.2f} (n={data['count']})")
        else:
            print("No results to summarize.")

    except Exception as e:
        print(f"Failed to generate report: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- End-to-End Evaluation Test Finished ---")


if __name__ == "__main__":
    run_evaluation_pipeline()
