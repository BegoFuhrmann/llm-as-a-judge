#!/usr/bin/env python3
"""
Configuration Management CLI for RAG vs Agentic AI Evaluation Framework.

This tool helps manage configurations, validate environment setup,
and provide guidance for optimal evaluation settings.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add package to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rag_agentic_evaluation.advanced_config import (
    AdvancedEvalConfig,
    ConfigurationProfile, 
    SemanticSimilarityMethod,
    validate_environment_setup
)


def create_config_command(args):
    """Create a new configuration file."""
    print(f"Creating configuration for profile: {args.profile}")
    
    # Get profile enum
    try:
        profile = ConfigurationProfile(args.profile)
    except ValueError:
        print(f"Error: Invalid profile '{args.profile}'")
        print(f"Available profiles: {[p.value for p in ConfigurationProfile]}")
        return 1
    
    # Create configuration
    config = AdvancedEvalConfig.create_profile_config(profile)
    
    # Customize based on CLI arguments
    if args.model:
        config.llm_provider.model_name = args.model
    if args.temperature is not None:
        config.llm_provider.temperature = args.temperature
    if args.similarity_method:
        try:
            method = SemanticSimilarityMethod(args.similarity_method)
            config.semantic_similarity.method = method
        except ValueError:
            print(f"Warning: Invalid similarity method '{args.similarity_method}', using default")
    
    # Save configuration
    output_path = Path(args.output) if args.output else Path(f"config_{profile.value}.json")
    config.save_to_file(output_path)
    
    print(f"Configuration saved to: {output_path}")
    print(f"Model: {config.llm_provider.model_name}")
    print(f"Temperature: {config.llm_provider.temperature}")
    print(f"Similarity method: {config.semantic_similarity.method.value}")
    print(f"Evaluation dimensions: {len(config.evaluation.dimensions)}")
    
    return 0


def validate_config_command(args):
    """Validate an existing configuration file."""
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1
    
    try:
        config = AdvancedEvalConfig.load_from_file(config_path)
        print(f"✅ Configuration file is valid: {config_path}")
        print(f"Profile: {config.profile.value}")
        print(f"Configuration version: {config.config_version}")
        
        # Validate environment compatibility
        env_status = validate_environment_setup()
        print("\n🔧 Environment Status:")
        
        for var, status in env_status.items():
            if var.endswith('_optional'):
                continue
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {var}: {'Set' if status else 'Not set'}")
        
        if env_status.get('valid_auth', False):
            print("\n✅ Authentication setup is valid")
        else:
            print("\n❌ Authentication setup needs attention")
            print("   Please set either:")
            print("   - AZURE_CLIENT_ID and AZURE_CLIENT_SECRET (recommended)")
            print("   - OPENAI_API_KEY (alternative)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1


def show_profiles_command(args):
    """Show available configuration profiles."""
    print("📋 Available Configuration Profiles:\n")
    
    profiles_info = {
        ConfigurationProfile.ACADEMIC_RESEARCH: {
            "description": "Rigorous academic evaluation with high confidence thresholds",
            "use_case": "Research papers, thesis work, academic studies"
        },
        ConfigurationProfile.PRODUCTION_EVALUATION: {
            "description": "Production-ready evaluation with robust error handling",
            "use_case": "Production AI system monitoring and evaluation"
        },
        ConfigurationProfile.DEVELOPMENT_TESTING: {
            "description": "Fast iteration with detailed logging for development",
            "use_case": "AI system development and debugging"
        },
        ConfigurationProfile.COMPARATIVE_STUDY: {
            "description": "Statistical rigor optimized for A/B testing",
            "use_case": "Comparing multiple AI systems or approaches"
        },
        ConfigurationProfile.RAG_FOCUSED: {
            "description": "Enhanced weights for RAG system evaluation",
            "use_case": "Evaluating retrieval-augmented generation systems"
        },
        ConfigurationProfile.AGENTIC_FOCUSED: {
            "description": "Optimized for agentic AI system capabilities",
            "use_case": "Evaluating autonomous AI agents and reasoning systems"
        }
    }
    
    for profile, info in profiles_info.items():
        print(f"🏷️  {profile.value}")
        print(f"   Description: {info['description']}")
        print(f"   Use case: {info['use_case']}")
        print()
    
    return 0


def show_similarity_methods_command(args):
    """Show available semantic similarity methods."""
    print("🔍 Available Semantic Similarity Methods:\n")
    
    methods_info = {
        SemanticSimilarityMethod.TOKEN_OVERLAP: {
            "description": "Enhanced token overlap with precision/recall metrics",
            "performance": "Fast",
            "accuracy": "Good for lexical similarity"
        },
        SemanticSimilarityMethod.COSINE_SIMILARITY: {
            "description": "TF-IDF vectorization with cosine similarity",
            "performance": "Medium",
            "accuracy": "Better for semantic similarity"
        },
        SemanticSimilarityMethod.SENTENCE_TRANSFORMERS: {
            "description": "Deep learning embeddings for semantic understanding",
            "performance": "Slower",
            "accuracy": "Best for semantic similarity"
        },
        SemanticSimilarityMethod.HYBRID_WEIGHTED: {
            "description": "Combines multiple methods with configurable weights",
            "performance": "Medium",
            "accuracy": "Balanced approach (recommended)"
        }
    }
    
    for method, info in methods_info.items():
        print(f"🎯 {method.value}")
        print(f"   Description: {info['description']}")
        print(f"   Performance: {info['performance']}")
        print(f"   Accuracy: {info['accuracy']}")
        print()
    
    print("💡 Recommendation: Use 'hybrid_weighted' for best balance of accuracy and performance")
    return 0


def environment_check_command(args):
    """Check environment setup and provide recommendations."""
    print("🔍 Environment Setup Check\n")
    
    env_status = validate_environment_setup()
    
    print("Required Variables:")
    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    for var in required:
        status = env_status.get(var, False)
        icon = "✅" if status else "❌"
        value = os.getenv(var, "Not set")
        print(f"  {icon} {var}: {value if status else 'Not set'}")
    
    print("\nAuthentication:")
    auth_vars = ["AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "OPENAI_API_KEY"]
    for var in auth_vars:
        status = env_status.get(f"{var}_optional", False)
        icon = "✅" if status else "❌"
        value = "***" if status else "Not set"
        print(f"  {icon} {var}: {value}")
    
    print("\nRecommendations:")
    if not env_status.get('valid_auth', False):
        print("  🔧 Set up authentication:")
        print("     Option 1 (Recommended): AZURE_CLIENT_ID + AZURE_CLIENT_SECRET")
        print("     Option 2 (Alternative): OPENAI_API_KEY")
    
    if not all(env_status.get(var, False) for var in required):
        print("  🔧 Set required Azure OpenAI variables in .env file")
        print("     Example .env file:")
        print("     AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        print("     AZURE_OPENAI_DEPLOYMENT=your-deployment-name")
    
    if env_status.get('recommended_setup', False):
        print("  ✅ Your setup follows all recommendations!")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Configuration Management for RAG vs Agentic AI Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create academic research configuration
  python eval_config_manager.py create --profile academic_research --output my_config.json
  
  # Validate existing configuration
  python eval_config_manager.py validate --config my_config.json
  
  # Check environment setup
  python eval_config_manager.py env-check
  
  # Show available profiles
  python eval_config_manager.py profiles
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create config command
    create_parser = subparsers.add_parser('create', help='Create new configuration')
    create_parser.add_argument(
        '--profile', 
        required=True,
        choices=[p.value for p in ConfigurationProfile],
        help='Configuration profile to use'
    )
    create_parser.add_argument('--output', help='Output file path')
    create_parser.add_argument('--model', help='LLM model name')
    create_parser.add_argument('--temperature', type=float, help='LLM temperature')
    create_parser.add_argument(
        '--similarity-method',
        choices=[m.value for m in SemanticSimilarityMethod],
        help='Semantic similarity method'
    )
    
    # Validate config command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('--config', required=True, help='Configuration file to validate')
    
    # Show profiles command
    subparsers.add_parser('profiles', help='Show available configuration profiles')
    
    # Show similarity methods command
    subparsers.add_parser('similarity-methods', help='Show available similarity methods')
    
    # Environment check command
    subparsers.add_parser('env-check', help='Check environment setup')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Dispatch to appropriate command
    if args.command == 'create':
        return create_config_command(args)
    elif args.command == 'validate':
        return validate_config_command(args)
    elif args.command == 'profiles':
        return show_profiles_command(args)
    elif args.command == 'similarity-methods':
        return show_similarity_methods_command(args)
    elif args.command == 'env-check':
        return environment_check_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
