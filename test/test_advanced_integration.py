"""
Integration tests for advanced configuration and semantic similarity features.

This test module validates the integration between the new advanced configuration
system and enhanced semantic similarity calculations with the existing framework.
"""

import pytest
import json
from pathlib import Path
from typing import List, Dict, Any

from rag_agentic_evaluation.advanced_config import (
    AdvancedEvalConfig,
    ConfigurationProfile,
    SemanticSimilarityMethod
)
from rag_agentic_evaluation.semantic_similarity import SemanticSimilarityCalculator
from rag_agentic_evaluation.models import SourceSnippet
from rag_agentic_evaluation.utils import calculate_snippet_grounding_score


class TestAdvancedConfigurationIntegration:
    """Test integration of advanced configuration system."""
    
    def test_config_profile_creation(self):
        """Test creating configurations for different profiles."""
        for profile in ConfigurationProfile:
            config = AdvancedEvalConfig.create_profile_config(profile)
            
            # Basic validation
            assert config.profile == profile
            assert config.config_version == "2.0"
            assert len(config.evaluation.dimensions) == 5
            
            # Profile-specific validation
            if profile == ConfigurationProfile.ACADEMIC_RESEARCH:
                assert config.llm_provider.temperature <= 0.3  # Low temperature for consistency
                assert config.evaluation.confidence_threshold >= 0.8  # High confidence
            elif profile == ConfigurationProfile.DEVELOPMENT_TESTING:
                assert config.logging.log_level == "DEBUG"
                assert config.evaluation.verbose_feedback is True
    
    def test_config_serialization(self, tmp_path):
        """Test configuration save/load functionality."""
        config = AdvancedEvalConfig.create_profile_config(ConfigurationProfile.ACADEMIC_RESEARCH)
        
        # Customize configuration
        config.llm_provider.model_name = "test-model"
        config.semantic_similarity.method = SemanticSimilarityMethod.HYBRID_WEIGHTED
        
        # Save to file
        config_file = tmp_path / "test_config.json"
        config.save_to_file(config_file)
        
        # Load from file
        loaded_config = AdvancedEvalConfig.load_from_file(config_file)
        
        # Verify loaded configuration
        assert loaded_config.profile == ConfigurationProfile.ACADEMIC_RESEARCH
        assert loaded_config.llm_provider.model_name == "test-model"
        assert loaded_config.semantic_similarity.method == SemanticSimilarityMethod.HYBRID_WEIGHTED
    
    def test_environment_variable_mapping(self, monkeypatch):
        """Test environment variable mapping functionality."""
        # Set test environment variables
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test-deployment")
        monkeypatch.setenv("AZURE_CLIENT_ID", "test-client-id")
        
        config = AdvancedEvalConfig.create_profile_config(ConfigurationProfile.PRODUCTION_EVALUATION)
        config.apply_environment_variables()
        
        assert config.llm_provider.azure_endpoint == "https://test.openai.azure.com/"
        assert config.llm_provider.deployment_name == "test-deployment"
    
    def test_config_validation(self, tmp_path):
        """Test configuration validation functionality."""
        # Create valid configuration
        config = AdvancedEvalConfig.create_profile_config(ConfigurationProfile.ACADEMIC_RESEARCH)
        
        # Test validation passes
        is_valid, errors = config.validate()
        assert is_valid
        assert len(errors) == 0
        
        # Test invalid configuration
        config.llm_provider.temperature = 2.5  # Invalid temperature
        config.evaluation.dimensions = {}  # Empty dimensions
        
        is_valid, errors = config.validate()
        assert not is_valid
        assert len(errors) > 0
        assert any("temperature" in error.lower() for error in errors)


class TestSemanticSimilarityIntegration:
    """Test integration of enhanced semantic similarity calculations."""
    
    def setup_method(self):
        """Set up test data for semantic similarity tests."""
        self.test_answer = """
        Machine learning is a subset of artificial intelligence that enables computers 
        to learn and improve from experience without being explicitly programmed. 
        It uses algorithms to analyze data, identify patterns, and make predictions.
        """
        
        self.test_snippets = [
            SourceSnippet(
                snippet_id="snippet_1",
                content="Machine learning is a subset of AI that allows systems to learn from data.",
                source_title="Introduction to ML",
                relevance_score=0.9
            ),
            SourceSnippet(
                snippet_id="snippet_2", 
                content="Deep learning uses neural networks with multiple layers to process data.",
                source_title="Deep Learning Basics",
                relevance_score=0.7
            ),
            SourceSnippet(
                snippet_id="snippet_3",
                content="Algorithms analyze data patterns to make predictions and classifications.",
                source_title="Algorithm Design",
                relevance_score=0.8
            )
        ]
    
    def test_semantic_similarity_calculator_initialization(self):
        """Test initialization of semantic similarity calculator."""
        for method in SemanticSimilarityMethod:
            calculator = SemanticSimilarityCalculator(method)
            assert calculator.method == method
            
            # Test method-specific initialization
            if method == SemanticSimilarityMethod.SENTENCE_TRANSFORMERS:
                assert calculator.sentence_model is not None
            elif method == SemanticSimilarityMethod.COSINE_SIMILARITY:
                assert calculator.vectorizer is not None
    
    def test_similarity_calculation_methods(self):
        """Test different similarity calculation methods."""
        text1 = "Machine learning enables computers to learn from data"
        text2 = "ML allows systems to learn without explicit programming"
        
        for method in SemanticSimilarityMethod:
            calculator = SemanticSimilarityCalculator(method)
            similarity = calculator.calculate_similarity(text1, text2)
            
            # Basic validation
            assert 0 <= similarity <= 1, f"Similarity score out of range for {method.value}"
            assert isinstance(similarity, float), f"Similarity score not float for {method.value}"
            
            # Method-specific validation
            if method == SemanticSimilarityMethod.SENTENCE_TRANSFORMERS:
                # Should detect high semantic similarity
                assert similarity > 0.6, f"Sentence transformers should detect high similarity"
    
    def test_hybrid_weighted_similarity(self):
        """Test hybrid weighted similarity calculation."""
        calculator = SemanticSimilarityCalculator(SemanticSimilarityMethod.HYBRID_WEIGHTED)
        
        # Test with high similarity texts
        high_sim_text1 = "Machine learning algorithms learn from data"
        high_sim_text2 = "ML algorithms use data to learn patterns"
        high_similarity = calculator.calculate_similarity(high_sim_text1, high_sim_text2)
        
        # Test with low similarity texts
        low_sim_text1 = "Machine learning algorithms learn from data"
        low_sim_text2 = "The weather is sunny today with clear skies"
        low_similarity = calculator.calculate_similarity(low_sim_text1, low_sim_text2)
        
        assert high_similarity > low_similarity, "Hybrid method should distinguish similarity levels"
        assert high_similarity > 0.5, "High similarity should be detected"
        assert low_similarity < 0.3, "Low similarity should be detected"
    
    def test_grounding_score_integration(self):
        """Test integration of enhanced grounding score calculation."""
        # Test with basic token overlap (fallback)
        basic_score = calculate_snippet_grounding_score(
            self.test_answer, 
            self.test_snippets,
            use_enhanced_similarity=False
        )
        
        # Test with enhanced semantic similarity
        enhanced_score = calculate_snippet_grounding_score(
            self.test_answer, 
            self.test_snippets,
            use_enhanced_similarity=True
        )
        
        # Both should be valid scores
        assert 0 <= basic_score <= 1
        assert 0 <= enhanced_score <= 1
        
        # Enhanced method should generally provide more nuanced scores
        # (This is a heuristic test - exact values depend on content)
        assert isinstance(enhanced_score, float)
    
    def test_performance_comparison(self):
        """Test performance characteristics of different similarity methods."""
        import time
        
        text1 = self.test_answer
        text2 = self.test_snippets[0].content
        
        performance_results = {}
        
        for method in SemanticSimilarityMethod:
            calculator = SemanticSimilarityCalculator(method)
            
            # Warm-up calculation
            calculator.calculate_similarity(text1, text2)
            
            # Measure performance
            start_time = time.time()
            for _ in range(10):  # Multiple runs for better measurement
                calculator.calculate_similarity(text1, text2)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            performance_results[method.value] = avg_time
        
        # Validate performance expectations
        assert performance_results[SemanticSimilarityMethod.TOKEN_OVERLAP.value] < 0.1
        # Sentence transformers may be slower but should still be reasonable
        assert performance_results[SemanticSimilarityMethod.SENTENCE_TRANSFORMERS.value] < 5.0


class TestConfigurationAndSimilarityIntegration:
    """Test integration between advanced configuration and semantic similarity."""
    
    def test_config_controlled_similarity_method(self):
        """Test that configuration properly controls similarity method selection."""
        for method in SemanticSimilarityMethod:
            config = AdvancedEvalConfig.create_profile_config(ConfigurationProfile.ACADEMIC_RESEARCH)
            config.semantic_similarity.method = method
            
            # Create calculator based on config
            calculator = SemanticSimilarityCalculator(config.semantic_similarity.method)
            
            assert calculator.method == method
    
    def test_profile_optimized_similarity_settings(self):
        """Test that different profiles have appropriate similarity settings."""
        # Academic research should prioritize accuracy
        academic_config = AdvancedEvalConfig.create_profile_config(ConfigurationProfile.ACADEMIC_RESEARCH)
        assert academic_config.semantic_similarity.method in [
            SemanticSimilarityMethod.SENTENCE_TRANSFORMERS,
            SemanticSimilarityMethod.HYBRID_WEIGHTED
        ]
        
        # Development testing should balance speed and accuracy
        dev_config = AdvancedEvalConfig.create_profile_config(ConfigurationProfile.DEVELOPMENT_TESTING)
        assert dev_config.semantic_similarity.method in [
            SemanticSimilarityMethod.HYBRID_WEIGHTED,
            SemanticSimilarityMethod.COSINE_SIMILARITY
        ]
    
    def test_end_to_end_integration(self, tmp_path):
        """Test complete end-to-end integration of advanced features."""
        # Create advanced configuration
        config = AdvancedEvalConfig.create_profile_config(ConfigurationProfile.ACADEMIC_RESEARCH)
        config.semantic_similarity.method = SemanticSimilarityMethod.HYBRID_WEIGHTED
        
        # Save configuration
        config_file = tmp_path / "integration_test_config.json"
        config.save_to_file(config_file)
        
        # Load configuration
        loaded_config = AdvancedEvalConfig.load_from_file(config_file)
        
        # Use configuration for similarity calculation
        calculator = SemanticSimilarityCalculator(loaded_config.semantic_similarity.method)
        
        # Perform similarity calculation
        similarity = calculator.calculate_similarity(
            "Machine learning enables automated pattern recognition",
            "ML systems can automatically detect patterns in data"
        )
        
        # Validate integration works
        assert 0 <= similarity <= 1
        assert similarity > 0.4  # Should detect semantic similarity


if __name__ == "__main__":
    """Run integration tests directly."""
    import sys
    import os
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Run tests
    pytest.main([__file__, "-v"])
