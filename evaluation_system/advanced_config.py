"""
Advanced Configuration Management for RAG vs Agentic AI Evaluation Framework.

This module provides sophisticated configuration management with:
- Environment-aware configurations
- Profile-based settings
- Validation and schema enforcement
- Dynamic configuration updates
- Academic preset configurations
"""

import os
import json
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

from .enums import EvaluationDimension
from .models import EvaluationRubric


logger = logging.getLogger(__name__)


class ConfigurationProfile(Enum):
    """Pre-defined configuration profiles for different use cases."""
    ACADEMIC_RESEARCH = "academic_research"
    PRODUCTION_EVALUATION = "production_evaluation"
    DEVELOPMENT_TESTING = "development_testing"
    COMPARATIVE_STUDY = "comparative_study"
    RAG_FOCUSED = "rag_focused"
    AGENTIC_FOCUSED = "agentic_focused"


class SemanticSimilarityMethod(Enum):
    """Available semantic similarity calculation methods."""
    TOKEN_OVERLAP = "token_overlap"
    COSINE_SIMILARITY = "cosine_similarity"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HYBRID_WEIGHTED = "hybrid_weighted"


@dataclass
class SemanticSimilarityConfig:
    """Configuration for semantic similarity calculations."""
    method: SemanticSimilarityMethod = SemanticSimilarityMethod.HYBRID_WEIGHTED
    model_name: str = "all-MiniLM-L6-v2"  # For sentence transformers
    similarity_threshold: float = 0.3
    weight_token_overlap: float = 0.3
    weight_semantic_similarity: float = 0.7
    enable_caching: bool = True
    cache_size: int = 1000


@dataclass 
class LLMProviderConfig:
    """Configuration for LLM providers."""
    provider: str = "azure_openai"  # azure_openai, openai, anthropic
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Azure-specific settings
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_version: str = "2024-02-15-preview"
    
    # Authentication
    api_key: Optional[str] = None
    use_managed_identity: bool = True


@dataclass
class EvaluationConfig:
    """Core evaluation configuration."""
    dimensions: List[EvaluationDimension] = field(default_factory=lambda: [
        EvaluationDimension.FACTUAL_ACCURACY,
        EvaluationDimension.RELEVANCE,
        EvaluationDimension.COMPLETENESS,
        EvaluationDimension.CLARITY,
        EvaluationDimension.CITATION_QUALITY,
        # Include regulatory dimensions by default
        EvaluationDimension.GDPR_COMPLIANCE,
        EvaluationDimension.EU_AI_ACT_ALIGNMENT,
        EvaluationDimension.AUDIT_TRAIL_QUALITY
    ])
    
    # Academic rigor settings
    confidence_threshold: float = 0.7
    inter_rater_agreement_threshold: float = 0.8
    minimum_sample_size: int = 10
    significance_level: float = 0.05
    
    # Evaluation behavior
    require_evidence_justification: bool = True
    enable_confidence_scoring: bool = True
    enable_snippet_analysis: bool = True
    
    # Customizable weights (will be loaded from rubrics)
    custom_weights: Optional[Dict[str, float]] = None


@dataclass
class OutputConfig:
    """Output and reporting configuration."""
    default_format: str = "json"
    include_academic_citations: bool = True
    include_statistical_analysis: bool = True
    include_confidence_intervals: bool = True
    generate_latex_tables: bool = False
    
    # File naming
    timestamp_format: str = "%Y%m%d_%H%M%S"
    output_prefix: str = "evaluation_results"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    detailed_logging: bool = False


@dataclass
class AdvancedEvalConfig:
    """Advanced evaluation configuration with all components."""
    
    # Core configuration
    profile: ConfigurationProfile = ConfigurationProfile.ACADEMIC_RESEARCH
    config_version: str = "2.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Component configurations
    llm_provider: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig) 
    semantic_similarity: SemanticSimilarityConfig = field(default_factory=SemanticSimilarityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Academic rubrics
    rubrics: Dict[EvaluationDimension, EvaluationRubric] = field(default_factory=dict)
    
    # Environment variables mapping
    env_var_mapping: Dict[str, str] = field(default_factory=lambda: {
        "AZURE_OPENAI_ENDPOINT": "llm_provider.azure_endpoint",
        "AZURE_OPENAI_DEPLOYMENT": "llm_provider.azure_deployment", 
        "AZURE_OPENAI_API_VERSION": "llm_provider.azure_api_version",
        "OPENAI_API_KEY": "llm_provider.api_key",
        "EVAL_MODEL": "llm_provider.model_name",
        "EVAL_TEMPERATURE": "llm_provider.temperature",
        "CONFIDENCE_THRESHOLD": "evaluation.confidence_threshold",
        "LOG_LEVEL": "output.log_level"
    })
    
    def __post_init__(self):
        """Initialize rubrics and apply environment variables."""
        if not self.rubrics:
            self.rubrics = self._get_default_rubrics()
        self._apply_environment_variables()
        self._validate_configuration()
        
        # Compatibility: expose dimensions at top level for existing evaluation code
        self.dimensions = self.evaluation.dimensions
        
        # Additional compatibility attributes
        self.azure_endpoint = self.llm_provider.azure_endpoint
        self.deployment_name = self.llm_provider.azure_deployment
        self.api_version = self.llm_provider.azure_api_version
        self.model_name = self.llm_provider.model_name
        self.temperature = self.llm_provider.temperature
        self.max_retries = self.llm_provider.max_retries
        self.retry_delay = self.llm_provider.retry_delay
        self.timeout = self.llm_provider.timeout
        self.confidence_threshold = self.evaluation.confidence_threshold
    
    def _get_default_rubrics(self) -> Dict[EvaluationDimension, EvaluationRubric]:
        """Get default academic rubrics with profile-specific weights."""
        base_rubrics = {
            EvaluationDimension.FACTUAL_ACCURACY: EvaluationRubric(
                dimension=EvaluationDimension.FACTUAL_ACCURACY,
                description="Accuracy of factual claims against ground truth or authoritative sources",
                criteria_5="All factual claims are accurate and verifiable (>95% accuracy)",
                criteria_4="Most factual claims accurate with minor inaccuracies (85-95% accuracy)", 
                criteria_3="Generally accurate with some notable errors (75-84% accuracy)",
                criteria_2="Multiple factual errors present (65-74% accuracy)",
                criteria_1="Significant factual inaccuracies or misinformation (<65% accuracy)",
                weight=0.25
            ),
            EvaluationDimension.RELEVANCE: EvaluationRubric(
                dimension=EvaluationDimension.RELEVANCE,
                description="Alignment between response content and query intent",
                criteria_5="Perfectly addresses query intent with no extraneous information",
                criteria_4="Directly addresses query with minimal off-topic content",
                criteria_3="Generally relevant with some tangential information",
                criteria_2="Partially relevant but includes significant off-topic content",
                criteria_1="Minimally relevant or addresses different question entirely",
                weight=0.2
            ),
            EvaluationDimension.COMPLETENESS: EvaluationRubric(
                dimension=EvaluationDimension.COMPLETENESS,
                description="Coverage of all necessary aspects to fully answer the query",
                criteria_5="Comprehensive coverage of all relevant aspects and sub-questions",
                criteria_4="Covers most important aspects with minor gaps",
                criteria_3="Adequate coverage of main points with some omissions",
                criteria_2="Partial coverage leaving important aspects unaddressed",
                criteria_1="Incomplete response missing critical information",
                weight=0.15
            ),
            EvaluationDimension.CLARITY: EvaluationRubric(
                dimension=EvaluationDimension.CLARITY,
                description="Readability, organization, and linguistic quality",
                criteria_5="Exceptionally clear, well-organized, and easily understood",
                criteria_4="Clear and well-structured with good flow",
                criteria_3="Generally clear with minor organizational issues",
                criteria_2="Somewhat unclear or poorly organized",
                criteria_1="Difficult to understand due to poor structure or language",
                weight=0.13
            ),
            EvaluationDimension.CITATION_QUALITY: EvaluationRubric(
                dimension=EvaluationDimension.CITATION_QUALITY,
                description="Quality and appropriateness of source citations and references",
                criteria_5="Excellent citations from authoritative sources, properly formatted",
                criteria_4="Good citations with minor formatting or relevance issues",
                criteria_3="Adequate citations with some quality concerns",
                criteria_2="Poor citation quality or inappropriate sources",
                criteria_1="Missing, invalid, or misleading citations",
                weight=0.07
            ),
            EvaluationDimension.GDPR_COMPLIANCE: EvaluationRubric(
                dimension=EvaluationDimension.GDPR_COMPLIANCE,
                description="Adherence to GDPR data protection principles and requirements",
                criteria_5="Full GDPR compliance with explicit consent, data minimization, and clear legal basis",
                criteria_4="Strong GDPR adherence with minor procedural gaps",
                criteria_3="Generally GDPR compliant with some technical requirements missing",
                criteria_2="Partial GDPR compliance with significant gaps in data protection",
                criteria_1="Poor GDPR compliance with potential legal violations",
                weight=0.12
            ),
            EvaluationDimension.EU_AI_ACT_ALIGNMENT: EvaluationRubric(
                dimension=EvaluationDimension.EU_AI_ACT_ALIGNMENT,
                description="Conformity with EU AI Act requirements for AI system transparency and risk management",
                criteria_5="Fully compliant with EU AI Act including risk assessment, transparency, and human oversight",
                criteria_4="Strong AI Act alignment with minor documentation gaps",
                criteria_3="Generally compliant with some risk management deficiencies",
                criteria_2="Partial AI Act compliance with significant regulatory gaps",
                criteria_1="Poor AI Act alignment with potential regulatory violations",
                weight=0.12
            ),
            EvaluationDimension.AUDIT_TRAIL_QUALITY: EvaluationRubric(
                dimension=EvaluationDimension.AUDIT_TRAIL_QUALITY,
                description="Completeness and quality of audit documentation and traceability",
                criteria_5="Comprehensive audit trail with complete decision traceability and metadata",
                criteria_4="Good audit documentation with minor logging gaps",
                criteria_3="Adequate audit trail with some missing decision points",
                criteria_2="Partial audit documentation with significant traceability gaps",
                criteria_1="Poor or missing audit trail compromising accountability",
                weight=0.08
            ),
        }
        
        # Apply profile-specific weight adjustments
        if self.profile == ConfigurationProfile.RAG_FOCUSED:
            base_rubrics[EvaluationDimension.CITATION_QUALITY].weight = 0.15
            base_rubrics[EvaluationDimension.FACTUAL_ACCURACY].weight = 0.3
        elif self.profile == ConfigurationProfile.AGENTIC_FOCUSED:
            base_rubrics[EvaluationDimension.COMPLETENESS].weight = 0.25
            base_rubrics[EvaluationDimension.RELEVANCE].weight = 0.25
        
        return base_rubrics
    
    def _apply_environment_variables(self):
        """Apply environment variables to configuration."""
        for env_var, config_path in self.env_var_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_attribute(config_path, value)
    
    def _set_nested_attribute(self, path: str, value: str):
        """Set nested attribute using dot notation."""
        parts = path.split('.')
        obj = self
        
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        # Type conversion based on existing attribute type
        existing_value = getattr(obj, parts[-1])
        if isinstance(existing_value, bool):
            value = value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(existing_value, int):
            value = int(value)
        elif isinstance(existing_value, float):
            value = float(value)
        
        setattr(obj, parts[-1], value)
        logger.debug(f"Set {path} = {value} from environment")
    
    def _validate_configuration(self):
        """Validate configuration consistency and constraints."""
        # Validate dimension weights sum
        total_weight = sum(rubric.weight for rubric in self.rubrics.values())
        if not (0.7 <= total_weight <= 1.0):
            logger.warning(f"Dimension weights sum to {total_weight:.3f}, should be ~0.8-1.0")

        # Validate regulatory dimension weights
        regulatory_dimensions = [
            EvaluationDimension.GDPR_COMPLIANCE,
            EvaluationDimension.EU_AI_ACT_ALIGNMENT,
            EvaluationDimension.AUDIT_TRAIL_QUALITY
        ]
        regulatory_weight = sum(self.rubrics[dim].weight for dim in regulatory_dimensions if dim in self.rubrics)
        
        if total_weight > 0:
            regulatory_percentage = (regulatory_weight / total_weight) * 100
            if not (20 <= regulatory_percentage <= 40):
                logger.warning(
                    f"Regulatory dimensions constitute {regulatory_percentage:.1f}% of total weight, "
                    f"which is outside the recommended 20-40% range."
                )
        
        # Validate thresholds
        if not (0.0 <= self.evaluation.confidence_threshold <= 1.0):
            raise ValueError(f"Confidence threshold must be 0-1, got {self.evaluation.confidence_threshold}")
        
        if not (0.0 <= self.evaluation.significance_level <= 0.1):
            raise ValueError(f"Significance level should be ≤0.1, got {self.evaluation.significance_level}")
        
        logger.info(f"Configuration validated successfully (profile: {self.profile.value})")
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Public validation method that returns validation status and errors.
        
        Returns:
            Tuple of (is_valid, error_list)
        """
        errors = []
        
        try:
            # Validate dimension weights sum
            total_weight = sum(rubric.weight for rubric in self.rubrics.values())
            if not (0.7 <= total_weight <= 1.0):
                errors.append(f"Dimension weights sum to {total_weight:.3f}, should be ~0.8-1.0")
            
            # Validate thresholds
            if not (0.0 <= self.evaluation.confidence_threshold <= 1.0):
                errors.append(f"Confidence threshold must be 0-1, got {self.evaluation.confidence_threshold}")
            
            if not (0.0 <= self.evaluation.significance_level <= 0.1):
                errors.append(f"Significance level should be ≤0.1, got {self.evaluation.significance_level}")
                
            # Validate temperature
            if not (0.0 <= self.llm_provider.temperature <= 2.0):
                errors.append(f"Temperature must be 0-2, got {self.llm_provider.temperature}")
            
            # Validate required rubrics
            if not self.rubrics:
                errors.append("No evaluation rubrics defined")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'AdvancedEvalConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct configuration with proper types
        config = cls()
        config._update_from_dict(data)
        return config
    
    def _update_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary data."""
        # This would need proper implementation for complex nested updates
        # For now, basic implementation
        logger.info("Loading configuration from dictionary")
    
    @classmethod
    def create_profile_config(cls, profile: ConfigurationProfile) -> 'AdvancedEvalConfig':
        """Create configuration for specific profile."""
        config = cls(profile=profile)
        
        # Profile-specific customizations
        if profile == ConfigurationProfile.DEVELOPMENT_TESTING:
            config.llm_provider.temperature = 0.0
            config.llm_provider.max_retries = 1
            config.evaluation.confidence_threshold = 0.5
            config.output.detailed_logging = True
            
        elif profile == ConfigurationProfile.PRODUCTION_EVALUATION:
            config.llm_provider.temperature = 0.05
            config.llm_provider.max_retries = 5
            config.evaluation.confidence_threshold = 0.8
            config.output.include_statistical_analysis = True
            
        elif profile == ConfigurationProfile.COMPARATIVE_STUDY:
            config.evaluation.minimum_sample_size = 20
            config.evaluation.significance_level = 0.01
            config.output.include_confidence_intervals = True
            config.output.generate_latex_tables = True
            
        return config


def create_default_config() -> AdvancedEvalConfig:
    """Create default configuration."""
    return AdvancedEvalConfig()


def get_config_for_profile(profile: ConfigurationProfile) -> AdvancedEvalConfig:
    """Get configuration optimized for specific use case."""
    return AdvancedEvalConfig.create_profile_config(profile)


def validate_environment_setup() -> Dict[str, bool]:
    """Validate that required environment variables are set."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT"
    ]
    
    optional_vars = [
        "AZURE_OPENAI_API_VERSION", 
        "OPENAI_API_KEY",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET"
    ]
    
    status = {}
    
    # Check required variables
    for var in required_vars:
        status[var] = os.getenv(var) is not None
    
    # Check optional variables
    for var in optional_vars:
        status[f"{var}_optional"] = os.getenv(var) is not None
    
    # Determine if setup is valid
    azure_oauth_setup = all(
        os.getenv(var) for var in ["AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"]
    )
    openai_key_setup = os.getenv("OPENAI_API_KEY") is not None
    
    status["valid_auth"] = azure_oauth_setup or openai_key_setup
    status["recommended_setup"] = all(status[var] for var in required_vars) and azure_oauth_setup
    
    return status
