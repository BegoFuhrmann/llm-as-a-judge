"""
Configuration Management for Enhanced Agentic System
==================================================

Comprehensive configuration system supporting:
- Azure OpenAI integration with managed identity
- Regulatory compliance settings for financial services
- Academic evaluation compatibility
- Multi-modal document processing parameters
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field
from enum import Enum

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Enumeration of agent roles in the system."""
    DOCUMENT_COLLECTION = "document_collection"
    INFORMATION_EXTRACTION = "information_extraction" 
    DECISION_COORDINATION = "decision_coordination"


class ComplianceFramework(Enum):
    """Regulatory compliance frameworks for financial services."""
    GDPR = "gdpr"
    EU_AI_ACT = "eu_ai_act"
    BASEL_III = "basel_iii"
    SOX = "sarbanes_oxley"
    MIFID_II = "mifid_ii"


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration with managed identity support."""
    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    deployment_name: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"))
    api_version: str = "2024-02-15-preview"
    use_managed_identity: bool = True
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60
    max_retries: int = 3
    
    def get_token_provider(self):
        """Return Azure AD token provider using service principal from environment."""
        import os
        from azure.identity import ClientSecretCredential, get_bearer_token_provider
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        if not all([tenant_id, client_id, client_secret]):
            raise ValueError("Missing SP credentials: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
        creds = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        return get_bearer_token_provider(
            creds,
            "https://cognitiveservices.azure.com/.default"
        )


@dataclass
class DocumentProcessingConfig:
    """Configuration for multi-modal document processing."""
    supported_formats: List[str] = field(default_factory=lambda: [".pdf", ".docx", ".doc", ".txt", ".xlsx"])
    max_file_size_mb: int = 50
    extract_tables: bool = True
    extract_images: bool = True
    ocr_enabled: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    pdf_parser: str = "pdfplumber"


class AgenticConfig(BaseSettings):
    """Main configuration class for the enhanced agentic system."""
    
    # System identification
    system_name: str = "Enhanced Agentic System"
    version: str = "2.0.0"
    environment: str = Field(default="development", env="AGENTIC_ENV")
    
    # Core configurations
    azure_openai: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = "ignore"  # Allow extra fields to be ignored


def load_config(config_type: str = "development") -> AgenticConfig:
    """Load configuration based on environment type."""
    return AgenticConfig()
