"""
Configuration management for the Agentic System.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI services."""
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_version: str = "2024-06-01"
    
    # Model configurations
    chat_model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-large"
    
    # Request parameters
    temperature: float = 0.1
    max_tokens: int = 4000
    top_p: float = 0.95
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable must be set")
        if not self.endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable must be set")


@dataclass
class DocumentProcessingConfig:
    """Configuration for document processing capabilities."""
    supported_formats: List[str] = field(default_factory=lambda: [".pdf", ".docx", ".txt"])
    chunk_size: int = 1000
    chunk_overlap: int = 200
    processed_docs_dir: str = "processed_documents"
    embeddings_dir: str = "embeddings"
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        Path(self.processed_docs_dir).mkdir(exist_ok=True)
        Path(self.embeddings_dir).mkdir(exist_ok=True)


@dataclass
class VectorStoreConfig:
    """Configuration for Chroma vector store."""
    collection_name: str = "financial_documents"
    persist_directory: str = "chroma_db"
    distance_metric: str = "cosine"
    top_k: int = 5
    score_threshold: float = 0.7
    
    def __post_init__(self):
        """Create persist directory if it doesn't exist."""
        Path(self.persist_directory).mkdir(exist_ok=True)


@dataclass
class AgentCoordinationConfig:
    """Configuration for agent coordination and orchestration."""
    agent_timeout: int = 30
    coordination_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    confidence_threshold: float = 0.8
    consensus_required: bool = True
    audit_enabled: bool = True
    audit_log_path: str = "audit_trail.jsonl"


@dataclass
class AgenticConfig:
    """Main configuration class for the Agentic System."""
    azure_openai: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    agent_coordination: AgentCoordinationConfig = field(default_factory=AgentCoordinationConfig)
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    def __post_init__(self):
        """Configure logging based on settings."""
        if self.debug_mode:
            self.log_level = "DEBUG"
        logging.getLogger().setLevel(getattr(logging, self.log_level.upper()))
    
    @classmethod
    def from_env(cls) -> "AgenticConfig":
        """Create configuration from environment variables."""
        return cls()


# Global configuration instance
config = AgenticConfig.from_env()
