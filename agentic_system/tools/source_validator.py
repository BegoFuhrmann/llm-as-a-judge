"""
Source Validator Tool - Verifies the credibility and relevance of information sources.
"""
from typing import List, Dict, Any
from urllib.parse import urlparse

# Handle imports
try:
    from ..core.base import AgentResponse
    from ..enums import LogLevel
    from ..audit.audit_log import audit_logger
    from .web_search import SearchResult
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir.parent))
    
    from agentic_system.core.base import AgentResponse
    from agentic_system.enums import LogLevel
    from agentic_system.audit.audit_log import audit_logger
    from agentic_system.tools.web_search import SearchResult


class SourceValidator:
    """
    A tool to validate the credibility and relevance of sources.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.trusted_domains = self.config.get("trusted_domains", [
            "wikipedia.org", "reuters.com", "apnews.com", "bbc.com", "github.com"
        ])

    async def validate(self, search_results: List[SearchResult]) -> AgentResponse:
        """
        Validates a list of search results based on predefined criteria.
        """
        await audit_logger.log_system_event(
            event_type="validation_started",
            message="Starting source validation.",
            log_level=LogLevel.INFO
        )
        
        validated_sources = []
        for res in search_results:
            domain = urlparse(res.url).netloc
            is_trusted = any(trusted in domain for trusted in self.trusted_domains)
            
            if is_trusted and res.relevance_score > 0.5:
                validated_sources.append(res)
        
        await audit_logger.log_system_event(
            event_type="validation_completed",
            message=f"Validation completed. Found {len(validated_sources)} trusted sources.",
            log_level=LogLevel.INFO
        )
        
        return AgentResponse(
            success=True,
            data={"validated_sources": validated_sources}
        )
