"""
Web Search Tool - Provides external knowledge acquisition through DuckDuckGo search.
"""
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import re
from dataclasses import dataclass, field

# Web search and scraping
import aiohttp
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# Handle imports for both module and direct execution
try:
    # Try relative imports first (when used as a module)
    from ..core.base import BaseAgent, Task, AgentResponse
    from ..enums import AgentType, LogLevel, Priority, SearchScope
    from ..audit.audit_log import audit_logger
except ImportError:
    # Fallback to absolute imports (when run directly)
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    
    from core.base import BaseAgent, Task, AgentResponse
    from enums import AgentType, LogLevel, Priority, SearchScope
    from audit.audit_log import audit_logger


@dataclass
class SearchResult:
    """Represents a single search result with metadata."""
    title: str = ""
    url: str = ""
    snippet: str = ""
    content: str = ""
    relevance_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    source_type: str = "web"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Represents a search query with configuration."""
    query: str = ""
    max_results: int = 5
    search_scope: SearchScope = SearchScope.WEB_SEARCH
    filters: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    region: str = "us-en"
    safe_search: str = "moderate"


class WebSearchTool:
    """
    Advanced web search tool using DuckDuckGo for external knowledge acquisition.
    
    Features:
    - Async search capabilities
    - Content extraction and cleaning
    - Rate limiting and error handling
    - Result ranking and filtering
    - Audit logging integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Search configuration
        self.max_results_per_query = self.config.get('max_results_per_query', 10)
        self.max_content_length = self.config.get('max_content_length', 5000)
        self.timeout_seconds = self.config.get('timeout_seconds', 30)
        self.rate_limit_delay = self.config.get('rate_limit_delay', 1.0)
        
        # Content extraction settings
        self.extract_full_content = self.config.get('extract_full_content', True)
        self.min_content_length = self.config.get('min_content_length', 100)
        self.content_quality_threshold = self.config.get('content_quality_threshold', 0.5)
        
        # Search client
        self.ddgs = DDGS()
        
        # Rate limiting
        self._last_search_time = 0
        self._search_count = 0
        
        # User agent for web scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    async def search(self, query: Union[str, SearchQuery]) -> List[SearchResult]:
        """
        Perform web search with comprehensive result processing.
        
        Args:
            query: Search query string or SearchQuery object
            
        Returns:
            List of SearchResult objects with extracted content
        """
        # Convert string query to SearchQuery object
        if isinstance(query, str):
            search_query = SearchQuery(query=query)
        else:
            search_query = query
            
        # Audit log the search initiation
        await audit_logger.log_agent_action(
            agent_id="web_search_tool",
            agent_type=AgentType.AUTONOMOUS,
            action="search_initiated",
            log_level=LogLevel.INFO,
            query=search_query.query,
            max_results=search_query.max_results,
            search_scope=search_query.search_scope.value
        )
        
        try:
            # Apply rate limiting
            await self._apply_rate_limit()
            
            # Perform the search
            raw_results = await self._perform_search(search_query)
            
            # Process and extract content
            processed_results = await self._process_search_results(raw_results, search_query)
            
            # Rank and filter results
            final_results = self._rank_and_filter_results(processed_results, search_query)
            
            # Audit log successful search completion
            await audit_logger.log_agent_action(
                agent_id="web_search_tool",
                agent_type=AgentType.AUTONOMOUS,
                action="search_completed",
                log_level=LogLevel.INFO,
                query=search_query.query,
                results_found=len(final_results),
                processing_time=time.time() - self._last_search_time
            )
            
            return final_results
            
        except Exception as e:
            # Audit log the error
            await audit_logger.log_agent_action(
                agent_id="web_search_tool",
                agent_type=AgentType.AUTONOMOUS,
                action="search_error",
                log_level=LogLevel.ERROR,
                query=search_query.query,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    async def _perform_search(self, search_query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform the actual DuckDuckGo search."""
        try:
            # Use DuckDuckGo search with list output format
            search_results = self.ddgs.text(
                keywords=search_query.query,
                region=search_query.region,
                safesearch=search_query.safe_search,
                max_results=min(search_query.max_results, self.max_results_per_query)
            )
            
            # Convert generator to list
            return list(search_results)
            
        except Exception as e:
            # Audit log DuckDuckGo search error
            await audit_logger.log_agent_action(
                agent_id="web_search_tool",
                agent_type=AgentType.AUTONOMOUS,
                action="ddg_search_error",
                log_level=LogLevel.ERROR,
                query=search_query.query,
                error=str(e)
            )
            return []
    
    async def _process_search_results(self, raw_results: List[Dict[str, Any]], 
                                     search_query: SearchQuery) -> List[SearchResult]:
        """Process raw search results and extract content."""
        processed_results = []
        
        for result in raw_results:
            try:
                # Create base SearchResult
                search_result = SearchResult(
                    title=result.get('title', ''),
                    url=result.get('href', ''),
                    snippet=result.get('body', ''),
                    metadata={
                        'raw_result': result,
                        'search_query': search_query.query
                    }
                )
                
                # Extract full content if enabled
                if self.extract_full_content and search_result.url:
                    content = await self._extract_webpage_content(search_result.url)
                    search_result.content = content
                    
                    # Calculate relevance score
                    search_result.relevance_score = self._calculate_relevance_score(
                        search_result, search_query.query
                    )
                
                # Only include results that meet quality threshold
                if (len(search_result.content) >= self.min_content_length and
                    search_result.relevance_score >= self.content_quality_threshold):
                    processed_results.append(search_result)
                    
            except Exception as e:
                # Audit log result processing warning
                await audit_logger.log_agent_action(
                    agent_id="web_search_tool",
                    agent_type=AgentType.AUTONOMOUS,
                    action="result_processing_error",
                    log_level=LogLevel.WARNING,
                    url=result.get('href', 'unknown'),
                    error=str(e)
                )
                continue
        
        return processed_results
    
    async def _extract_webpage_content(self, url: str) -> str:
        """Extract clean text content from a webpage."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        return self._clean_html_content(html_content)
                    else:
                        return ""
        except Exception as e:
            # Audit log content extraction warning
            await audit_logger.log_agent_action(
                agent_id="web_search_tool",
                agent_type=AgentType.AUTONOMOUS,
                action="content_extraction_error",
                log_level=LogLevel.WARNING,
                url=url,
                error=str(e)
            )
            return ""
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean and extract meaningful text from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length] + "..."
            
            return text
            
        except Exception as e:
            # Audit log HTML cleaning warning (fire-and-forget)
            import asyncio
            asyncio.create_task(
                audit_logger.log_agent_action(
                    agent_id="web_search_tool",
                    agent_type=AgentType.AUTONOMOUS,
                    action="html_cleaning_error",
                    log_level=LogLevel.WARNING,
                    error=str(e)
                )
            )
            return ""
    
    def _calculate_relevance_score(self, result: SearchResult, query: str) -> float:
        """Calculate relevance score for a search result."""
        try:
            query_terms = set(query.lower().split())
            
            # Check title relevance
            title_score = len([term for term in query_terms 
                             if term in result.title.lower()]) / len(query_terms)
            
            # Check content relevance
            content_text = f"{result.snippet} {result.content}".lower()
            content_score = len([term for term in query_terms 
                               if term in content_text]) / len(query_terms)
            
            # Weighted average (title weighted more heavily)
            final_score = (title_score * 0.4) + (content_score * 0.6)
            
            return min(final_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _rank_and_filter_results(self, results: List[SearchResult], 
                                search_query: SearchQuery) -> List[SearchResult]:
        """Rank and filter search results by relevance."""
        # Sort by relevance score
        ranked_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        
        # Apply max results limit
        return ranked_results[:search_query.max_results]
    
    async def _apply_rate_limit(self):
        """Apply rate limiting between searches."""
        current_time = time.time()
        time_since_last_search = current_time - self._last_search_time
        
        if time_since_last_search < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last_search)
        
        self._last_search_time = time.time()
        self._search_count += 1
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search usage statistics."""
        return {
            "total_searches": self._search_count,
            "last_search_time": datetime.fromtimestamp(self._last_search_time),
            "rate_limit_delay": self.rate_limit_delay,
            "max_results_per_query": self.max_results_per_query
        }


# Utility functions for external use
async def quick_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Perform a quick web search with default settings."""
    search_tool = WebSearchTool()
    search_query = SearchQuery(query=query, max_results=max_results)
    return await search_tool.search(search_query)


def create_search_tool(config: Dict[str, Any] = None) -> WebSearchTool:
    """Factory function to create a configured WebSearchTool."""
    return WebSearchTool(config)


# Example usage and testing
if __name__ == "__main__":
    async def test_search():
        search_tool = WebSearchTool({
            'max_results_per_query': 5,
            'extract_full_content': True,
            'rate_limit_delay': 2.0
        })
        
        query = "Azure OpenAI best practices"
        results = await search_tool.search(query)
        
        print(f"Found {len(results)} results for '{query}':")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Relevance: {result.relevance_score:.2f}")
            print(f"   Content length: {len(result.content)} chars")
            print(f"   Snippet: {result.snippet[:100]}...")
    
    # Run test if executed directly
    asyncio.run(test_search())
