"""
Web Search Tool - Provides external knowledge acquisition through Tavily search.
"""
import asyncio
import time
import os
import ssl
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import re
from dataclasses import dataclass, field
import dotenv

# Load environment variables from the project root
dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Web search
import aiohttp
import requests
from langchain_tavily import TavilySearch

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
    """Represents a search query with configuration for Tavily."""
    query: str = ""
    max_results: int = 5
    search_scope: SearchScope = SearchScope.WEB_SEARCH
    filters: Dict[str, Any] = field(default_factory=dict)
    topic: str = "general"  # "general", "news", or "finance"
    include_answer: bool = False
    include_raw_content: bool = True
    include_images: bool = False
    include_image_descriptions: bool = False
    search_depth: str = "basic"  # "basic" or "advanced"
    time_range: Optional[str] = None  # "day", "week", "month", or "year"
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None


class WebSearchTool:
    """
    Advanced web search tool using Tavily for external knowledge acquisition.
    
    Features:
    - Async search capabilities with Tavily API
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
        
        # Initialize Tavily search client
        self.tavily_search = self._create_tavily_client()
        
        # Rate limiting
        self._last_search_time = 0
        self._search_count = 0
        
        # User agent for web scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Load environment variables
        self._load_environment_config()
        
    def _create_tavily_client(self):
        """Create Tavily search client."""
        try:
            # Verify API key is available
            api_key = os.getenv('TAVILY_API_KEY')
            if not api_key:
                print("Warning: TAVILY_API_KEY not found in environment variables.")
                print("Please add your Tavily API key to your .env file.")
                print("You can get an API key from: https://tavily.com/")
                return None
                
            # Create Tavily search tool with default settings
            return TavilySearch(
                max_results=self.max_results_per_query,
                topic="general",
                include_answer=False,
                include_raw_content=True,
                include_images=False,
                include_image_descriptions=False,
                search_depth="basic"
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Tavily search client: {e}")
            return None
        
    def _load_environment_config(self):
        """Load environment configuration."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # dotenv not available, skip
        
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
        """Perform the actual Tavily search."""
        try:
            # Check if Tavily client is available
            if self.tavily_search is None:
                await audit_logger.log_agent_action(
                    agent_id="web_search_tool",
                    agent_type=AgentType.AUTONOMOUS,
                    action="tavily_unavailable",
                    log_level=LogLevel.WARNING,
                    query=search_query.query,
                    message="Tavily search client not available, returning mock results"
                )
                
                # Return mock results when Tavily is unavailable
                return [{
                    'title': f"Search Results for: {search_query.query}",
                    'url': "https://example.com/search-unavailable",
                    'content': f"Tavily search is not configured. Please add your TAVILY_API_KEY to the .env file. Query was: '{search_query.query}'",
                    'score': 0.5
                }]
            
            # Update Tavily search configuration based on query parameters
            tavily_config = {
                'max_results': min(search_query.max_results, self.max_results_per_query),
                'topic': search_query.topic,
                'include_answer': search_query.include_answer,
                'include_raw_content': search_query.include_raw_content,
                'include_images': search_query.include_images,
                'include_image_descriptions': search_query.include_image_descriptions,
                'search_depth': search_query.search_depth,
            }
            
            # Add optional parameters if specified
            if search_query.time_range:
                tavily_config['time_range'] = search_query.time_range
            if search_query.include_domains:
                tavily_config['include_domains'] = search_query.include_domains
            if search_query.exclude_domains:
                tavily_config['exclude_domains'] = search_query.exclude_domains
            
            # Create a new TavilySearch instance with updated configuration
            tavily_search = TavilySearch(**tavily_config)
            
            # Perform the search using Tavily's invoke method
            search_results = tavily_search.invoke({"query": search_query.query})
            
            # Parse the JSON response if it's a string
            if isinstance(search_results, str):
                search_results = json.loads(search_results)
            
            # Extract results from Tavily response format
            if isinstance(search_results, dict) and 'results' in search_results:
                return search_results['results']
            else:
                return []
            
        except Exception as e:
            # Audit log Tavily search error
            await audit_logger.log_agent_action(
                agent_id="web_search_tool",
                agent_type=AgentType.AUTONOMOUS,
                action="tavily_search_error",
                log_level=LogLevel.ERROR,
                query=search_query.query,
                error=str(e),
                error_type=type(e).__name__
            )
            return []
    
    async def _process_search_results(self, raw_results: List[Dict[str, Any]], 
                                     search_query: SearchQuery) -> List[SearchResult]:
        """Process raw search results from Tavily."""
        processed_results = []
        
        for result in raw_results:
            try:
                # Create base SearchResult from Tavily result format
                search_result = SearchResult(
                    title=result.get('title', ''),
                    url=result.get('url', ''),
                    snippet=result.get('content', ''),
                    content=result.get('raw_content', '') or result.get('content', ''),
                    metadata={
                        'raw_result': result,
                        'search_query': search_query.query,
                        'score': result.get('score', 0.0)
                    }
                )
                
                # Use Tavily's score as relevance score if available
                search_result.relevance_score = result.get('score', 0.0)
                
                # If no score provided, calculate our own
                if search_result.relevance_score == 0.0:
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
                    url=result.get('url', 'unknown'),
                    error=str(e)
                )
                continue
        
        return processed_results
    
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
            'max_results_per_query': 3,
            'extract_full_content': True,
            'rate_limit_delay': 2.0
        })
        
        # Test basic search
        query = "Azure OpenAI best practices"
        results = await search_tool.search(query)
        
        print(f"Found {len(results)} results for '{query}':")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Relevance: {result.relevance_score:.2f}")
            print(f"   Content length: {len(result.content)} chars")
            print(f"   Snippet: {result.snippet[:100]}...")
        
        # Test advanced search with domains
        search_query = SearchQuery(
            query="Azure Functions deployment",
            max_results=3,
            include_domains=["docs.microsoft.com"],
            topic="general",
            search_depth="advanced"
        )
        
        advanced_results = await search_tool.search(search_query)
        print(f"\n\nAdvanced search results ({len(advanced_results)} results):")
        for i, result in enumerate(advanced_results):
            print(f"{i+1}. {result.title} - {result.url}")
    
    # Run test if executed directly
    asyncio.run(test_search())
