import sys, os
# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import pytest
from agentic_system.tools.web_search import quick_search


def test_quick_search_basic():
    """Basic functionality test for WebSearchTool.quick_search"""
    # Perform a quick search for a simple query
    results = asyncio.run(quick_search("What is Python programming?", max_results=3))
    
    # Validate results structure
    assert isinstance(results, list), "Results should be a list"
    # If any results are returned, validate their structure
    for res in results:
        assert hasattr(res, 'title'), "Result should have a title"
        assert hasattr(res, 'url'), "Result should have a URL"
        assert res.url.startswith("http"), "URL should be a valid http link"
        assert hasattr(res, 'relevance_score'), "Result should have a relevance score"
        assert 0.0 <= res.relevance_score <= 1.0, "Relevance score should be between 0 and 1"
