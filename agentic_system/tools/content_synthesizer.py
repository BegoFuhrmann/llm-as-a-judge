"""
Content Synthesizer Tool - Uses LLMs to summarize and synthesize text content.
"""
import asyncio
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv

# Azure OpenAI and LangChain
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Handle imports for both module and direct execution
try:
    from ..core.base import AgentResponse
    from ..enums import AgentType, LogLevel
    from ..audit.audit_log import audit_logger
    from .web_search import SearchResult
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir.parent))
    
    from agentic_system.core.base import AgentResponse
    from agentic_system.enums import AgentType, LogLevel
    from agentic_system.audit.audit_log import audit_logger
    from agentic_system.tools.web_search import SearchResult


class ContentSynthesizer:
    """
    A tool for synthesizing content from various sources using an LLM.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Load environment variables from .env
        load_dotenv()
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", self.config.get("azure_deployment", "gpt-4"))
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        api_key = os.getenv("AZURE_CLIENT_SECRET")
        # Other confidentials
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        openai_scope = os.getenv("OPENAI_AZURE_SCOPE")
        requests_ca_bundle = os.getenv("REQUESTS_CA_BUNDLE")
        # Langfuse
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        # Confluence
        confluence_token = os.getenv("CONFLUENCE_TOKEN")
        # Initialize Azure LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 1500)
        )
        
        # Default prompt template
        self.prompt_template = self.config.get("prompt_template", """
            Based on the following context, please synthesize a clear and concise answer to the user's query.
            
            Context:
            {context}
            
            Query: {query}
            
            Synthesized Answer:
        """)
        
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.prompt_template)
        )

    async def synthesize(self, query: str, search_results: List[SearchResult]) -> AgentResponse:
        """
        Synthesizes content from a list of search results.
        """
        context = "\n\n".join([f"Source URL: {res.url}\nContent: {res.content}" for res in search_results])
        
        await audit_logger.log_system_event(
            event_type="synthesis_started",
            message=f"Starting content synthesis for query: {query}",
            log_level=LogLevel.INFO,
            query=query,
            context_length=len(context)
        )
        
        try:
            response = await self.llm_chain.acall(inputs={"context": context, "query": query})
            synthesized_text = response["text"]
            
            await audit_logger.log_system_event(
                event_type="synthesis_completed",
                message="Content synthesis successful.",
                log_level=LogLevel.INFO,
                query=query
            )
            
            return AgentResponse(
                success=True,
                data={"synthesized_text": synthesized_text},
                sources=[res.url for res in search_results]
            )
            
        except Exception as e:
            await audit_logger.log_system_event(
                event_type="synthesis_failed",
                message=f"Content synthesis failed: {e}",
                log_level=LogLevel.ERROR,
                query=query,
                error=str(e)
            )
            return AgentResponse(success=False, error_message=str(e))
