"""
RAG-based Agent - Handles structured, compliance-sensitive queries using knowledge bases.
"""
import asyncio
import time
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from openai import AsyncAzureOpenAI

# Azure OpenAI and embeddings setup
from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from dotenv import load_dotenv

from ..core.base import BaseAgent, Task, AgentResponse
from ..enums import AgentType, LogLevel
from ..audit.audit_log import audit_logger
from .topic_identification_agent import TopicIdentificationAgent


class RAGAgent(BaseAgent):
    """
    RAG-based agent that handles structured, compliance-sensitive queries.
    
    This agent retrieves precise, traceable information from knowledge bases
    and provides responses with proper source attribution using ChromaDB.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, AgentType.RAG_BASED, config)
        self.config = config or {}
        
        # Set NO_PROXY environment variable for Azure
        os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
        
        # Load environment variables
        load_dotenv()
        
        # Azure OpenAI and LLM setup
        self.llm: Optional[AzureChatOpenAI] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        
        # ChromaDB configuration for different databases
        self.confluence_db_path = r"\\dnsbego.de\dfsbego\home04\FuhrmannD\Documents\01_Trainee\Master\Thesis\code\agentic_system\data\confluence_chroma_db"
        self.newhq_db_path = r"\\dnsbego.de\dfsbego\home04\FuhrmannD\Documents\01_Trainee\Master\Thesis\code\agentic_system\data\newHQ_chroma_db"
        self.newhq_collection_name = "berenberg_newhq_docs"
        
        # Vector databases
        self.confluence_vectordb: Optional[Chroma] = None
        self.newhq_vectordb: Optional[Chroma] = None
        self.confluence_qa_chain: Optional[RetrievalQA] = None
        self.newhq_qa_chain: Optional[RetrievalQA] = None
        
        # RAG parameters
        self.max_retrieval_results = self.config.get('max_retrieval_results', 5)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.max_context_length = self.config.get('max_context_length', 8000)
        self.temperature = self.config.get('temperature', 0)
        
        # Initialize tools
        self.confluence_tool: Optional[Tool] = None
        self.newhq_tool: Optional[Tool] = None
        
        # Topic Identification Agent for intelligent routing
        self.topic_agent: Optional[TopicIdentificationAgent] = None
        
        # Database selection mapping based on topics
        self.topic_to_database_mapping = {
            # Confluence database topics
            'confluence': ['confluence', 'project', 'baia', 'begochat', 'rpa', 'ai', 'automation', 
                          'development', 'technology', 'innovation', 'digital', 'workflow'],
            # NewHQ database topics  
            'newhq': ['office', 'building', 'parking', 'facilities', 'location', 'space', 
                     'headquarters', 'workplace', 'infrastructure', 'amenities', 'newhq']
        }
        
        # Response templates
        self.rag_prompt_template = """
        You are a knowledgeable assistant providing accurate information based on retrieved context.
        
        User Query: {query}
        
        Retrieved Context:
        {context}
        
        Instructions:
        1. Answer the query using ONLY the information provided in the context
        2. If the context doesn't contain sufficient information, clearly state this
        3. Cite specific sources when making claims
        4. Be precise and avoid speculation
        5. For compliance-related queries, emphasize accuracy and provide exact references
        
        Response:
        """
        
        self.compliance_prompt_template = """
        You are a compliance expert providing regulatory guidance based on official documentation.
        
        User Query: {query}
        
        Retrieved Regulatory Context:
        {context}
        
        Instructions:
        1. Provide accurate compliance information based on the retrieved context
        2. Cite specific regulations, sections, and sources
        3. Highlight any requirements, obligations, or restrictions
        4. If information is incomplete, clearly state what additional sources should be consulted
        5. Use precise legal language where appropriate
        6. Include confidence level in your assessment
        
        Compliance Response:
        """
    
    async def initialize(self) -> bool:
        """Initialize Azure OpenAI, embeddings, and ChromaDB connections."""
        try:
            # Initialize Azure OpenAI with DefaultAzureCredential
            await self._initialize_azure_openai()
            
            # Initialize HuggingFace embeddings
            await self._initialize_embeddings()
            
            # Initialize ChromaDB databases
            await self._initialize_chromadb_databases()
            
            # Initialize tools
            await self._initialize_tools()
            
            # Initialize Topic Identification Agent
            await self._initialize_topic_agent()
            
            self.is_active = True
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "initialize",
                log_level=LogLevel.INFO
            )
            return True
            
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "initialize_failed",
                log_level=LogLevel.ERROR,
                error=str(e)
            )
            return False
    
    async def _initialize_azure_openai(self):
        """Initialize Azure OpenAI with DefaultAzureCredential."""
        try:
            default_credential = DefaultAzureCredential()
            token = default_credential.get_token("https://cognitiveservices.azure.com/.default")
            
            # Set token to environment variable
            os.environ["AZURE_OPENAI_API_KEY"] = token.token
            
            self.llm = AzureChatOpenAI(
                azure_endpoint="https://begobaiatest.openai.azure.com/",
                azure_deployment="gpt-4o-mini",
                openai_api_version="2024-05-01-preview",
                temperature=self.temperature,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "azure_openai_initialized",
                log_level=LogLevel.INFO
            )
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "azure_openai_init_failed",
                log_level=LogLevel.ERROR,
                error=str(e)
            )
            raise
    
    async def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings."""
        try:
            EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}
            )
            
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "embeddings_initialized",
                log_level=LogLevel.INFO
            )
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "embeddings_init_failed",
                log_level=LogLevel.ERROR,
                error=str(e)
            )
            raise
    
    async def _initialize_chromadb_databases(self):
        """Initialize ChromaDB databases for Confluence and NewHQ."""
        try:
            # Initialize Confluence ChromaDB
            if os.path.exists(self.confluence_db_path) and len(os.listdir(self.confluence_db_path)) > 0:
                self.confluence_vectordb = Chroma(
                    persist_directory=self.confluence_db_path,
                    embedding_function=self.embeddings
                )
                
                # Create QA chain for Confluence
                confluence_retriever = self.confluence_vectordb.as_retriever(search_kwargs={"k": self.max_retrieval_results})
                self.confluence_qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=confluence_retriever,
                    return_source_documents=True
                )
                
                await audit_logger.log_agent_action(
                    self.agent_id, self.agent_type, "confluence_db_loaded",
                    log_level=LogLevel.INFO
                )
            
            # Initialize NewHQ ChromaDB
            if os.path.exists(self.newhq_db_path) and len(os.listdir(self.newhq_db_path)) > 0:
                self.newhq_vectordb = Chroma(
                    persist_directory=self.newhq_db_path,
                    embedding_function=self.embeddings,
                    collection_name=self.newhq_collection_name
                )
                
                # Check document count
                try:
                    doc_count = self.newhq_vectordb._collection.count()
                    if doc_count > 0:
                        # Create QA chain for NewHQ
                        newhq_retriever = self.newhq_vectordb.as_retriever(search_kwargs={"k": self.max_retrieval_results})
                        self.newhq_qa_chain = RetrievalQA.from_chain_type(
                            llm=self.llm,
                            chain_type="stuff",
                            retriever=newhq_retriever,
                            return_source_documents=True
                        )
                        
                        await audit_logger.log_agent_action(
                            self.agent_id, self.agent_type, "newhq_db_loaded",
                            log_level=LogLevel.INFO,
                            doc_count=doc_count
                        )
                    else:
                        await audit_logger.log_agent_action(
                            self.agent_id, self.agent_type, "newhq_db_empty",
                            log_level=LogLevel.WARNING
                        )
                except Exception as e:
                    await audit_logger.log_agent_action(
                        self.agent_id, self.agent_type, "newhq_db_access_error",
                        log_level=LogLevel.WARNING,
                        error=str(e)
                    )
                    
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "chromadb_init_failed",
                log_level=LogLevel.ERROR,
                error=str(e)
            )
            raise
    
    async def _initialize_tools(self):
        """Initialize search tools for both databases."""
        try:
            # Confluence search tool
            self.confluence_tool = Tool(
                name="confluence_search",
                func=self._confluence_search,
                description="Searches the Confluence vector database to retrieve, filter, and summarize relevant documentation about RPA, AI, and specific projects like BAIA or BegoChat. Provide your search query as input to get relevant information from the indexed Confluence data."
            )
            
            # NewHQ search tool
            self.newhq_tool = Tool(
                name="newhq_search",
                func=self._newhq_search,
                description="Searches the NewHQ vector database for information about office facilities, parking, and building-related queries. Provide your search query as input to get relevant information about the new headquarters."
            )
            
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "tools_initialized",
                log_level=LogLevel.INFO
            )
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "tools_init_failed",
                log_level=LogLevel.ERROR,
                error=str(e)
            )
            raise
    
    async def _initialize_topic_agent(self):
        """Initialize the Topic Identification Agent for intelligent routing."""
        try:
            self.topic_agent = TopicIdentificationAgent(
                agent_id=f"{self.agent_id}_topic_identifier",
                config=self.config
            )
            
            # Initialize the topic agent
            await self.topic_agent.initialize()
            
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "topic_agent_initialized",
                log_level=LogLevel.INFO
            )
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "topic_agent_init_failed",
                log_level=LogLevel.WARNING,
                error=str(e)
            )
            # Continue without topic agent - fallback to keyword-based routing
            self.topic_agent = None
    
    def _confluence_search(self, query: str) -> str:
        """
        Search the Confluence vector database and return a formatted response.
        """
        try:
            # Ensure query is a string
            if not isinstance(query, str):
                query = str(query)
            
            print(f"[DEBUG] Searching Confluence for: {query}")
            
            # Check if we have a working QA chain
            if self.confluence_qa_chain is None:
                return f"I searched for information about '{query}', but the Confluence vector database appears to be empty or not properly loaded. Please check the database configuration."
            
            # Call the QA chain with the query
            result = self.confluence_qa_chain.invoke({"query": query})
            
            print(f"[DEBUG] Confluence QA chain result keys: {result.keys()}")
            
            # Extract the answer from the result
            answer = result.get("result", "No answer found")
            source_docs = result.get("source_documents", [])
            
            print(f"[DEBUG] Found {len(source_docs)} source documents")
            
            # Format the response with sources
            response = f"Based on my search of the Confluence database for '{query}':\n\n{answer}"
            
            if source_docs:
                response += "\n\nSources:\n"
                for i, doc in enumerate(source_docs[:3], 1):  # Limit to top 3 sources
                    metadata = doc.metadata
                    source_info = metadata.get("source", "Unknown source")
                    response += f"{i}. {source_info}\n"
            
            return response
            
        except Exception as e:
            error_msg = f"I encountered an error while searching Confluence for '{query}': {str(e)}"
            print(f"[DEBUG] {error_msg}")
            return error_msg
    
    def _newhq_search(self, query: str) -> str:
        """
        Search the NewHQ vector database and return a formatted response.
        """
        try:
            # Ensure query is a string
            if not isinstance(query, str):
                query = str(query)
            
            print(f"[DEBUG] Searching NewHQ for: {query}")
            
            # Check if we have a working QA chain
            if self.newhq_qa_chain is None:
                return f"I searched for information about '{query}', but the NewHQ vector database appears to be empty or not properly loaded. Please check the database configuration."
            
            # Call the QA chain with the query
            result = self.newhq_qa_chain.invoke({"query": query})
            
            print(f"[DEBUG] NewHQ QA chain result keys: {result.keys()}")
            
            # Extract the answer from the result
            answer = result.get("result", "No answer found")
            source_docs = result.get("source_documents", [])
            
            print(f"[DEBUG] Found {len(source_docs)} source documents")
            
            # Format the response with sources
            response = f"Based on my search of the NewHQ database for '{query}':\n\n{answer}"
            
            if source_docs:
                response += "\n\nSources:\n"
                for i, doc in enumerate(source_docs[:3], 1):  # Limit to top 3 sources
                    metadata = doc.metadata
                    source_info = metadata.get("source", "Unknown source")
                    # Clean up the source path for better readability
                    if "\\15 newHQ data\\_new\\" in source_info:
                        source_info = source_info.split("\\15 newHQ data\\_new\\")[-1]
                    response += f"{i}. {source_info}\n"
            
            return response
            
        except Exception as e:
            error_msg = f"I encountered an error while searching NewHQ for '{query}': {str(e)}"
            print(f"[DEBUG] {error_msg}")
            return error_msg
    
    def get_search_tools(self) -> List[Tool]:
        """Get the available search tools."""
        tools = []
        if self.confluence_tool:
            tools.append(self.confluence_tool)
        if self.newhq_tool:
            tools.append(self.newhq_tool)
        return tools
    async def process(self, task: Task) -> AgentResponse:
        """
        Process a query using RAG methodology with ChromaDB.
        
        Args:
            task: Task containing the query and routing information
            
        Returns:
            AgentResponse with RAG-based answer and source attribution
        """
        start_time = time.time()
        
        try:
            query = task.input_data.get('query', '')
            identification_result = task.input_data.get('identification_result', {})
            
            if not query:
                return AgentResponse(
                    success=False,
                    error_message="No query provided for RAG processing",
                    execution_time=time.time() - start_time
                )
            
            # Determine which database to search based on query content
            search_result = await self._smart_search(query, identification_result)
            
            execution_time = time.time() - start_time
            
            response = AgentResponse(
                success=True,
                data={
                    'query': query,
                    'answer': search_result['answer'],
                    'search_method': search_result['method'],
                    'database_used': search_result['database'],
                    'rag_metadata': {
                        'processing_method': 'chromadb_search',
                        'execution_time': execution_time
                    }
                },
                confidence_score=search_result.get('confidence', 0.8),
                execution_time=execution_time,
                sources=search_result.get('sources', [])
            )
            
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "rag_processing_completed",
                task=task, response=response, log_level=LogLevel.INFO
            )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_response = AgentResponse(
                success=False,
                error_message=f"RAG processing failed: {str(e)}",
                execution_time=execution_time
            )
            
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "rag_processing_failed",
                task=task, response=error_response, log_level=LogLevel.ERROR
            )
            
            return error_response
    
    async def _smart_search(self, query: str, identification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Determine which database to search using topic identification and perform the search."""
        
        # First, try to use existing identification result if available
        if identification_result:
            database_choice = self._determine_database_from_identification(identification_result)
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "using_existing_topic_identification",
                log_level=LogLevel.INFO,
                database_choice=database_choice
            )
        else:
            # Use topic identification agent for intelligent routing
            database_choice = await self._identify_database_using_topics(query)
        
        try:
            if database_choice == 'newhq' and self.newhq_tool:
                # Search NewHQ database
                answer = self._newhq_search(query)
                return {
                    'answer': answer,
                    'method': 'newhq_search_topic_identified',
                    'database': 'NewHQ ChromaDB',
                    'confidence': 0.85,
                    'sources': ['NewHQ Database'],
                    'selection_reason': 'Topic identification indicated office/facility query'
                }
            elif database_choice == 'confluence' and self.confluence_tool:
                # Search Confluence database
                answer = self._confluence_search(query)
                return {
                    'answer': answer,
                    'method': 'confluence_search_topic_identified',
                    'database': 'Confluence ChromaDB',
                    'confidence': 0.85,
                    'sources': ['Confluence Database'],
                    'selection_reason': 'Topic identification indicated project/technical query'
                }
            elif database_choice == 'both':
                # Search both databases and combine results
                return await self._search_both_databases(query)
            else:
                # Fallback to keyword-based search
                return await self._fallback_keyword_search(query)
                    
        except Exception as e:
            return {
                'answer': f"I encountered an error while searching for '{query}': {str(e)}",
                'method': 'error',
                'database': 'Error',
                'confidence': 0.1,
                'sources': [],
                'error': str(e)
            }
    
    async def _identify_database_using_topics(self, query: str) -> str:
        """Use Topic Identification Agent to determine the best database."""
        if not self.topic_agent:
            return await self._fallback_keyword_routing(query)
        
        try:
            # Create a task for topic identification
            topic_task = Task(
                task_id=f"topic_id_{int(time.time())}",
                input_data={'query': query},
                agent_type=AgentType.TOPIC_IDENTIFICATION,
                priority='medium'
            )
            
            # Process the task with topic identification agent
            topic_response = await self.topic_agent.process(topic_task)
            
            if topic_response.success:
                identification_result = topic_response.data.get('identification_result', {})
                return self._determine_database_from_identification(identification_result)
            else:
                await audit_logger.log_agent_action(
                    self.agent_id, self.agent_type, "topic_identification_failed",
                    log_level=LogLevel.WARNING,
                    error=topic_response.error_message
                )
                return await self._fallback_keyword_routing(query)
                
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "topic_agent_error",
                log_level=LogLevel.WARNING,
                error=str(e)
            )
            return await self._fallback_keyword_routing(query)
    
    def _determine_database_from_identification(self, identification_result: Dict[str, Any]) -> str:
        """Determine database selection based on topic identification results."""
        topics = identification_result.get('topics', [])
        metadata = identification_result.get('metadata', {})
        domain = metadata.get('domain', '').lower()
        
        confluence_score = 0
        newhq_score = 0
        
        # Score based on identified topics
        for topic in topics:
            topic_lower = topic.lower()
            
            # Check confluence keywords
            if any(keyword in topic_lower for keyword in self.topic_to_database_mapping['confluence']):
                confluence_score += 1
            
            # Check newhq keywords
            if any(keyword in topic_lower for keyword in self.topic_to_database_mapping['newhq']):
                newhq_score += 1
        
        # Score based on domain
        if domain in ['technical', 'business', 'compliance']:
            confluence_score += 2
        elif domain in ['facilities', 'office', 'building']:
            newhq_score += 2
        
        # Determine database choice
        if newhq_score > confluence_score:
            return 'newhq'
        elif confluence_score > newhq_score:
            return 'confluence'
        elif confluence_score > 0 and newhq_score > 0:
            return 'both'  # Search both if unclear
        else:
            return 'confluence'  # Default to confluence for general queries
    
    async def _fallback_keyword_routing(self, query: str) -> str:
        """Fallback keyword-based routing when topic identification is not available."""
        query_lower = query.lower()
        
        # Keywords for different databases
        newhq_keywords = self.topic_to_database_mapping['newhq']
        confluence_keywords = self.topic_to_database_mapping['confluence']
        
        # Count keyword matches
        newhq_matches = sum(1 for keyword in newhq_keywords if keyword in query_lower)
        confluence_matches = sum(1 for keyword in confluence_keywords if keyword in query_lower)
        
        if newhq_matches > confluence_matches:
            return 'newhq'
        elif confluence_matches > newhq_matches:
            return 'confluence'
        else:
            return 'confluence'  # Default fallback
    
    async def _search_both_databases(self, query: str) -> Dict[str, Any]:
        """Search both databases and combine results when topic is ambiguous."""
        confluence_result = None
        newhq_result = None
        
        try:
            # Search both databases
            if self.confluence_tool:
                confluence_answer = self._confluence_search(query)
                confluence_result = {
                    'answer': confluence_answer,
                    'database': 'Confluence',
                    'success': 'error' not in confluence_answer.lower()
                }
            
            if self.newhq_tool:
                newhq_answer = self._newhq_search(query)
                newhq_result = {
                    'answer': newhq_answer,
                    'database': 'NewHQ',
                    'success': 'error' not in newhq_answer.lower()
                }
            
            # Combine results
            combined_answer = "I searched both our knowledge bases for your query:\n\n"
            
            if confluence_result and confluence_result['success']:
                combined_answer += f"**From Confluence Database:**\n{confluence_result['answer']}\n\n"
            
            if newhq_result and newhq_result['success']:
                combined_answer += f"**From NewHQ Database:**\n{newhq_result['answer']}\n\n"
            
            if not (confluence_result and confluence_result['success']) and not (newhq_result and newhq_result['success']):
                combined_answer = "I couldn't find relevant information in either of our knowledge bases. Please try rephrasing your query or contact support for assistance."
            
            return {
                'answer': combined_answer,
                'method': 'dual_database_search',
                'database': 'Both (Confluence + NewHQ)',
                'confidence': 0.75,
                'sources': ['Confluence Database', 'NewHQ Database'],
                'selection_reason': 'Ambiguous topic required searching both databases'
            }
            
        except Exception as e:
            return {
                'answer': f"I encountered an error while searching both databases for '{query}': {str(e)}",
                'method': 'dual_search_error',
                'database': 'Both (Error)',
                'confidence': 0.1,
                'sources': [],
                'error': str(e)
            }
    
    async def _fallback_keyword_search(self, query: str) -> Dict[str, Any]:
        """Fallback search when neither database is clearly indicated."""
        # Try confluence first as it's more general
        if self.confluence_tool:
            answer = self._confluence_search(query)
            if "error" not in answer.lower() and "not found" not in answer.lower():
                return {
                    'answer': answer,
                    'method': 'confluence_fallback',
                    'database': 'Confluence ChromaDB',
                    'confidence': 0.6,
                    'sources': ['Confluence Database'],
                    'selection_reason': 'Fallback to Confluence database'
                }
        
        # Try NewHQ if confluence didn't work
        if self.newhq_tool:
            answer = self._newhq_search(query)
            return {
                'answer': answer,
                'method': 'newhq_fallback',
                'database': 'NewHQ ChromaDB',
                'confidence': 0.6,
                'sources': ['NewHQ Database'],
                'selection_reason': 'Fallback to NewHQ database'
            }
        
        return {
            'answer': f"I apologize, but I couldn't find relevant information for '{query}'. The ChromaDB databases may not be properly initialized or may be empty.",
            'method': 'no_database_available',
            'database': 'None',
            'confidence': 0.1,
            'sources': [],
            'selection_reason': 'No databases available'
        }
    
    async def shutdown(self) -> bool:
        """Shutdown the RAG agent gracefully."""
        try:
            # Shutdown topic agent
            if self.topic_agent:
                await self.topic_agent.shutdown()
                self.topic_agent = None
            
            # ChromaDB clients don't need explicit cleanup
            self.confluence_vectordb = None
            self.newhq_vectordb = None
            self.confluence_qa_chain = None
            self.newhq_qa_chain = None
            
            self.is_active = False
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "shutdown",
                log_level=LogLevel.INFO
            )
            return True
            
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "shutdown_failed",
                log_level=LogLevel.ERROR,
                error=str(e)
            )
            return False
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge bases."""
        try:
            stats = {
                'confluence_db': {
                    'status': 'available' if self.confluence_vectordb else 'unavailable',
                    'path': self.confluence_db_path,
                    'qa_chain': 'initialized' if self.confluence_qa_chain else 'not_initialized'
                },
                'newhq_db': {
                    'status': 'available' if self.newhq_vectordb else 'unavailable',
                    'path': self.newhq_db_path,
                    'collection_name': self.newhq_collection_name,
                    'qa_chain': 'initialized' if self.newhq_qa_chain else 'not_initialized'
                },
                'configuration': {
                    'max_retrieval_results': self.max_retrieval_results,
                    'similarity_threshold': self.similarity_threshold,
                    'temperature': self.temperature
                }
            }
            
            # Try to get document counts
            if self.confluence_vectordb:
                try:
                    stats['confluence_db']['document_count'] = self.confluence_vectordb._collection.count()
                except:
                    stats['confluence_db']['document_count'] = 'unknown'
            
            if self.newhq_vectordb:
                try:
                    stats['newhq_db']['document_count'] = self.newhq_vectordb._collection.count()
                except:
                    stats['newhq_db']['document_count'] = 'unknown'
            
            return stats
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }