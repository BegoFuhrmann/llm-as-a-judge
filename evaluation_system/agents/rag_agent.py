"""
Base RAG Agent for the RAG vs Agentic AI Evaluation Framework.

This module provides a simplified RAG agent specifically designed for the evaluation system,
focusing on regulatory and compliance knowledge retrieval.
"""

import os
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings

try:
    from langchain_openai import AzureChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from dotenv import load_dotenv


class RAGAgent:
    """
    Base RAG agent for the evaluation system.
    
    Provides foundation for specialized compliance and regulatory agents
    that need to query knowledge bases for evaluation purposes.
    """
    
    def __init__(self, collection_name: str = None, db_path: str = None, agent_name: str = "RAG Agent"):
        """
        Initialize the RAG agent.
        
        Args:
            collection_name: Name of the ChromaDB collection
            db_path: Path to the ChromaDB database
            agent_name: Human-readable name for the agent
        """
        self.collection_name = collection_name or "default_collection"
        self.db_path = db_path
        self.agent_name = agent_name
        
        # Set NO_PROXY environment variable for Azure
        os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
        
        # Load environment variables
        load_dotenv()
        
        # Components
        self.llm: Optional[AzureChatOpenAI] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vectordb: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA] = None
        
        # Configuration
        self.max_retrieval_results = 5
        self.temperature = 0
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize the RAG agent components."""
        if not LANGCHAIN_AVAILABLE:
            print(f"Warning: LangChain dependencies not available for {self.agent_name}")
            return False
        
        try:
            # Initialize components
            self._initialize_llm()
            self._initialize_embeddings()
            if self.db_path:
                self._initialize_vectordb()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing {self.agent_name}: {str(e)}")
            return False
    
    def _initialize_llm(self):
        """Initialize Azure OpenAI LLM."""
        try:
            from azure.identity import DefaultAzureCredential
            
            # Get Azure credential token
            credential = DefaultAzureCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            
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
            
        except Exception as e:
            print(f"Failed to initialize Azure OpenAI for {self.agent_name}: {str(e)}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings."""
        try:
            EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}
            )
            
        except Exception as e:
            print(f"Failed to initialize embeddings for {self.agent_name}: {str(e)}")
            raise
    
    def _initialize_vectordb(self):
        """Initialize ChromaDB vector database."""
        try:
            if not os.path.exists(self.db_path):
                print(f"Warning: Database path {self.db_path} does not exist for {self.agent_name}")
                return
            
            if len(os.listdir(self.db_path)) == 0:
                print(f"Warning: Database path {self.db_path} is empty for {self.agent_name}")
                return
            
            # Initialize ChromaDB
            self.vectordb = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Create QA chain
            retriever = self.vectordb.as_retriever(
                search_kwargs={"k": self.max_retrieval_results}
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            # Test document count
            doc_count = self.vectordb._collection.count()
            print(f"Initialized {self.agent_name} with {doc_count} documents")
            
        except Exception as e:
            print(f"Failed to initialize vector database for {self.agent_name}: {str(e)}")
            raise
    
    def query(self, query: str, n_results: int = None) -> Dict[str, Any]:
        """
        Query the knowledge base.
        
        Args:
            query: The query string
            n_results: Number of results to return (defaults to max_retrieval_results)
            
        Returns:
            Dictionary containing the query results
        """
        if not self.is_initialized:
            if not self.initialize():
                return {
                    "error": f"{self.agent_name} not properly initialized",
                    "answer": "Unable to process query - initialization failed",
                    "sources": []
                }
        
        if not self.qa_chain:
            return {
                "error": f"No knowledge base available for {self.agent_name}",
                "answer": "Unable to process query - no knowledge base loaded",
                "sources": []
            }
        
        try:
            # Execute the query
            result = self.qa_chain.invoke({"query": query})
            
            # Extract results
            answer = result.get("result", "No answer found")
            source_docs = result.get("source_documents", [])
            
            # Format sources
            sources = []
            for doc in source_docs:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "query": query,
                "agent": self.agent_name
            }
            
        except Exception as e:
            error_msg = f"Error querying {self.agent_name}: {str(e)}"
            return {
                "error": error_msg,
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "sources": []
            }
    
    def search_documents(self, query: str, n_results: int = None) -> List[Dict[str, Any]]:
        """
        Search documents directly without LLM processing.
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            List of matching documents with similarity scores
        """
        if not self.vectordb:
            return []
        
        try:
            n_results = n_results or self.max_retrieval_results
            
            # Perform similarity search
            docs = self.vectordb.similarity_search_with_score(
                query, k=n_results
            )
            
            results = []
            for doc, score in docs:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error searching documents in {self.agent_name}: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        if not self.vectordb:
            return {
                "status": "not_initialized",
                "agent_name": self.agent_name,
                "database_path": self.db_path
            }
        
        try:
            doc_count = self.vectordb._collection.count()
            
            return {
                "status": "active",
                "agent_name": self.agent_name,
                "database_path": self.db_path,
                "collection_name": self.collection_name,
                "document_count": doc_count,
                "max_retrieval_results": self.max_retrieval_results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "agent_name": self.agent_name,
                "error": str(e)
            }
