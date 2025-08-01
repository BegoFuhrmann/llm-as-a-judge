"""
Information Extraction Agent
===========================

The second agent in the multi-agent system responsible for:
- LangChain-based information extraction from processed documents
- Chroma vector database querying for semantic retrieval
- Structured data extraction using Azure OpenAI
- Financial document specific extraction patterns
- Integration with audit trail for compliance

This agent takes processed documents from the Document Collection Agent
and extracts structured information for decision-making processes.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# LangChain ecosystem
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pydantic for structured outputs
from pydantic import BaseModel, Field
from typing import List as TypingList

import os
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential, get_bearer_token_provider, DefaultAzureCredential
# Azure integration

from ..core.agentic_config import AgenticConfig
from ..core.base_agent import BaseAgent
from ..core.audit_trail import AuditEvent, AuditTrailManager, AuditEventType
from .document_collection import DocumentCollectionAgent, DocumentCollectionResult

logger = logging.getLogger(__name__)


# Pydantic models for structured extraction
class ExtractedEntity(BaseModel):
    """Represents an extracted entity with metadata."""
    entity_type: str = Field(description="Type of entity (person, organization, amount, date, etc.)")
    value: str = Field(description="The extracted value")
    confidence: float = Field(description="Confidence score 0-1")
    context: str = Field(description="Surrounding context")
    source_chunk: str = Field(description="Source document chunk ID")


class FinancialMetric(BaseModel):
    """Represents extracted financial metrics."""
    metric_name: str = Field(description="Name of the financial metric")
    value: float = Field(description="Numeric value")
    currency: Optional[str] = Field(description="Currency if applicable")
    period: Optional[str] = Field(description="Time period")
    source: str = Field(description="Source document reference")


class ExtractedInformation(BaseModel):
    """Complete structured information extraction result."""
    entities: TypingList[ExtractedEntity] = Field(description="Extracted entities")
    financial_metrics: TypingList[FinancialMetric] = Field(description="Financial metrics")
    key_insights: TypingList[str] = Field(description="Key insights and findings")
    risk_indicators: TypingList[str] = Field(description="Identified risk indicators")
    compliance_notes: TypingList[str] = Field(description="Compliance-related notes")
    confidence_score: float = Field(description="Overall extraction confidence")


@dataclass
class ExtractionResult:
    """Result of information extraction processing."""
    extraction_id: str
    source_collection: str
    extracted_info: ExtractedInformation
    processing_metadata: Dict[str, Any]
    audit_trail_id: str
    timestamp: datetime


class InformationExtractionAgent(BaseAgent):
    """
    Information Extraction Agent implementing structured data extraction
    with LangChain integration and Azure OpenAI.
    """
    
    def __init__(self, config: AgenticConfig, audit_manager: AuditTrailManager):
        super().__init__(
            agent_id="info_extraction_001",
            role="information_extraction",
            config=config,
            audit_manager=audit_manager
        )
        
        self.llm: Optional[AzureChatOpenAI] = None
        self.vector_store: Optional[Chroma] = None
        self.extraction_chain: Optional[LLMChain] = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize Azure OpenAI LLM and extraction chains."""
        try:
            # Initialize Azure OpenAI with managed identity
            credential = DefaultAzureCredential()
            
            self.llm = AzureChatOpenAI(
                azure_endpoint=self.config.azure_openai.endpoint,
                azure_deployment=self.config.azure_openai.deployment_name,
                api_version=self.config.azure_openai.api_version,
                azure_ad_token_provider=credential,
                temperature=self.config.azure_openai.temperature,
                max_tokens=self.config.azure_openai.max_tokens
            )
            
            # Initialize vector store connection
            self.vector_store = Chroma(
                collection_name="documents_doc_collection_001",
                persist_directory="./chroma_storage"
            )
            
            # Initialize extraction chain
            self._setup_extraction_chain()
            
            logger.info("Information Extraction Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Information Extraction Agent: {e}")
            raise
    
    def _setup_extraction_chain(self) -> None:
        """Setup the LangChain extraction chain with structured output."""
        
        # Create output parser
        parser = PydanticOutputParser(pydantic_object=ExtractedInformation)
        
        # Create extraction prompt template
        extraction_prompt = PromptTemplate(
            input_variables=["document_content", "extraction_focus"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template="""
You are an expert financial document analyst. Analyze the following document content and extract structured information.

EXTRACTION FOCUS: {extraction_focus}

DOCUMENT CONTENT:
{document_content}

Extract the following information with high accuracy:
1. Named entities (organizations, people, dates, amounts)
2. Financial metrics and KPIs
3. Key insights and findings
4. Risk indicators or warning signs
5. Compliance-related information

Provide confidence scores for each extraction. Be conservative with confidence scores.

{format_instructions}

EXTRACTED INFORMATION:
"""
        )
        
        # Create extraction chain
        self.extraction_chain = LLMChain(
            llm=self.llm,
            prompt=extraction_prompt,
            output_parser=parser
        )
    
    async def extract_information(
        self,
        collection_result: DocumentCollectionResult,
        extraction_focus: str = "comprehensive financial analysis",
        similarity_threshold: float = 0.7
    ) -> ExtractionResult:
        """
        Extract structured information from processed documents.
        
        Args:
            collection_result: Result from Document Collection Agent
            extraction_focus: Specific focus for extraction
            similarity_threshold: Minimum similarity for relevant chunks
            
        Returns:
            ExtractionResult with structured extracted information
        """
        audit_event = AuditEvent(
            event_id="",
            agent_id=self.agent_id,
            event_type=AuditEventType.DATA_PROCESSING,
            action="extract_information",
            input_data={
                "collection_id": collection_result.vector_store_collection,
                "extraction_focus": extraction_focus,
                "num_documents": len(collection_result.processed_documents)
            },
            timestamp=datetime.now()
        )
        
        try:
            self.update_status("processing", "information_extraction")
            
            # Gather relevant document chunks
            relevant_chunks = await self._gather_relevant_chunks(
                collection_result, extraction_focus, similarity_threshold
            )
            
            # Combine chunks for extraction
            combined_content = self._combine_chunks(relevant_chunks)
            
            # Perform extraction using LangChain
            extracted_info = await self._perform_extraction(
                combined_content, extraction_focus
            )
            
            # Generate extraction metadata
            processing_metadata = {
                "chunks_processed": len(relevant_chunks),
                "total_content_length": len(combined_content),
                "extraction_focus": extraction_focus,
                "processing_time": datetime.now(),
                "model_used": self.config.azure_openai.deployment_name
            }
            
            # Create result
            result = ExtractionResult(
                extraction_id=f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source_collection=collection_result.vector_store_collection,
                extracted_info=extracted_info,
                processing_metadata=processing_metadata,
                audit_trail_id=audit_event.event_id,
                timestamp=datetime.now()
            )
            
            # Log performance metrics
            await self.log_performance_metric("extraction_confidence", extracted_info.confidence_score)
            await self.log_performance_metric("entities_extracted", len(extracted_info.entities))
            await self.log_performance_metric("financial_metrics_found", len(extracted_info.financial_metrics))
            
            # Complete audit event
            audit_event.output_data = {
                "extraction_id": result.extraction_id,
                "entities_count": len(extracted_info.entities),
                "financial_metrics_count": len(extracted_info.financial_metrics),
                "overall_confidence": extracted_info.confidence_score,
                "success": True
            }
            audit_event.completed_at = datetime.now()
            await self.audit_manager.log_event(audit_event)
            
            self.update_status("completed", None)
            return result
            
        except Exception as e:
            audit_event.error = str(e)
            audit_event.completed_at = datetime.now()
            await self.audit_manager.log_event(audit_event)
            
            self.update_status("error", None)
            logger.error(f"Information extraction failed: {e}")
            raise
    
    async def _gather_relevant_chunks(
        self,
        collection_result: DocumentCollectionResult,
        query: str,
        threshold: float
    ) -> List[Document]:
        """Gather relevant document chunks using semantic search."""
        
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        try:
            # Perform semantic search
            relevant_docs = self.vector_store.similarity_search_with_score(
                query=query,
                k=20  # Get more chunks for comprehensive extraction
            )
            
            # Filter by similarity threshold
            filtered_docs = [
                doc for doc, score in relevant_docs 
                if score >= threshold
            ]
            
            logger.info(f"Found {len(filtered_docs)} relevant chunks above threshold {threshold}")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Failed to gather relevant chunks: {e}")
            # Fallback: use all chunks from collection result
            all_chunks = []
            for doc in collection_result.processed_documents:
                all_chunks.extend(doc.chunks)
            return all_chunks[:10]  # Limit for processing
    
    def _combine_chunks(self, chunks: List[Document]) -> str:
        """Combine document chunks into coherent content for extraction."""
        
        # Sort chunks by source and preserve structure
        chunks_by_source = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)
        
        # Combine content with source information
        combined_parts = []
        for source, source_chunks in chunks_by_source.items():
            combined_parts.append(f"\n=== SOURCE: {source} ===\n")
            for chunk in source_chunks:
                combined_parts.append(chunk.page_content)
                combined_parts.append("\n---\n")
        
        return "\n".join(combined_parts)
    
    async def _perform_extraction(
        self,
        content: str,
        extraction_focus: str
    ) -> ExtractedInformation:
        """Perform the actual information extraction using LangChain."""
        
        try:
            # Run extraction chain
            result = await self.extraction_chain.arun(
                document_content=content,
                extraction_focus=extraction_focus
            )
            
            # Ensure result is ExtractedInformation object
            if isinstance(result, str):
                # Parse JSON if returned as string
                result_dict = json.loads(result)
                result = ExtractedInformation(**result_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Extraction chain failed: {e}")
            
            # Return empty but valid result
            return ExtractedInformation(
                entities=[],
                financial_metrics=[],
                key_insights=["Extraction failed due to processing error"],
                risk_indicators=["Unable to assess risks due to extraction failure"],
                compliance_notes=["Manual review required due to extraction failure"],
                confidence_score=0.0
            )
    
    async def query_extracted_information(
        self,
        extraction_result: ExtractionResult,
        query: str
    ) -> Dict[str, Any]:
        """
        Query the extracted information using natural language.
        
        Args:
            extraction_result: Previous extraction result
            query: Natural language query
            
        Returns:
            Query response with relevant information
        """
        audit_event = AuditEvent(
            event_id="",
            agent_id=self.agent_id,
            event_type=AuditEventType.USER_INTERACTION,
            action="query_extracted_information",
            input_data={"query": query, "extraction_id": extraction_result.extraction_id},
            timestamp=datetime.now()
        )
        
        try:
            # Create query context from extracted information
            context = self._create_query_context(extraction_result.extracted_info)
            
            # Create query prompt
            query_prompt = PromptTemplate(
                input_variables=["context", "query"],
                template="""
Based on the following extracted information, answer the user's query.

EXTRACTED INFORMATION CONTEXT:
{context}

USER QUERY: {query}

Provide a comprehensive answer based only on the available extracted information.
If the information is not available, clearly state that.

RESPONSE:
"""
            )
            
            # Create query chain
            query_chain = LLMChain(llm=self.llm, prompt=query_prompt)
            
            # Execute query
            response = await query_chain.arun(context=context, query=query)
            
            result = {
                "query": query,
                "response": response,
                "extraction_id": extraction_result.extraction_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Complete audit event
            audit_event.output_data = result
            audit_event.completed_at = datetime.now()
            await self.audit_manager.log_event(audit_event)
            
            return result
            
        except Exception as e:
            audit_event.error = str(e)
            audit_event.completed_at = datetime.now()
            await self.audit_manager.log_event(audit_event)
            
            logger.error(f"Query processing failed: {e}")
            raise
    
    def _create_query_context(self, extracted_info: ExtractedInformation) -> str:
        """Create context string from extracted information for querying."""
        context_parts = []
        
        # Add entities
        if extracted_info.entities:
            context_parts.append("ENTITIES:")
            for entity in extracted_info.entities:
                context_parts.append(f"- {entity.entity_type}: {entity.value} (confidence: {entity.confidence})")
        
        # Add financial metrics
        if extracted_info.financial_metrics:
            context_parts.append("\nFINANCIAL METRICS:")
            for metric in extracted_info.financial_metrics:
                currency_str = f" {metric.currency}" if metric.currency else ""
                period_str = f" ({metric.period})" if metric.period else ""
                context_parts.append(f"- {metric.metric_name}: {metric.value}{currency_str}{period_str}")
        
        # Add insights
        if extracted_info.key_insights:
            context_parts.append("\nKEY INSIGHTS:")
            for insight in extracted_info.key_insights:
                context_parts.append(f"- {insight}")
        
        # Add risk indicators
        if extracted_info.risk_indicators:
            context_parts.append("\nRISK INDICATORS:")
            for risk in extracted_info.risk_indicators:
                context_parts.append(f"- {risk}")
        
        return "\n".join(context_parts)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the Information Extraction Agent.
        
        Args:
            input_data: Should contain 'collection_result' and optional 'extraction_focus'
            
        Returns:
            Processing results
        """
        collection_result = input_data.get("collection_result")
        extraction_focus = input_data.get("extraction_focus", "comprehensive financial analysis")
        
        if not collection_result:
            raise ValueError("collection_result is required in input_data")
        
        result = await self.extract_information(collection_result, extraction_focus)
        
        return {
            "extraction_result": result,
            "success": True,
            "agent_id": self.agent_id
        }
