"""
Document Collection Agent
========================

The first agent in the multi-agent system responsible for:
- Multi-modal document parsing (PDF, DOCX, tables, images)
- LangChain-based document processing pipeline
- Chroma vector database indexing
- Integration with Azure OpenAI for semantic understanding

This agent processes document collections and prepares structured data
for the Information Extraction Agent while maintaining comprehensive
audit trails for regulatory compliance.
"""

import asyncio
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Document processing libraries
import pdfplumber
import fitz  # PyMuPDF
from docx import Document as DocxDocument
import pandas as pd
from PIL import Image

# LangChain ecosystem
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader
)

from ..core.agentic_config import AgenticConfig, DocumentProcessingConfig
from ..core.base_agent import BaseAgent
from ..core.audit_trail import AuditEvent, AuditTrailManager

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Structured representation of a processed document."""
    document_id: str
    source_path: str
    content: str
    metadata: Dict[str, Any]
    chunks: List[Document]
    tables: List[pd.DataFrame]
    images: List[Dict[str, Any]]
    processing_timestamp: datetime
    confidence_score: float


@dataclass 
class DocumentCollectionResult:
    """Result of document collection processing."""
    processed_documents: List[ProcessedDocument]
    vector_store_collection: str
    total_chunks: int
    processing_summary: Dict[str, Any]
    audit_trail_id: str


class DocumentCollectionAgent(BaseAgent):
    """
    Document Collection Agent implementing multi-modal document processing
    with LangChain integration and Chroma vector storage.
    """
    
    def __init__(self, config: AgenticConfig, audit_manager: AuditTrailManager):
        super().__init__(
            agent_id="doc_collection_001",
            role="document_collection",
            config=config,
            audit_manager=audit_manager
        )
        
        self.doc_config = config.document_processing
        self.vector_store: Optional[Chroma] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.doc_config.chunk_size,
            chunk_overlap=self.doc_config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize local HuggingFace embeddings and Chroma vector store."""
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        # Initialize Chroma vector store
        self.vector_store = Chroma(
            collection_name=f"documents_{self.agent_id}",
            embedding_function=self.embeddings,
            persist_directory="./chroma_storage"
        )
        logger.info("Document Collection Agent initialized with HuggingFace embeddings")
    
    async def process_document_collection(
        self,
        document_paths: List[Union[str, Path]],
        collection_name: str = "financial_documents"
    ) -> DocumentCollectionResult:
        """
        Process a collection of documents with multi-modal extraction.
        
        Args:
            document_paths: List of document file paths to process
            collection_name: Name for the document collection
            
        Returns:
            DocumentCollectionResult with processed documents and metadata
        """
        audit_event = AuditEvent(
            agent_id=self.agent_id,
            action="process_document_collection",
            input_data={"paths": [str(p) for p in document_paths], "collection": collection_name},
            timestamp=datetime.now()
        )
        
        try:
            processed_documents = []
            all_chunks = []
            
            # Process each document
            for doc_path in document_paths:
                doc_path = Path(doc_path)
                logger.info(f"Processing document: {doc_path}")
                
                # Process individual document
                processed_doc = await self._process_single_document(doc_path)
                processed_documents.append(processed_doc)
                all_chunks.extend(processed_doc.chunks)
            
            # Add all chunks to vector store
            if all_chunks and self.vector_store:
                self.vector_store.add_documents(all_chunks)
                # Persist to disk for reliability
                try:
                    self.vector_store.persist()
                except Exception:
                    logger.warning("Failed to persist vector store, continuing")
                logger.info(f"Added {len(all_chunks)} chunks to vector store")
            
            # Create processing summary
            processing_summary = {
                "total_documents": len(processed_documents),
                "total_chunks": len(all_chunks),
                "processing_time": datetime.now(),
                "average_confidence": sum(d.confidence_score for d in processed_documents) / len(processed_documents) if processed_documents else 0,
                "document_types": list(set(Path(d.source_path).suffix for d in processed_documents))
            }
            
            result = DocumentCollectionResult(
                processed_documents=processed_documents,
                vector_store_collection=collection_name,
                total_chunks=len(all_chunks),
                processing_summary=processing_summary,
                audit_trail_id=audit_event.event_id
            )
            
            # Complete audit event
            audit_event.output_data = {
                "result_summary": processing_summary,
                "success": True
            }
            audit_event.completed_at = datetime.now()
            await self.audit_manager.log_event(audit_event)
            
            return result
            
        except Exception as e:
            audit_event.error = str(e)
            audit_event.completed_at = datetime.now()
            await self.audit_manager.log_event(audit_event)
            logger.error(f"Document collection processing failed: {e}")
            raise
    
    async def _process_single_document(self, doc_path: Path) -> ProcessedDocument:
        """Process a single document with multi-modal extraction."""
        
        # Generate unique document ID
        doc_id = hashlib.md5(str(doc_path).encode()).hexdigest()[:12]
        
        # Determine document type and process accordingly
        suffix = doc_path.suffix.lower()
        
        if suffix == '.pdf':
            content, tables, images = await self._process_pdf(doc_path)
        elif suffix in ['.docx', '.doc']:
            content, tables, images = await self._process_docx(doc_path)
        elif suffix == '.txt':
            content, tables, images = await self._process_text(doc_path)
        elif suffix in ['.xlsx', '.xls']:
            content, tables, images = await self._process_excel(doc_path)
        else:
            raise ValueError(f"Unsupported document format: {suffix}")
        
        # Create text chunks
        chunks = self.text_splitter.create_documents(
            texts=[content],
            metadatas=[{
                "source": str(doc_path),
                "document_id": doc_id,
                "chunk_type": "text"
            }]
        )
        
        # Add table chunks if present
        for i, table in enumerate(tables):
            table_text = table.to_string()
            table_chunks = self.text_splitter.create_documents(
                texts=[table_text],
                metadatas=[{
                    "source": str(doc_path),
                    "document_id": doc_id,
                    "chunk_type": "table",
                    "table_index": i
                }]
            )
            chunks.extend(table_chunks)
        
        # Calculate confidence score (simplified)
        confidence_score = min(1.0, len(content) / 1000) if content else 0.0
        
        return ProcessedDocument(
            document_id=doc_id,
            source_path=str(doc_path),
            content=content,
            metadata={
                "file_size": doc_path.stat().st_size,
                "file_type": suffix,
                "num_tables": len(tables),
                "num_images": len(images)
            },
            chunks=chunks,
            tables=tables,
            images=images,
            processing_timestamp=datetime.now(),
            confidence_score=confidence_score
        )
    
    async def _process_pdf(self, pdf_path: Path) -> tuple[str, List[pd.DataFrame], List[Dict[str, Any]]]:
        """Process PDF document with multi-modal extraction."""
        content_parts = []
        tables = []
        images = []
        
        try:
            if self.doc_config.pdf_parser == "pdfplumber":
                # Use pdfplumber for text and table extraction
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        # Extract text
                        page_text = page.extract_text()
                        if page_text:
                            content_parts.append(page_text)
                        
                        # Extract tables if enabled
                        if self.doc_config.extract_tables:
                            page_tables = page.extract_tables()
                            for table_data in page_tables:
                                if table_data:
                                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                    tables.append(df)
            
            # Use PyMuPDF for image extraction if enabled
            if self.doc_config.extract_images:
                doc = fitz.open(pdf_path)
                for page_num, page in enumerate(doc):
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        images.append({
                            "page": page_num,
                            "index": img_index,
                            "xref": img[0]
                        })
                doc.close()
        
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            
        content = "\n\n".join(content_parts)
        return content, tables, images
    
    async def _process_docx(self, docx_path: Path) -> tuple[str, List[pd.DataFrame], List[Dict[str, Any]]]:
        """Process DOCX document."""
        content_parts = []
        tables = []
        images = []
        
        try:
            doc = DocxDocument(docx_path)
            
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content_parts.append(paragraph.text)
            
            # Extract tables if enabled
            if self.doc_config.extract_tables:
                for table in doc.tables:
                    data = []
                    for row in table.rows:
                        row_data = [cell.text for cell in row.cells]
                        data.append(row_data)
                    
                    if data:
                        df = pd.DataFrame(data[1:], columns=data[0] if len(data) > 1 else None)
                        tables.append(df)
        
        except Exception as e:
            logger.error(f"Error processing DOCX {docx_path}: {e}")
        
        content = "\n\n".join(content_parts)
        return content, tables, images
    
    async def _process_text(self, text_path: Path) -> tuple[str, List[pd.DataFrame], List[Dict[str, Any]]]:
        """Process plain text document."""
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            logger.error(f"Error processing text file {text_path}: {e}")
            content = ""
        
        return content, [], []  # No tables or images in plain text
    
    async def _process_excel(self, excel_path: Path) -> tuple[str, List[pd.DataFrame], List[Dict[str, Any]]]:
        """Process Excel file extracting all sheets as tables."""
        tables = []
        try:
            xls = pd.ExcelFile(excel_path)
            for sheet in xls.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet)
                tables.append(df)
        except Exception as e:
            logger.error(f"Error processing Excel {excel_path}: {e}")
        # Convert tables to text content
        content = "\n\n".join(df.to_string(index=False) for df in tables)
        return content, tables, []
    
    async def search_documents(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search processed documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant document chunks
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query=query,
                k=top_k,
                filter=filter_metadata
            )
            
            logger.info(f"Found {len(results)} relevant documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the processed document collection."""
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        try:
            # Get collection info
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": collection.name,
                "embedding_dimension": "384",  # all-MiniLM-L6-v2 dimension
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the Document Collection Agent.
        
        Args:
            input_data: Should contain 'document_paths' and optional 'collection_name'
            
        Returns:
            Processing results
        """
        document_paths = input_data.get("document_paths")
        collection_name = input_data.get("collection_name", "financial_documents")
        
        if not document_paths:
            raise ValueError("document_paths is required in input_data")
        
        result = await self.process_document_collection(document_paths, collection_name)
        
        return {
            "collection_result": result,
            "success": True,
            "agent_id": self.agent_id
        }
