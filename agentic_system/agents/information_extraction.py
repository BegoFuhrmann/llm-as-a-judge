"""
Information Extraction Agent for the Agentic System.

This agent handles domain-specific entity recognition and relationship mapping
using Azure OpenAI for advanced NLP capabilities.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Azure OpenAI integration
from openai import AsyncAzureOpenAI

# Local imports
from ..core.models import (
    AgentMessage, ExtractionResult, ExtractedEntity, ExtractedRelationship,
    AgentType, MessageType, ExtractionStatus, AuditEntry
)
from ..core.agentic_config import config, logger


class InformationExtractionAgent:
    """
    Agent responsible for domain-specific information extraction.
    
    Capabilities:
    - Named Entity Recognition (NER)
    - Relationship extraction between entities
    - Domain-specific pattern recognition
    - Confidence scoring and validation
    """
    
    def __init__(self):
        """Initialize the Information Extraction Agent."""
        self.agent_type = AgentType.INFORMATION_EXTRACTION
        self.config = config
        self.logger = logger.bind(agent=self.agent_type.value)
        
        # Initialize Azure OpenAI client
        self.openai_client = AsyncAzureOpenAI(
            api_key=self.config.azure_openai.api_key,
            api_version=self.config.azure_openai.api_version,
            azure_endpoint=self.config.azure_openai.endpoint
        )
        
        # Financial domain entities and relationships
        self.financial_entities = {
            "FINANCIAL_INSTRUMENT": ["bond", "stock", "derivative", "option", "future", "swap"],
            "REGULATION": ["GDPR", "MiFID", "Basel", "Solvency", "EMIR", "PCI DSS"],
            "ORGANIZATION": ["bank", "institution", "authority", "regulator", "ECB", "EBA"],
            "CURRENCY": ["EUR", "USD", "GBP", "CHF", "JPY"],
            "AMOUNT": r"\d+(?:,\d{3})*(?:\.\d{2})?",
            "DATE": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}",
            "PERCENTAGE": r"\d+(?:\.\d+)?%"
        }
        
        self.relationship_types = [
            "REGULATORY_COMPLIANCE",
            "FINANCIAL_EXPOSURE",
            "COUNTERPARTY_RISK",
            "REPORTING_OBLIGATION",
            "CAPITAL_REQUIREMENT",
            "OPERATIONAL_DEPENDENCY"
        ]
        
        self.logger.info("Information Extraction Agent initialized")
    
    async def extract_information(self, text: str, document_id: str) -> ExtractionResult:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Text content to analyze
            document_id: Unique document identifier
            
        Returns:
            ExtractionResult with extracted entities and relationships
        """
        start_time = datetime.now()
        
        try:
            # Initialize result
            result = ExtractionResult(
                document_id=document_id,
                extraction_status=ExtractionStatus.IN_PROGRESS
            )
            
            # Extract entities using multiple methods
            entities = await self._extract_entities_hybrid(text)
            result.entities = entities
            
            # Extract relationships between entities
            relationships = await self._extract_relationships(text, entities)
            result.relationships = relationships
            
            # Generate summary and key findings
            summary = await self._generate_summary(text, entities, relationships)
            result.summary = summary["summary"]
            result.key_findings = summary["key_findings"]
            
            # Calculate confidence score
            result.confidence_score = self._calculate_confidence(entities, relationships)
            
            # Calculate processing time
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.extraction_status = ExtractionStatus.COMPLETED
            
            # Log audit entry
            await self._log_audit_entry("information_extracted", {
                "document_id": document_id,
                "entities_count": len(entities),
                "relationships_count": len(relationships),
                "confidence_score": result.confidence_score,
                "processing_time": result.processing_time
            })
            
            self.logger.info("Information extraction completed",
                           document_id=document_id,
                           entities_count=len(entities),
                           relationships_count=len(relationships),
                           confidence=result.confidence_score)
            
            return result
            
        except Exception as e:
            result = ExtractionResult(
                document_id=document_id,
                extraction_status=ExtractionStatus.FAILED,
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            await self._log_audit_entry("extraction_failed", {
                "document_id": document_id,
                "error": str(e)
            })
            
            self.logger.error("Information extraction failed",
                            document_id=document_id,
                            error=str(e))
            
            return result
    
    async def _extract_entities_hybrid(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using both rule-based and LLM-based methods."""
        entities = []
        
        # Rule-based extraction for well-defined patterns
        rule_based_entities = self._extract_entities_rules(text)
        entities.extend(rule_based_entities)
        
        # LLM-based extraction for complex entities
        llm_entities = await self._extract_entities_llm(text)
        entities.extend(llm_entities)
        
        # Deduplicate and merge entities
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _extract_entities_rules(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using rule-based pattern matching."""
        entities = []
        
        # Extract pattern-based entities
        for entity_type, pattern in self.financial_entities.items():
            if isinstance(pattern, str):  # Regex pattern
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append(ExtractedEntity(
                        entity_type=entity_type,
                        text=match.group(),
                        confidence=0.9,  # High confidence for rule-based
                        start_position=match.start(),
                        end_position=match.end(),
                        context=self._extract_context(text, match.start(), match.end())
                    ))
            elif isinstance(pattern, list):  # Keyword list
                for keyword in pattern:
                    matches = re.finditer(rf'\b{re.escape(keyword)}\b', text, re.IGNORECASE)
                    for match in matches:
                        entities.append(ExtractedEntity(
                            entity_type=entity_type,
                            text=match.group(),
                            confidence=0.85,
                            start_position=match.start(),
                            end_position=match.end(),
                            context=self._extract_context(text, match.start(), match.end())
                        ))
        
        return entities
    
    async def _extract_entities_llm(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using Azure OpenAI LLM."""
        try:
            prompt = f"""
            Extract financial and regulatory entities from the following text. 
            Focus on:
            - Financial instruments and products
            - Regulatory frameworks and compliance requirements
            - Organizations and institutions
            - Risk factors and exposures
            - Compliance obligations
            
            Return a JSON array with entities in this format:
            [{{"entity_type": "TYPE", "text": "entity text", "confidence": 0.8, "context": "surrounding context"}}]
            
            Text: {text[:2000]}  # Limit to avoid token limits
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.azure_openai.chat_model,
                messages=[
                    {"role": "system", "content": "You are a financial regulatory expert specializing in entity extraction."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.azure_openai.temperature,
                max_tokens=1500
            )
            
            # Parse JSON response
            entities_data = json.loads(response.choices[0].message.content)
            
            entities = []
            for entity_data in entities_data:
                # Find entity position in text
                entity_text = entity_data["text"]
                start_pos = text.lower().find(entity_text.lower())
                
                if start_pos != -1:
                    entities.append(ExtractedEntity(
                        entity_type=entity_data["entity_type"],
                        text=entity_text,
                        confidence=entity_data.get("confidence", 0.7),
                        start_position=start_pos,
                        end_position=start_pos + len(entity_text),
                        context=entity_data.get("context", "")
                    ))
            
            return entities
            
        except Exception as e:
            self.logger.error("LLM entity extraction failed", error=str(e))
            return []
    
    async def _extract_relationships(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Extract relationships between identified entities."""
        if len(entities) < 2:
            return []
        
        try:
            # Create entity map for reference
            entity_texts = [entity.text for entity in entities]
            
            prompt = f"""
            Analyze the relationships between these financial entities in the given text:
            Entities: {', '.join(entity_texts)}
            
            Text: {text[:1500]}
            
            Identify relationships such as:
            - Regulatory compliance requirements
            - Financial exposures and dependencies
            - Risk relationships
            - Reporting obligations
            
            Return a JSON array with relationships:
            [{{"source_entity": "entity1", "target_entity": "entity2", "relationship_type": "TYPE", "confidence": 0.8, "context": "explanation"}}]
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.azure_openai.chat_model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert in regulatory relationships."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.azure_openai.temperature,
                max_tokens=1000
            )
            
            # Parse JSON response
            relationships_data = json.loads(response.choices[0].message.content)
            
            relationships = []
            for rel_data in relationships_data:
                relationships.append(ExtractedRelationship(
                    source_entity=rel_data["source_entity"],
                    target_entity=rel_data["target_entity"],
                    relationship_type=rel_data["relationship_type"],
                    confidence=rel_data.get("confidence", 0.7),
                    context=rel_data.get("context", "")
                ))
            
            return relationships
            
        except Exception as e:
            self.logger.error("Relationship extraction failed", error=str(e))
            return []
    
    async def _generate_summary(self, text: str, entities: List[ExtractedEntity], 
                              relationships: List[ExtractedRelationship]) -> Dict[str, Any]:
        """Generate summary and key findings from extracted information."""
        try:
            entity_summary = {entity_type: [] for entity_type in self.financial_entities.keys()}
            for entity in entities:
                if entity.entity_type in entity_summary:
                    entity_summary[entity.entity_type].append(entity.text)
            
            prompt = f"""
            Based on the extracted entities and relationships, provide:
            1. A concise summary of the document's regulatory and financial content
            2. Key findings and important insights
            
            Entities found: {json.dumps(entity_summary, indent=2)}
            Relationships: {len(relationships)} relationships identified
            
            Text excerpt: {text[:1000]}
            
            Return JSON format:
            {{"summary": "document summary", "key_findings": ["finding1", "finding2", ...]}}
            """
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.azure_openai.chat_model,
                messages=[
                    {"role": "system", "content": "You are a financial regulatory analyst providing executive summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.azure_openai.temperature,
                max_tokens=800
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error("Summary generation failed", error=str(e))
            return {"summary": "Summary generation failed", "key_findings": []}
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around an entity."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities based on text and position overlap."""
        unique_entities = []
        
        for entity in entities:
            is_duplicate = False
            for existing in unique_entities:
                # Check for text overlap
                if (entity.text.lower() == existing.text.lower() or
                    self._positions_overlap(entity, existing)):
                    # Keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        unique_entities.remove(existing)
                        unique_entities.append(entity)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_entities.append(entity)
        
        return unique_entities
    
    def _positions_overlap(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> bool:
        """Check if two entities have overlapping positions."""
        return not (entity1.end_position <= entity2.start_position or 
                   entity2.end_position <= entity1.start_position)
    
    def _calculate_confidence(self, entities: List[ExtractedEntity], 
                            relationships: List[ExtractedRelationship]) -> float:
        """Calculate overall confidence score for extraction results."""
        if not entities:
            return 0.0
        
        entity_confidence = sum(entity.confidence for entity in entities) / len(entities)
        
        if not relationships:
            return entity_confidence * 0.8  # Penalize lack of relationships
        
        relationship_confidence = sum(rel.confidence for rel in relationships) / len(relationships)
        
        # Weighted average
        return (entity_confidence * 0.6 + relationship_confidence * 0.4)
    
    async def _log_audit_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Log audit entry for extraction actions."""
        if not self.config.agent_coordination.audit_enabled:
            return
        
        audit_entry = AuditEntry(
            entry_id=f"extract_{datetime.now().timestamp()}",
            agent_type=self.agent_type,
            action=action,
            details=details
        )
        
        try:
            with open(self.config.agent_coordination.audit_log_path, 'a') as f:
                f.write(audit_entry.to_json() + "\n")
        except Exception as e:
            self.logger.error("Failed to write audit entry", error=str(e))
    
    async def handle_message(self, message: AgentMessage) -> AgentMessage:
        """Handle incoming messages from other agents."""
        try:
            if message.message_type == MessageType.REQUEST:
                content = message.content
                
                if content.get("action") == "extract_information":
                    text = content["text"]
                    document_id = content["document_id"]
                    
                    result = await self.extract_information(text, document_id)
                    
                    return AgentMessage(
                        sender=self.agent_type,
                        recipient=message.sender,
                        message_type=MessageType.RESPONSE,
                        content={
                            "status": "success",
                            "extraction_result": result.to_dict()
                        },
                        correlation_id=message.message_id
                    )
            
            # Unknown action
            return AgentMessage(
                sender=self.agent_type,
                recipient=message.sender,
                message_type=MessageType.ERROR,
                content={
                    "error": f"Unknown action: {message.content.get('action', 'none')}"
                },
                correlation_id=message.message_id
            )
            
        except Exception as e:
            self.logger.error("Message handling failed", error=str(e))
            return AgentMessage(
                sender=self.agent_type,
                recipient=message.sender,
                message_type=MessageType.ERROR,
                content={
                    "error": str(e)
                },
                correlation_id=message.message_id
            )
