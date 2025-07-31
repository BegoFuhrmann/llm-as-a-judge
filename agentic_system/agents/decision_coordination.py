"""
Decision Coordination Agent
===========================

The third agent in the multi-agent system responsible for:
- Orchestrating the Document Collection and Information Extraction agents
- Translating extracted information into high-level decisions
- Maintaining a comprehensive audit trail for all decision steps
- Reporting final recommendations or actions
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from ..core.agentic_config import AgenticConfig
from ..core.base_agent import BaseAgent
from ..core.audit_trail import AuditEvent, AuditTrailManager, AuditEventType
from .document_collection import DocumentCollectionAgent, DocumentCollectionResult
from .information_extraction import InformationExtractionAgent, ExtractionResult

logger = logging.getLogger(__name__)


class DecisionCoordinatorAgent(BaseAgent):
    """
    Agent to coordinate multi-agent workflow and make high-level decisions.
    """
    def __init__(
        self,
        config: AgenticConfig,
        audit_manager: AuditTrailManager
    ):
        super().__init__(
            agent_id="decision_coordination_001",
            role="decision_coordination",
            config=config,
            audit_manager=audit_manager
        )

        # Instantiate sub-agents
        self.document_agent = DocumentCollectionAgent(config, audit_manager)
        self.extraction_agent = InformationExtractionAgent(config, audit_manager)

    async def coordinate(
        self,
        document_paths: List[Union[str, bytes]],
        collection_name: str = "default_collection"
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline: collection, extraction, decision
        """
        audit_event = AuditEvent(
            event_id="",
            agent_id=self.agent_id,
            event_type=AuditEventType.AGENT_ACTION,
            action="coordinate_pipeline",
            input_data={
                "document_paths": [str(p) for p in document_paths],
                "collection_name": collection_name
            },
            timestamp=datetime.now()
        )
        try:
            self.update_status("processing", "coordinate_pipeline")
            # Phase 1: Document Collection
            collection_result: DocumentCollectionResult = await self.document_agent.process_document_collection(
                document_paths=document_paths,
                collection_name=collection_name
            )

            # Phase 2: Information Extraction
            extraction_result: ExtractionResult = await self.extraction_agent.extract_information(
                collection_result=collection_result,
                extraction_focus="comprehensive financial analysis"
            )

            # Phase 3: Decision Making
            decision_output = self._make_decision(extraction_result)

            # Log metrics
            await self.log_performance_metric("total_documents", len(collection_result.processed_documents))
            await self.log_performance_metric("entities_extracted", len(extraction_result.extracted_info.entities))
            await self.log_performance_metric("decision_confidence", decision_output.get("confidence", 0.0))

            # Complete audit event
            audit_event.output_data = {
                "collection_summary": collection_result.processing_summary,
                "extraction_summary": {
                    "entities": len(extraction_result.extracted_info.entities),
                    "financial_metrics": len(extraction_result.extracted_info.financial_metrics)
                },
                "decision": decision_output
            }
            audit_event.event_type = AuditEventType.DECISION_MADE
            audit_event.completed_at = datetime.now()
            await self.audit_manager.log_event(audit_event)

            self.update_status("completed", None)
            return {
                "decision": decision_output,
                "audit_trail_id": audit_event.event_id,
                "success": True
            }

        except Exception as e:
            audit_event.error = str(e)
            audit_event.completed_at = datetime.now()
            await self.audit_manager.log_event(audit_event)
            self.update_status("error", None)
            logger.error(f"Pipeline coordination failed: {e}")
            raise

    def _make_decision(self, extraction_result: ExtractionResult) -> Dict[str, Any]:
        """
        Create a high-level decision based on extracted information.
        This is a placeholder; real logic would involve more complex analysis.
        """
        info = extraction_result.extracted_info
        # Example: summarize key insights and risk indicators
        summary_insights = info.key_insights[:3]
        summary_risks = info.risk_indicators[:3]

        # Confidence aggregated from extraction and entity confidence
        avg_entity_conf = (
            sum(e.confidence for e in info.entities) / len(info.entities)
            if info.entities else 0.0
        )
        overall_conf = min(info.confidence_score, avg_entity_conf)

        decision = {
            "recommendation": (
                "Proceed with investment" if overall_conf > 0.7 and not summary_risks else
                "Review risk factors before proceeding"
            ),
            "insights": summary_insights,
            "risks": summary_risks,
            "confidence": round(overall_conf, 2)
        }
        return decision

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing entrypoint for coordinator agent.
        input_data should contain 'document_paths' and optional 'collection_name'.
        """
        paths = input_data.get("document_paths")
        name = input_data.get("collection_name", "default_collection")
        if not paths:
            raise ValueError("document_paths is required for coordination")
        return await self.coordinate(paths, name)
