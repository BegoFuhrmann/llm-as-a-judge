"""
GDPR Compliance Agent for the RAG vs Agentic AI Evaluation Framework.

This module provides a specialized RAG agent for assessing GDPR compliance
across key regulatory requirements and data protection principles.
"""

from typing import Dict, List, Any, Optional
from .rag_agent import RAGAgent


class GDPRComplianceAgent(RAGAgent):
    """
    Specialized agent for GDPR compliance assessment.
    
    Focuses on evaluating compliance with key GDPR provisions:
    - Lawfulness of processing (Art. 6)
    - Data minimization (Art. 5)
    - Purpose limitation (Art. 5)
    - Right to explanation (Recital 71)
    - Data subject rights (Art. 12-22)
    - General principle for data transfers (Art. 44)
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the GDPR Compliance Agent.
        
        Args:
            db_path: Path to the GDPR knowledge base
        """
        super().__init__(
            collection_name="gdpr_2016",
            db_path=db_path,
            agent_name="GDPR Compliance Agent"
        )
        
        # GDPR-specific configuration
        self.gdpr_principles = {
            "lawfulness": "Article 6 - Lawfulness of processing",
            "data_minimization": "Article 5(1)(c) - Data minimisation",
            "purpose_limitation": "Article 5(1)(b) - Purpose limitation",
            "right_to_explanation": "Recital 71 - Right to explanation",
            "data_subject_rights": "Articles 12-22 - Data subject rights",
            "transfer_principles": "Article 44 - General principle for transfers"
        }
        
        # Assessment criteria templates
        self.assessment_templates = {
            "lawfulness": {
                "prompt": "Assess whether the described data processing has a lawful basis under Article 6 GDPR. Consider: consent, contract, legal obligation, vital interests, public task, or legitimate interests.",
                "criteria": ["legal_basis_identified", "basis_appropriate", "documented_properly"]
            },
            "data_minimization": {
                "prompt": "Evaluate compliance with data minimisation principle under Article 5(1)(c) GDPR. Consider: necessity, relevance, and limitation to purpose.",
                "criteria": ["data_necessary", "data_relevant", "limited_to_purpose"]
            },
            "purpose_limitation": {
                "prompt": "Assess compliance with purpose limitation under Article 5(1)(b) GDPR. Consider: specified purposes, explicit purposes, legitimate purposes.",
                "criteria": ["purposes_specified", "purposes_explicit", "purposes_legitimate"]
            },
            "right_to_explanation": {
                "prompt": "Evaluate compliance with right to explanation requirements under Recital 71 GDPR for automated decision-making.",
                "criteria": ["automated_decision_identified", "explanation_provided", "human_intervention_available"]
            },
            "data_subject_rights": {
                "prompt": "Assess provisions for data subject rights under Articles 12-22 GDPR. Consider: access, rectification, erasure, portability, objection.",
                "criteria": ["rights_facilitated", "procedures_clear", "response_timely"]
            },
            "transfer_principles": {
                "prompt": "Evaluate compliance with international transfer principles under Article 44 GDPR. Consider: adequacy decisions, safeguards, derogations.",
                "criteria": ["transfer_basis_identified", "safeguards_appropriate", "documented_properly"]
            }
        }
    
    def assess_gdpr_compliance(self, context: str, focus_area: str = None) -> Dict[str, Any]:
        """
        Comprehensive GDPR compliance assessment.
        
        Args:
            context: Description of the data processing scenario
            focus_area: Specific GDPR area to focus on (optional)
            
        Returns:
            Detailed compliance assessment results
        """
        if focus_area and focus_area not in self.gdpr_principles:
            return {
                "error": f"Unknown focus area: {focus_area}",
                "valid_areas": list(self.gdpr_principles.keys())
            }
        
        # Determine assessment scope
        areas_to_assess = [focus_area] if focus_area else list(self.gdpr_principles.keys())
        
        results = {
            "overall_assessment": {},
            "detailed_assessments": {},
            "recommendations": [],
            "compliance_score": 0,
            "risk_level": "unknown"
        }
        
        total_score = 0
        max_possible_score = 0
        
        for area in areas_to_assess:
            assessment = self._assess_specific_area(context, area)
            results["detailed_assessments"][area] = assessment
            
            # Calculate scores
            area_score = assessment.get("compliance_score", 0)
            total_score += area_score
            max_possible_score += 100  # Each area scored out of 100
            
            # Collect recommendations
            if assessment.get("recommendations"):
                results["recommendations"].extend(assessment["recommendations"])
        
        # Calculate overall compliance score
        if max_possible_score > 0:
            results["compliance_score"] = round((total_score / max_possible_score) * 100, 2)
        
        # Determine risk level
        results["risk_level"] = self._determine_risk_level(results["compliance_score"])
        
        # Generate overall assessment summary
        results["overall_assessment"] = self._generate_overall_summary(results)
        
        return results
    
    def _assess_specific_area(self, context: str, area: str) -> Dict[str, Any]:
        """
        Assess compliance for a specific GDPR area.
        
        Args:
            context: Data processing context
            area: GDPR area to assess
            
        Returns:
            Area-specific assessment results
        """
        template = self.assessment_templates.get(area, {})
        prompt_template = template.get("prompt", "")
        criteria = template.get("criteria", [])
        
        # Construct assessment query
        query = f"""
        {prompt_template}
        
        Context: {context}
        
        Please provide a detailed assessment covering:
        {', '.join(criteria)}
        
        Include specific recommendations for improvement if needed.
        """
        
        # Query the knowledge base
        query_result = self.query(query)
        
        # Parse the response for structured assessment
        assessment = {
            "area": area,
            "gdpr_reference": self.gdpr_principles[area],
            "assessment_criteria": criteria,
            "detailed_analysis": query_result.get("answer", "No analysis available"),
            "sources": query_result.get("sources", []),
            "compliance_score": self._calculate_area_score(query_result.get("answer", "")),
            "recommendations": self._extract_recommendations(query_result.get("answer", "")),
            "risk_indicators": self._identify_risk_indicators(query_result.get("answer", ""))
        }
        
        return assessment
    
    def _calculate_area_score(self, analysis: str) -> int:
        """
        Calculate compliance score for an area based on analysis.
        
        Args:
            analysis: The detailed analysis text
            
        Returns:
            Compliance score (0-100)
        """
        # Simple scoring based on key compliance indicators
        positive_indicators = [
            "compliant", "adequate", "appropriate", "sufficient",
            "meets requirements", "properly implemented", "well documented"
        ]
        
        negative_indicators = [
            "non-compliant", "inadequate", "insufficient", "missing",
            "unclear", "poorly documented", "lacks", "fails to"
        ]
        
        analysis_lower = analysis.lower()
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in analysis_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in analysis_lower)
        
        # Base score calculation
        if negative_count > positive_count:
            base_score = max(20, 60 - (negative_count * 15))
        elif positive_count > negative_count:
            base_score = min(90, 70 + (positive_count * 10))
        else:
            base_score = 50
        
        return base_score
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """
        Extract recommendations from analysis text.
        
        Args:
            analysis: The analysis text
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Look for common recommendation patterns
        recommendation_patterns = [
            "recommend", "should", "must", "need to", "consider",
            "improve", "implement", "ensure", "establish"
        ]
        
        sentences = analysis.split('.')
        for sentence in sentences:
            if any(pattern in sentence.lower() for pattern in recommendation_patterns):
                recommendations.append(sentence.strip())
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _identify_risk_indicators(self, analysis: str) -> List[str]:
        """
        Identify risk indicators from analysis.
        
        Args:
            analysis: The analysis text
            
        Returns:
            List of risk indicators
        """
        risk_indicators = []
        
        high_risk_terms = [
            "violation", "breach", "non-compliance", "penalty",
            "fine", "enforcement", "investigation", "legal action"
        ]
        
        medium_risk_terms = [
            "gap", "weakness", "insufficient", "unclear",
            "missing documentation", "incomplete"
        ]
        
        analysis_lower = analysis.lower()
        
        for term in high_risk_terms:
            if term in analysis_lower:
                risk_indicators.append(f"High Risk: {term} identified")
        
        for term in medium_risk_terms:
            if term in analysis_lower:
                risk_indicators.append(f"Medium Risk: {term} identified")
        
        return risk_indicators
    
    def _determine_risk_level(self, compliance_score: float) -> str:
        """
        Determine overall risk level based on compliance score.
        
        Args:
            compliance_score: Overall compliance score
            
        Returns:
            Risk level classification
        """
        if compliance_score >= 80:
            return "low"
        elif compliance_score >= 60:
            return "medium"
        elif compliance_score >= 40:
            return "high"
        else:
            return "critical"
    
    def _generate_overall_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall assessment summary.
        
        Args:
            results: Assessment results
            
        Returns:
            Overall summary
        """
        compliance_score = results["compliance_score"]
        risk_level = results["risk_level"]
        
        summary = {
            "compliance_status": self._get_compliance_status(compliance_score),
            "key_strengths": [],
            "critical_gaps": [],
            "priority_actions": results["recommendations"][:3],
            "regulatory_risk": risk_level
        }
        
        # Analyze detailed assessments for strengths and gaps
        for area, assessment in results["detailed_assessments"].items():
            area_score = assessment.get("compliance_score", 0)
            
            if area_score >= 80:
                summary["key_strengths"].append(f"{area}: {self.gdpr_principles[area]}")
            elif area_score < 50:
                summary["critical_gaps"].append(f"{area}: {self.gdpr_principles[area]}")
        
        return summary
    
    def _get_compliance_status(self, score: float) -> str:
        """Get compliance status description."""
        if score >= 80:
            return "Strong compliance - minor improvements needed"
        elif score >= 60:
            return "Moderate compliance - several areas need attention"
        elif score >= 40:
            return "Weak compliance - significant improvements required"
        else:
            return "Poor compliance - comprehensive remediation needed"
    
    def assess_lawfulness_of_processing(self, context: str) -> Dict[str, Any]:
        """Assess lawfulness of processing under Article 6 GDPR."""
        return self.assess_gdpr_compliance(context, "lawfulness")
    
    def assess_data_minimization(self, context: str) -> Dict[str, Any]:
        """Assess data minimization compliance under Article 5(1)(c) GDPR."""
        return self.assess_gdpr_compliance(context, "data_minimization")
    
    def assess_purpose_limitation(self, context: str) -> Dict[str, Any]:
        """Assess purpose limitation compliance under Article 5(1)(b) GDPR."""
        return self.assess_gdpr_compliance(context, "purpose_limitation")
    
    def assess_right_to_explanation(self, context: str) -> Dict[str, Any]:
        """Assess right to explanation compliance under Recital 71 GDPR."""
        return self.assess_gdpr_compliance(context, "right_to_explanation")
    
    def assess_data_subject_rights(self, context: str) -> Dict[str, Any]:
        """Assess data subject rights provisions under Articles 12-22 GDPR."""
        return self.assess_gdpr_compliance(context, "data_subject_rights")
    
    def assess_transfer_principles(self, context: str) -> Dict[str, Any]:
        """Assess international transfer compliance under Article 44 GDPR."""
        return self.assess_gdpr_compliance(context, "transfer_principles")
    
    def get_gdpr_guidance(self, topic: str) -> Dict[str, Any]:
        """
        Get specific GDPR guidance on a topic.
        
        Args:
            topic: The GDPR topic to get guidance on
            
        Returns:
            Guidance information from the knowledge base
        """
        query = f"Provide detailed guidance on {topic} under GDPR including requirements, best practices, and common compliance issues."
        return self.query(query)
    
    def generate_compliance_report(self, assessment_results: Dict[str, Any]) -> str:
        """
        Generate a formatted compliance report.
        
        Args:
            assessment_results: Results from assess_gdpr_compliance
            
        Returns:
            Formatted compliance report
        """
        report = []
        report.append("# GDPR COMPLIANCE ASSESSMENT REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Overall summary
        overall = assessment_results.get("overall_assessment", {})
        report.append(f"**Overall Compliance Score:** {assessment_results.get('compliance_score', 0)}%")
        report.append(f"**Risk Level:** {assessment_results.get('risk_level', 'unknown').upper()}")
        report.append(f"**Status:** {overall.get('compliance_status', 'Unknown')}")
        report.append("")
        
        # Key strengths
        if overall.get("key_strengths"):
            report.append("## Key Strengths")
            for strength in overall["key_strengths"]:
                report.append(f"- {strength}")
            report.append("")
        
        # Critical gaps
        if overall.get("critical_gaps"):
            report.append("## Critical Gaps")
            for gap in overall["critical_gaps"]:
                report.append(f"- {gap}")
            report.append("")
        
        # Detailed assessments
        report.append("## Detailed Assessments")
        for area, assessment in assessment_results.get("detailed_assessments", {}).items():
            report.append(f"### {area.replace('_', ' ').title()}")
            report.append(f"**Reference:** {assessment.get('gdpr_reference', 'N/A')}")
            report.append(f"**Score:** {assessment.get('compliance_score', 0)}%")
            
            if assessment.get("risk_indicators"):
                report.append("**Risk Indicators:**")
                for indicator in assessment["risk_indicators"]:
                    report.append(f"- {indicator}")
            
            report.append("")
        
        # Recommendations
        if assessment_results.get("recommendations"):
            report.append("## Priority Recommendations")
            for i, rec in enumerate(assessment_results["recommendations"][:5], 1):
                report.append(f"{i}. {rec}")
        
        return "\n".join(report)
