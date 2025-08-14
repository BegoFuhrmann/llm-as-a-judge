"""
EU AI Act Compliance Agent for the RAG vs Agentic AI Evaluation Framework.

This module provides a specialized RAG agent for assessing compliance with the EU AI Act,
focusing on risk categorization, prohibited practices, and requirements for high-risk systems.
"""

from typing import Dict, List, Any, Optional
from .rag_agent import RAGAgent


class EUAIActAgent(RAGAgent):
    """
    Specialized agent for EU AI Act compliance assessment.
    
    Focuses on evaluating compliance with key EU AI Act provisions:
    - Risk categorization (Art. 6-15)
    - Prohibited practices (Art. 5)
    - High-risk system requirements (Art. 16-51)
    - Transparency obligations (Art. 52)
    - CE marking requirements (Art. 48)
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the EU AI Act Compliance Agent.
        
        Args:
            db_path: Path to the EU AI Act knowledge base
        """
        super().__init__(
            collection_name="eu_ai_act_compliance",
            db_path=db_path,
            agent_name="EU AI Act Compliance Agent"
        )
        
        # EU AI Act-specific configuration
        self.eu_ai_act_articles = {
            "risk_categorization": "Articles 6-15 - Risk Categorization",
            "prohibited_practices": "Article 5 - Prohibited AI Practices",
            "high_risk_requirements": "Articles 16-51 - Requirements for High-Risk AI Systems",
            "transparency_obligations": "Article 52 - Transparency Obligations",
            "ce_marking": "Article 48 - CE Marking"
        }
        
        # Assessment criteria templates
        self.assessment_templates = {
            "risk_categorization": {
                "prompt": "Assess the AI system's risk categorization under Articles 6-15 of the EU AI Act. Consider if it falls into prohibited, high-risk, limited-risk, or minimal-risk categories.",
                "criteria": ["risk_level_identified", "categorization_justified", "annexes_checked"]
            },
            "prohibited_practices": {
                "prompt": "Evaluate compliance with the list of prohibited AI practices under Article 5 of the EU AI Act. Check for manipulation, exploitation of vulnerabilities, or social scoring.",
                "criteria": ["no_manipulation", "no_exploitation", "no_social_scoring"]
            },
            "high_risk_requirements": {
                "prompt": "Assess compliance with requirements for high-risk AI systems under Articles 16-51. Consider risk management, data governance, technical documentation, record-keeping, transparency, human oversight, and cybersecurity.",
                "criteria": ["risk_management_system", "data_governance", "technical_documentation", "record_keeping", "transparency_and_information", "human_oversight", "accuracy_robustness_cybersecurity"]
            },
            "transparency_obligations": {
                "prompt": "Evaluate compliance with transparency obligations under Article 52 for systems that interact with humans, or generate synthetic content.",
                "criteria": ["is_ai_interaction_disclosed", "is_deep_fake_disclosed", "is_synthetic_content_disclosed"]
            },
            "ce_marking": {
                "prompt": "Assess compliance with CE marking requirements under Article 48 for high-risk AI systems, including conformity assessment and declaration of conformity.",
                "criteria": ["conformity_assessment_done", "declaration_of_conformity_drawn_up", "ce_marking_affixed"]
            }
        }
    
    def assess_eu_ai_act_compliance(self, context: str, focus_area: str = None) -> Dict[str, Any]:
        """
        Comprehensive EU AI Act compliance assessment.
        
        Args:
            context: Description of the AI system and its use case
            focus_area: Specific EU AI Act area to focus on (optional)
            
        Returns:
            Detailed compliance assessment results
        """
        if focus_area and focus_area not in self.eu_ai_act_articles:
            return {
                "error": f"Unknown focus area: {focus_area}",
                "valid_areas": list(self.eu_ai_act_articles.keys())
            }
        
        areas_to_assess = [focus_area] if focus_area else list(self.eu_ai_act_articles.keys())
        
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
            
            area_score = assessment.get("compliance_score", 0)
            total_score += area_score
            max_possible_score += 100
            
            if assessment.get("recommendations"):
                results["recommendations"].extend(assessment["recommendations"])
        
        if max_possible_score > 0:
            results["compliance_score"] = round((total_score / max_possible_score) * 100, 2)
        
        results["risk_level"] = self._determine_risk_level(results["compliance_score"])
        results["overall_assessment"] = self._generate_overall_summary(results)
        
        return results
    
    def _assess_specific_area(self, context: str, area: str) -> Dict[str, Any]:
        """Assess compliance for a specific EU AI Act area."""
        template = self.assessment_templates.get(area, {})
        prompt_template = template.get("prompt", "")
        criteria = template.get("criteria", [])
        
        query = f"""
        {prompt_template}
        
        Context: {context}
        
        Please provide a detailed assessment covering:
        {', '.join(criteria)}
        
        Include specific recommendations for improvement if needed.
        """
        
        query_result = self.query(query)
        
        analysis = query_result.get("answer", "No analysis available")
        assessment = {
            "area": area,
            "eu_ai_act_reference": self.eu_ai_act_articles[area],
            "assessment_criteria": criteria,
            "detailed_analysis": analysis,
            "sources": query_result.get("sources", []),
            "compliance_score": self._calculate_area_score(analysis),
            "recommendations": self._extract_recommendations(analysis),
            "risk_indicators": self._identify_risk_indicators(analysis)
        }
        
        return assessment
    
    def _calculate_area_score(self, analysis: str) -> int:
        """Calculate compliance score for an area based on analysis."""
        positive_indicators = ["compliant", "adequate", "appropriate", "sufficient", "meets requirements"]
        negative_indicators = ["non-compliant", "inadequate", "insufficient", "missing", "unclear", "lacks"]
        
        analysis_lower = analysis.lower()
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in analysis_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in analysis_lower)
        
        if negative_count > positive_count:
            return max(20, 60 - (negative_count * 15))
        elif positive_count > negative_count:
            return min(90, 70 + (positive_count * 10))
        else:
            return 50
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract recommendations from analysis text."""
        recommendations = []
        patterns = ["recommend", "should", "must", "need to", "consider", "improve", "implement", "ensure"]
        sentences = analysis.split('.')
        for sentence in sentences:
            if any(pattern in sentence.lower() for pattern in patterns):
                recommendations.append(sentence.strip())
        return recommendations[:5]
    
    def _identify_risk_indicators(self, analysis: str) -> List[str]:
        """Identify risk indicators from analysis."""
        risk_indicators = []
        high_risk_terms = ["violation", "breach", "non-compliance", "penalty", "fine"]
        medium_risk_terms = ["gap", "weakness", "insufficient", "unclear", "incomplete"]
        
        analysis_lower = analysis.lower()
        
        for term in high_risk_terms:
            if term in analysis_lower:
                risk_indicators.append(f"High Risk: {term} identified")
        for term in medium_risk_terms:
            if term in analysis_lower:
                risk_indicators.append(f"Medium Risk: {term} identified")
        
        return risk_indicators

    def _determine_risk_level(self, compliance_score: float) -> str:
        """Determine overall risk level based on compliance score."""
        if compliance_score >= 80:
            return "low"
        elif compliance_score >= 60:
            return "medium"
        elif compliance_score >= 40:
            return "high"
        else:
            return "critical"

    def _generate_overall_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment summary."""
        compliance_score = results["compliance_score"]
        risk_level = results["risk_level"]
        
        summary = {
            "compliance_status": self._get_compliance_status(compliance_score),
            "key_strengths": [],
            "critical_gaps": [],
            "priority_actions": results["recommendations"][:3],
            "regulatory_risk": risk_level
        }
        
        for area, assessment in results["detailed_assessments"].items():
            area_score = assessment.get("compliance_score", 0)
            if area_score >= 80:
                summary["key_strengths"].append(f"{area}: {self.eu_ai_act_articles[area]}")
            elif area_score < 50:
                summary["critical_gaps"].append(f"{area}: {self.eu_ai_act_articles[area]}")
        
        return summary

    def _get_compliance_status(self, score: float) -> str:
        """Get compliance status description."""
        if score >= 80:
            return "Strong compliance"
        elif score >= 60:
            return "Moderate compliance"
        elif score >= 40:
            return "Weak compliance"
        else:
            return "Poor compliance"

    def assess_risk_categorization(self, context: str) -> Dict[str, Any]:
        """Assess risk categorization under Articles 6-15."""
        return self.assess_eu_ai_act_compliance(context, "risk_categorization")

    def assess_prohibited_practices(self, context: str) -> Dict[str, Any]:
        """Assess prohibited practices under Article 5."""
        return self.assess_eu_ai_act_compliance(context, "prohibited_practices")

    def assess_high_risk_requirements(self, context: str) -> Dict[str, Any]:
        """Assess requirements for high-risk systems under Articles 16-51."""
        return self.assess_eu_ai_act_compliance(context, "high_risk_requirements")

    def assess_transparency_obligations(self, context: str) -> Dict[str, Any]:
        """Assess transparency obligations under Article 52."""
        return self.assess_eu_ai_act_compliance(context, "transparency_obligations")

    def assess_ce_marking_requirements(self, context: str) -> Dict[str, Any]:
        """Assess CE marking requirements under Article 48."""
        return self.assess_eu_ai_act_compliance(context, "ce_marking")

    def get_ai_act_guidance(self, topic: str) -> Dict[str, Any]:
        """Get specific EU AI Act guidance on a topic."""
        query = f"Provide detailed guidance on {topic} under the EU AI Act, including requirements, best practices, and common compliance issues."
        return self.query(query)

    def generate_compliance_report(self, assessment_results: Dict[str, Any]) -> str:
        """Generate a formatted compliance report."""
        report = ["# EU AI ACT COMPLIANCE ASSESSMENT REPORT", "="*50, ""]
        
        overall = assessment_results.get("overall_assessment", {})
        report.append(f"**Overall Compliance Score:** {assessment_results.get('compliance_score', 0)}%")
        report.append(f"**Risk Level:** {assessment_results.get('risk_level', 'unknown').upper()}")
        report.append(f"**Status:** {overall.get('compliance_status', 'Unknown')}")
        report.append("")
        
        if overall.get("key_strengths"):
            report.append("## Key Strengths")
            for strength in overall["key_strengths"]:
                report.append(f"- {strength}")
            report.append("")
        
        if overall.get("critical_gaps"):
            report.append("## Critical Gaps")
            for gap in overall["critical_gaps"]:
                report.append(f"- {gap}")
            report.append("")
        
        report.append("## Detailed Assessments")
        for area, assessment in assessment_results.get("detailed_assessments", {}).items():
            report.append(f"### {area.replace('_', ' ').title()}")
            report.append(f"**Reference:** {assessment.get('eu_ai_act_reference', 'N/A')}")
            report.append(f"**Score:** {assessment.get('compliance_score', 0)}%")
            if assessment.get("risk_indicators"):
                report.append("**Risk Indicators:**")
                for indicator in assessment["risk_indicators"]:
                    report.append(f"- {indicator}")
            report.append("")
        
        if assessment_results.get("recommendations"):
            report.append("## Priority Recommendations")
            for i, rec in enumerate(assessment_results["recommendations"][:5], 1):
                report.append(f"{i}. {rec}")
        
        return "\n".join(report)
