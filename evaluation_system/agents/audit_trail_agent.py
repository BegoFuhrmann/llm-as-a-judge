"""
Audit Trail Agent for the RAG vs Agentic AI Evaluation Framework.

This module provides a specialized agent for evaluating audit trails based on
completeness, quality, and compliance with best practices.
"""

import os
import json
from typing import Dict, List, Any, Optional
from .rag_agent import RAGAgent


class AuditTrailAgent(RAGAgent):
    """
    Specialized agent for audit trail evaluation.
    
    Focuses on assessing audit logs for:
    - Completeness of event logging
    - Quality and consistency of log data
    - Adherence to audit trail best practices
    """
    
    def __init__(self, audit_log_path: str, agent_name: str = "Audit Trail Agent"):
        """
        Initialize the Audit Trail Agent.
        
        Args:
            audit_log_path: Path to the directory containing audit log files.
            agent_name: Human-readable name for the agent.
        """
        super().__init__(agent_name=agent_name)
        self.audit_log_path = audit_log_path
        
        # Assessment areas and templates
        self.assessment_areas = {
            "completeness": "Completeness of Audit Trail",
            "quality": "Quality and Consistency of Log Data"
        }
        
        self.assessment_templates = {
            "completeness": {
                "prompt": "Assess the completeness of the audit trail. Check if all critical events are logged, if there are any time gaps, and if essential data fields (timestamp, user, action, resource, status) are present for each event.",
                "criteria": ["all_events_logged", "no_time_gaps", "essential_fields_present"]
            },
            "quality": {
                "prompt": "Evaluate the quality and consistency of the audit log data. Assess if the log format is consistent, if messages are clear and detailed enough to reconstruct events, and if there are excessive noise or irrelevant entries.",
                "criteria": ["consistent_format", "clear_messages", "sufficient_detail", "low_noise"]
            }
        }

    def initialize(self) -> bool:
        """Initialize the agent's LLM component."""
        if not self.is_initialized:
            try:
                self._initialize_llm()
                self.is_initialized = True
                print(f"{self.agent_name} initialized successfully.")
            except Exception as e:
                print(f"Error initializing {self.agent_name}: {str(e)}")
                self.is_initialized = False
        return self.is_initialized

    def _load_audit_logs(self) -> str:
        """Load and concatenate audit logs from the specified path."""
        log_content = []
        if not os.path.exists(self.audit_log_path):
            return "Error: Audit log path does not exist."

        for filename in os.listdir(self.audit_log_path):
            file_path = os.path.join(self.audit_log_path, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        log_content.append(f"--- Log file: {filename} ---\n")
                        log_content.append(f.read())
                        log_content.append("\n\n")
                except Exception as e:
                    log_content.append(f"--- Error reading {filename}: {e} ---\n")
        
        return "".join(log_content)

    def _llm_query(self, query: str) -> str:
        """Sends a direct query to the LLM and returns the response content."""
        if not self.is_initialized:
            self.initialize()
        
        if not self.llm:
            return "LLM not initialized."

        try:
            response = self.llm.invoke(query)
            return response.content
        except Exception as e:
            return f"Error querying LLM: {str(e)}"

    def assess_audit_trail(self, focus_area: str = None) -> Dict[str, Any]:
        """
        Comprehensive audit trail assessment.
        
        Args:
            focus_area: Specific area to focus on (completeness, quality, security).
            
        Returns:
            Detailed audit trail assessment results.
        """
        logs = self._load_audit_logs()
        if logs.startswith("Error:"):
            return {"error": logs}

        areas_to_assess = [focus_area] if focus_area and focus_area in self.assessment_areas else list(self.assessment_areas.keys())
        
        results = {
            "overall_assessment": {},
            "detailed_assessments": {},
            "recommendations": [],
            "overall_score": 0
        }
        
        total_score = 0
        for area in areas_to_assess:
            assessment = self._assess_specific_area(logs, area)
            results["detailed_assessments"][area] = assessment
            total_score += assessment.get("score", 0)
            if assessment.get("recommendations"):
                results["recommendations"].extend(assessment["recommendations"])

        if areas_to_assess:
            results["overall_score"] = round(total_score / len(areas_to_assess), 2)
        
        results["overall_assessment"] = self._generate_overall_summary(results)
        
        return results

    def _assess_specific_area(self, logs: str, area: str) -> Dict[str, Any]:
        """Assess a specific area of the audit trail."""
        template = self.assessment_templates[area]
        prompt = f"""
        {template['prompt']}
        
        Audit Log Content:
        ```
        {logs[:8000]} 
        ```
        
        Based on the log content, provide a detailed analysis for the '{area}' assessment area.
        Evaluate against these criteria: {', '.join(template['criteria'])}.
        Provide a score from 0 to 100 for this area and list specific recommendations for improvement.
        Format the output as a JSON object with keys: "analysis", "score", and "recommendations".
        """
        
        response_str = self._llm_query(prompt)
        
        try:
            # Clean the response string to ensure it's valid JSON
            if response_str.startswith("```json"):
                response_str = response_str[7:]
            if response_str.endswith("```"):
                response_str = response_str[:-3]
            
            response_json = json.loads(response_str)
            
            return {
                "area": area,
                "assessment_criteria": template['criteria'],
                "analysis": response_json.get("analysis", "No analysis available."),
                "score": response_json.get("score", 0),
                "recommendations": response_json.get("recommendations", [])
            }
        except (json.JSONDecodeError, TypeError):
            return {
                "area": area,
                "assessment_criteria": template['criteria'],
                "analysis": "Failed to parse LLM response.",
                "score": 0,
                "recommendations": ["Fix LLM response format to be valid JSON."]
            }

    def _generate_overall_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an overall summary of the audit trail assessment."""
        overall_score = results["overall_score"]
        
        summary = {
            "summary_statement": self._get_summary_statement(overall_score),
            "key_strengths": [],
            "critical_weaknesses": [],
            "priority_actions": results["recommendations"][:3]
        }
        
        for area, assessment in results["detailed_assessments"].items():
            if assessment.get("score", 0) >= 80:
                summary["key_strengths"].append(f"{area.title()}: Strong performance.")
            elif assessment.get("score", 0) < 50:
                summary["critical_weaknesses"].append(f"{area.title()}: Needs significant improvement.")
        
        return summary

    def _get_summary_statement(self, score: float) -> str:
        """Get a qualitative summary statement based on the overall score."""
        if score >= 80:
            return "The audit trail is comprehensive and high-quality, with minor areas for improvement."
        elif score >= 60:
            return "The audit trail is adequate but has several areas that require attention to meet best practices."
        elif score >= 40:
            return "The audit trail has significant gaps and quality issues that need to be addressed."
        else:
            return "The audit trail is critically insufficient and requires a comprehensive overhaul."

    def generate_assessment_report(self, assessment_results: Dict[str, Any]) -> str:
        """Generate a formatted text report from assessment results."""
        report = ["# AUDIT TRAIL ASSESSMENT REPORT", "="*50, ""]
        
        overall = assessment_results.get("overall_assessment", {})
        report.append(f"**Overall Score:** {assessment_results.get('overall_score', 0)}/100")
        report.append(f"**Summary:** {overall.get('summary_statement', 'N/A')}")
        report.append("")
        
        if overall.get("key_strengths"):
            report.append("## Key Strengths")
            for strength in overall["key_strengths"]:
                report.append(f"- {strength}")
            report.append("")
        
        if overall.get("critical_weaknesses"):
            report.append("## Critical Weaknesses")
            for weakness in overall["critical_weaknesses"]:
                report.append(f"- {weakness}")
            report.append("")
        
        report.append("## Detailed Assessments")
        for area, assessment in assessment_results.get("detailed_assessments", {}).items():
            report.append(f"### {area.title()} (Score: {assessment.get('score', 0)}/100)")
            report.append(f"**Analysis:** {assessment.get('analysis', 'N/A')}")
            if assessment.get("recommendations"):
                report.append("**Recommendations:**")
                for rec in assessment["recommendations"]:
                    report.append(f"- {rec}")
            report.append("")
            
        return "\n".join(report)
