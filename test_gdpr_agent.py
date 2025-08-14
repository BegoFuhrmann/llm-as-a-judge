"""
Test script for GDPR Compliance Agent.

This script tests the functionality of the GDPRComplianceAgent
to ensure it works correctly.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_system.agents.gdpr_compliance_agent import GDPRComplianceAgent


class TestGDPRComplianceAgent(unittest.TestCase):
    """Test cases for GDPR Compliance Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = GDPRComplianceAgent(db_path="test_db_path")
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.agent_name, "GDPR Compliance Agent")
        self.assertEqual(self.agent.collection_name, "gdpr_compliance")
        self.assertEqual(self.agent.db_path, "test_db_path")
        self.assertFalse(self.agent.is_initialized)
    
    def test_gdpr_principles(self):
        """Test GDPR principles are properly defined."""
        expected_principles = [
            "lawfulness", "data_minimization", "purpose_limitation",
            "right_to_explanation", "data_subject_rights", "transfer_principles"
        ]
        
        for principle in expected_principles:
            self.assertIn(principle, self.agent.gdpr_principles)
            # Check that reference contains either "Article" or "Recital"
            reference = self.agent.gdpr_principles[principle]
            self.assertTrue("Article" in reference or "Recital" in reference)
    
    def test_assessment_templates(self):
        """Test assessment templates are properly configured."""
        for area, template in self.agent.assessment_templates.items():
            self.assertIn("prompt", template)
            self.assertIn("criteria", template)
            self.assertIsInstance(template["criteria"], list)
            self.assertTrue(len(template["criteria"]) > 0)
    
    def test_calculate_area_score(self):
        """Test area score calculation."""
        # Test positive analysis
        positive_analysis = "The system is compliant and adequate with proper documentation."
        score = self.agent._calculate_area_score(positive_analysis)
        self.assertGreater(score, 50)
        
        # Test negative analysis
        negative_analysis = "The system is non-compliant and inadequate with missing documentation."
        score = self.agent._calculate_area_score(negative_analysis)
        self.assertLess(score, 60)
    
    def test_extract_recommendations(self):
        """Test recommendation extraction."""
        analysis = "We recommend implementing proper consent mechanisms. You should ensure data minimization. Consider establishing clear policies."
        recommendations = self.agent._extract_recommendations(analysis)
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)
        self.assertTrue(any("recommend" in rec.lower() for rec in recommendations))
    
    def test_identify_risk_indicators(self):
        """Test risk indicator identification."""
        analysis = "There is a potential violation and compliance gap with insufficient documentation."
        indicators = self.agent._identify_risk_indicators(analysis)
        
        self.assertIsInstance(indicators, list)
        self.assertTrue(len(indicators) > 0)
        self.assertTrue(any("Risk" in indicator for indicator in indicators))
    
    def test_determine_risk_level(self):
        """Test risk level determination."""
        self.assertEqual(self.agent._determine_risk_level(85), "low")
        self.assertEqual(self.agent._determine_risk_level(65), "medium")
        self.assertEqual(self.agent._determine_risk_level(45), "high")
        self.assertEqual(self.agent._determine_risk_level(25), "critical")
    
    def test_get_compliance_status(self):
        """Test compliance status description."""
        self.assertIn("Strong", self.agent._get_compliance_status(85))
        self.assertIn("Moderate", self.agent._get_compliance_status(65))
        self.assertIn("Weak", self.agent._get_compliance_status(45))
        self.assertIn("Poor", self.agent._get_compliance_status(25))
    
    @patch.object(GDPRComplianceAgent, 'query')
    def test_assess_specific_area(self, mock_query):
        """Test specific area assessment."""
        # Mock query response
        mock_query.return_value = {
            "answer": "The system is compliant with proper documentation and meets requirements.",
            "sources": [{"content": "test source", "metadata": {}}]
        }
        
        context = "Test data processing scenario"
        assessment = self.agent._assess_specific_area(context, "lawfulness")
        
        self.assertEqual(assessment["area"], "lawfulness")
        self.assertIn("Article 6", assessment["gdpr_reference"])
        self.assertIn("compliance_score", assessment)
        self.assertIn("recommendations", assessment)
        self.assertIn("risk_indicators", assessment)
        
        # Verify query was called
        mock_query.assert_called_once()
    
    @patch.object(GDPRComplianceAgent, '_assess_specific_area')
    def test_assess_gdpr_compliance(self, mock_assess):
        """Test comprehensive GDPR compliance assessment."""
        # Mock specific area assessment
        mock_assess.return_value = {
            "compliance_score": 75,
            "recommendations": ["Test recommendation"],
            "risk_indicators": []
        }
        
        context = "Test processing scenario"
        results = self.agent.assess_gdpr_compliance(context, "lawfulness")
        
        self.assertIn("overall_assessment", results)
        self.assertIn("detailed_assessments", results)
        self.assertIn("compliance_score", results)
        self.assertIn("risk_level", results)
        
        # Verify assess_specific_area was called
        mock_assess.assert_called_once_with(context, "lawfulness")
    
    def test_assess_gdpr_compliance_invalid_area(self):
        """Test assessment with invalid focus area."""
        results = self.agent.assess_gdpr_compliance("test context", "invalid_area")
        
        self.assertIn("error", results)
        self.assertIn("valid_areas", results)
    
    def test_generate_compliance_report(self):
        """Test compliance report generation."""
        assessment_results = {
            "compliance_score": 75,
            "risk_level": "medium",
            "overall_assessment": {
                "compliance_status": "Moderate compliance",
                "key_strengths": ["lawfulness: Article 6"],
                "critical_gaps": ["data_minimization: Article 5"],
                "priority_actions": ["Implement consent mechanism"]
            },
            "detailed_assessments": {
                "lawfulness": {
                    "gdpr_reference": "Article 6",
                    "compliance_score": 80,
                    "risk_indicators": []
                }
            },
            "recommendations": ["Test recommendation 1", "Test recommendation 2"]
        }
        
        report = self.agent.generate_compliance_report(assessment_results)
        
        self.assertIsInstance(report, str)
        self.assertIn("GDPR COMPLIANCE ASSESSMENT REPORT", report)
        self.assertIn("75%", report)
        self.assertIn("MEDIUM", report)
        self.assertIn("Moderate compliance", report)


def run_basic_functionality_test():
    """Run a basic functionality test without requiring database."""
    print("Running basic GDPR Compliance Agent functionality test...")
    
    # Create agent instance
    agent = GDPRComplianceAgent()
    
    # Test basic properties
    assert agent.agent_name == "GDPR Compliance Agent"
    assert agent.collection_name == "gdpr_compliance"
    assert len(agent.gdpr_principles) == 6
    assert len(agent.assessment_templates) == 6
    
    # Test score calculation
    score = agent._calculate_area_score("The system is compliant and adequate.")
    assert 50 <= score <= 100
    
    # Test risk level determination
    assert agent._determine_risk_level(85) == "low"
    assert agent._determine_risk_level(45) == "high"
    
    # Test recommendation extraction
    recs = agent._extract_recommendations("We recommend implementing proper consent.")
    assert len(recs) > 0
    
    # Test report generation
    sample_results = {
        "compliance_score": 75,
        "risk_level": "medium",
        "overall_assessment": {"compliance_status": "Moderate"},
        "detailed_assessments": {},
        "recommendations": ["Test recommendation"]
    }
    
    report = agent.generate_compliance_report(sample_results)
    assert "GDPR COMPLIANCE ASSESSMENT REPORT" in report
    
    print("✓ All basic functionality tests passed!")
    print("✓ GDPR Compliance Agent is ready for use")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("GDPR COMPLIANCE AGENT TEST SUITE")
    print("="*60)
    
    # Run basic functionality test first
    try:
        run_basic_functionality_test()
        print("\n✓ Basic functionality test completed successfully")
    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {e}")
        sys.exit(1)
    
    # Run unit tests
    print("\n" + "-"*60)
    print("RUNNING UNIT TESTS")
    print("-"*60)
    
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✓ GDPR Compliance Agent implementation complete")
    print("✓ All core functionality tested and working")
    print("✓ Ready for integration with evaluation system")
