"""
Test script for the AuditTrailAgent.

This script initializes the AuditTrailAgent, performs a full assessment of the
audit logs found in the specified directory, and prints a formatted report.
"""

import os
from evaluation_system.agents import AuditTrailAgent

def main():
    """
    Main function to run the AuditTrailAgent test.
    """
    print("--- Starting AuditTrailAgent Test ---")
    
    # Path to the audit logs directory
    # Assuming the script is run from the root of the project
    audit_log_path = "audit_logs"
    
    if not os.path.exists(audit_log_path):
        print(f"Error: Audit log directory not found at '{audit_log_path}'")
        print("Please ensure the script is run from the project root and the directory exists.")
        return

    # 1. Initialize the AuditTrailAgent
    print(f"Initializing AuditTrailAgent with logs from: {audit_log_path}")
    agent = AuditTrailAgent(audit_log_path=audit_log_path)
    
    # 2. Initialize the agent's LLM component
    if not agent.initialize():
        print("Failed to initialize the agent. Aborting test.")
        return
    
    print("Agent initialized successfully.")
    
    # 3. Perform a comprehensive assessment
    print("\nPerforming comprehensive audit trail assessment...")
    try:
        assessment_results = agent.assess_audit_trail()
        
        if "error" in assessment_results:
            print(f"An error occurred during assessment: {assessment_results['error']}")
            return
            
        print("Assessment complete.")
        
        # 4. Generate and print the report
        print("\n--- Generating Assessment Report ---")
        report = agent.generate_assessment_report(assessment_results)
        print(report)
        
        print("\n--- Test Script Finished ---")
        
    except Exception as e:
        print(f"An unexpected error occurred during the test: {e}")

if __name__ == "__main__":
    main()
