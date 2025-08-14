"""
Test script for the EUAIActAgent.

This script initializes the EUAIActAgent, provides a sample AI system context,
performs a full compliance assessment against the EU AI Act, and prints a
formatted report.
"""

import os
from evaluation_system.agents import EUAIActAgent

def main():
    """
    Main function to run the EUAIActAgent test.
    """
    print("--- Starting EUAIActAgent Test ---")

    # Path to the EU AI Act knowledge base directory.
    # This should be created by `create_regulatory_chroma_dbs.py`.
    db_path = os.path.join("data", "chroma_dbs", "eu_ai_act_compliance")

    if not os.path.exists(db_path) or not os.listdir(db_path):
        print(f"Warning: EU AI Act knowledge base not found or empty at '{db_path}'.")
        print("The agent will proceed but will rely only on its internal knowledge, not the specific knowledge base.")
        # Set db_path to None so the agent can initialize without a vector DB.
        db_path = None

    # 1. Initialize the EUAIActAgent
    print(f"Initializing EUAIActAgent with knowledge base from: {db_path or 'N/A'}")
    agent = EUAIActAgent(db_path=db_path)

    # 2. Initialize the agent's LLM and other components
    if not agent.initialize():
        print("Failed to initialize the agent. Aborting test.")
        return

    print("Agent initialized successfully.")

    # 3. Define a sample context for assessment
    scenario = """
    The system is an AI-powered recruitment tool used by a large corporation to screen
    and shortlist job candidates. It analyzes resumes, cover letters, and video interviews
    to rank candidates based on predicted job performance and cultural fit. The system's
    decision-making process is a black box, and the training data includes historical
    hiring data from the last 20 years, which may contain historical biases.
    The system is intended for use in the EU market.
    """
    print("\n--- Sample AI System Context ---")
    print(scenario.strip())
    print("---------------------------------")

    # 4. Perform a comprehensive assessment
    print("\nPerforming comprehensive EU AI Act compliance assessment...")
    try:
        assessment_results = agent.assess_eu_ai_act_compliance(scenario)
        
        if "error" in assessment_results:
            print(f"An error occurred during assessment: {assessment_results['error']}")
            return
            
        print("Assessment complete.")
        
        # 5. Generate and print the report
        print("\n--- Generating Assessment Report ---")
        report = agent.generate_compliance_report(assessment_results)
        print(report)
        
        print("\n--- Test Script Finished ---")
        
    except Exception as e:
        print(f"An unexpected error occurred during the test: {e}")

if __name__ == "__main__":
    main()
