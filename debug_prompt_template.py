#!/usr/bin/env python3
"""
Debug script to see the exact prompt being sent to Azure OpenAI
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from rag_agentic_evaluation.models import SourceSnippet
from rag_agentic_evaluation.templates import render_template, SNIPPET_EVALUATION_PROMPT

def main():
    # Load environment variables
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    
    # Test data matching your Excel file
    query = "Where can I park my car?"
    answer = """There are several parking options available to you at the new office building. 
    * **Internal parking spaces:** There are 115 internal parking spaces that can be booked flexibly. These are located in the building's underground parking garage. You can reserve a space via the "ParkHere" app. The parking spaces are booked on a daily basis and are billed monthly. With the app you can reserve a parking space up to two weeks in advance for a maximum of five days at a time. The parking garage can be accessed with your general access card. 
    * **External parking spaces:** There are external parking spaces in the nearby Überseering 32a parking garage, which is about a 3-minute walk away. 
    * **Guest parking spaces:** On weekends, all employees have the option of parking on the guest parking spaces free of charge. 
    * **Bicycle parking:** There are 157 bicycle parking spaces in the underground parking garage. Please note that if you enter the parking garage without a booking, you will be charged. Would you like more detailed instructions on how to use the "ParkHere" app, or is this information sufficient for now?"""
    
    snippets = [
        SourceSnippet(
            content="Several cars are parked on both sides of a central lane. The floor is marked with white lines indicating parking spaces and a central guide line.",
            snippet_id="1"
        ),
        SourceSnippet(
            content="The image is likely related to parking management or allocation. The presence of the text suggests that the location has designated parking spaces",
            snippet_id="2"
        ),
        SourceSnippet(
            content="parking area with orange columns and parking spaces marked with red stripes. The word BESUCHER is written on the wall.",
            snippet_id="4"
        ),
    ]
    
    # Render the prompt exactly as the framework does
    rendered_prompt = render_template(
        SNIPPET_EVALUATION_PROMPT,
        query=query,
        answer=answer,
        snippets=snippets
    )
    
    print("RENDERED PROMPT:")
    print("=" * 80)
    print(rendered_prompt)
    print("=" * 80)
    
    # Check for potential issues
    issues = []
    if "```json" in rendered_prompt:
        issues.append("Template contains ```json markers that may confuse JSON parsing")
    if "```" in rendered_prompt:
        issues.append("Template contains markdown code blocks")
    
    if issues:
        print("\n⚠️  POTENTIAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ No obvious template issues found")
    
    return 0

if __name__ == "__main__":
    exit(main())
