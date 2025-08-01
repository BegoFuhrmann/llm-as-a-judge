"""
Interactive RAG-Topic Query Script
- Enter a question, get routed via TopicIdentificationAgent to the correct ChromaDB via RAGAgent.
- Run: python interactive_rag_topic.py
"""
import asyncio
import sys

# Handle imports for both module and direct execution
try:
    from agents.rag_agent import RAGAgent
    from agents.topic_identification_agent import TopicIdentificationAgent
    from core.base import Task, Priority
except ImportError:
    sys.path.append('.')
    from agents.rag_agent import RAGAgent
    from agents.topic_identification_agent import TopicIdentificationAgent
    from core.base import Task, Priority

async def main():
    print("\n=== Interactive RAG-Topic Query ===\n")
    # Initialize agents
    rag_agent = RAGAgent(agent_id="interactive_rag_agent")
    await rag_agent.initialize()
    topic_agent = rag_agent.topic_agent  # Use the same topic agent instance

    while True:
        user_query = input("Enter your question (or 'exit' to quit): ")
        if user_query.strip().lower() in ("exit", "quit"): break

        # Step 1: Topic Identification
        topic_task = Task(
            id=f"topic_id_{int(asyncio.get_event_loop().time())}",
            name="Topic Identification",
            description="Identify topics and database routing for query",
            input_data={'query': user_query},
            priority=Priority.MEDIUM
        )
        topic_response = await topic_agent.process(topic_task)
        if not topic_response.success:
            print(f"[Topic Agent Error] {topic_response.error_message}")
            continue
        identification_result = topic_response.data.get('identification_result', {})
        print(f"\n[Topic Identification] Database: {identification_result.get('database_recommendation', 'unknown')}, Topics: {identification_result.get('topics', [])}")

        # Step 2: RAG Query
        rag_task = Task(
            id=f"rag_id_{int(asyncio.get_event_loop().time())}",
            name="RAG Query",
            description=user_query,
            input_data={'query': user_query, 'identification_result': identification_result},
            priority=Priority.MEDIUM
        )
        rag_response = await rag_agent.process(rag_task)
        if not rag_response.success:
            print(f"[RAG Agent Error] {rag_response.error_message}")
            continue
        answer = rag_response.data.get('answer') or rag_response.data.get('response')
        print(f"\n=== RAG Answer ===\n{answer}\n")
        print(f"[Database Used] {rag_response.data.get('database_used', 'unknown')}")
        print(f"[Meta] {rag_response.data.get('rag_metadata', {})}\n")

if __name__ == "__main__":
    asyncio.run(main())
