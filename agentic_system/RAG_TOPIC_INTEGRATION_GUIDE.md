# RAG Agent - Topic Identification Integration

## Overview

This document describes the integration between the **RAG Agent** and the **Topic Identification Agent** to intelligently choose the appropriate ChromaDB for query processing. This integration enables dynamic database selection based on semantic understanding of user queries.

## Architecture

### Components

1. **RAG Agent** (`rag_agent.py`)
   - Main agent handling knowledge retrieval and response generation
   - Manages two ChromaDB databases: Confluence and NewHQ
   - Integrates with Topic Identification Agent for intelligent routing

2. **Topic Identification Agent** (`topic_identification_agent.py`)
   - Analyzes user queries to identify topics, intent, and domain
   - Provides database recommendations based on semantic analysis
   - Uses Azure OpenAI for advanced topic classification

3. **ChromaDB Databases**
   - **Confluence DB**: Project documentation, technical info, RPA/AI projects, BAIA, BegoChat
   - **NewHQ DB**: Office facilities, building info, parking, workplace amenities

## Integration Flow

```
User Query → Topic Identification Agent → Database Selection → RAG Agent → Response
     ↓              ↓                           ↓              ↓           ↓
"Where to park?" → facilities/office → NewHQ DB → Search → "Parking info..."
"BAIA project?" → technical/project → Confluence DB → Search → "BAIA details..."
```

## Key Features

### 1. Intelligent Database Routing

The system automatically routes queries to the most appropriate database based on:

- **Topic Analysis**: Semantic understanding of query content
- **Domain Classification**: Technical, facilities, business, compliance
- **Keyword Mapping**: Enhanced keyword-to-database mapping
- **Confidence Scoring**: Reliability assessment of routing decisions

### 2. Multi-Database Search

When topic identification is ambiguous, the system can:
- Search both databases
- Combine results from multiple sources
- Provide comprehensive responses

### 3. Fallback Mechanisms

Multiple fallback strategies ensure robustness:
- Keyword-based routing when topic identification fails
- Sequential database searching
- Error handling with graceful degradation

## Configuration

### Database Paths

```python
confluence_db_path = "\\dnsbego.de\\dfsbego\\home04\\FuhrmannD\\Documents\\01_Trainee\\Master\\Thesis\\code\\agentic_system\\data\\confluence_chroma_db"
newhq_db_path = "\\dnsbego.de\\dfsbego\\home04\\FuhrmannD\\Documents\\01_Trainee\\Master\\Thesis\\code\\agentic_system\\data\\newHQ_chroma_db"
```

### Topic-to-Database Mapping

```python
topic_to_database_mapping = {
    'confluence': [
        'confluence', 'project', 'baia', 'begochat', 'rpa', 'ai', 
        'automation', 'development', 'technology', 'innovation'
    ],
    'newhq': [
        'office', 'building', 'parking', 'facilities', 'location', 
        'space', 'headquarters', 'workplace', 'infrastructure'
    ]
}
```

## Usage Examples

### Basic Usage

```python
from agents.rag_agent import RAGAgent
from core.base import Task
from enums import AgentType

# Initialize RAG Agent (automatically includes Topic Identification)
rag_agent = RAGAgent(agent_id="my_rag_agent")
await rag_agent.initialize()

# Process a query
task = Task(
    task_id="query_1",
    input_data={'query': 'Where can I park at the new office?'},
    agent_type=AgentType.RAG_BASED
)

response = await rag_agent.process(task)
print(f"Database used: {response.data['database_used']}")
print(f"Answer: {response.data['answer']}")
```

### Direct Topic Identification

```python
# Get database recommendation for a query
if rag_agent.topic_agent:
    recommendation = await rag_agent.topic_agent.get_database_recommendation(
        "What is the BAIA project?"
    )
    print(f"Recommended DB: {recommendation['database']}")
    print(f"Confidence: {recommendation['confidence']}")
    print(f"Reasoning: {recommendation['reasoning']}")
```

## Query Examples and Expected Routing

| Query | Expected Database | Reasoning |
|-------|------------------|-----------|
| "Where can I park at the new headquarters?" | NewHQ | Office facilities query |
| "What is the BAIA project about?" | Confluence | Project documentation |
| "How do I access building facilities?" | NewHQ | Building access/facilities |
| "Tell me about RPA automation workflows" | Confluence | Technical documentation |
| "What are the office hours?" | NewHQ | Office information |
| "How does BegoChat work?" | Confluence | AI project documentation |

## Response Format

### Successful Response
```python
{
    'success': True,
    'data': {
        'query': 'original query',
        'answer': 'generated response',
        'search_method': 'confluence_search_topic_identified',
        'database_used': 'Confluence ChromaDB',
        'selection_reason': 'Topic identification indicated project/technical query',
        'rag_metadata': {...}
    },
    'confidence_score': 0.85,
    'sources': ['Confluence Database']
}
```

### Topic Identification Details
```python
{
    'database': 'confluence',
    'confidence': 0.92,
    'topics': ['project', 'ai', 'automation'],
    'domain': 'technical',
    'reasoning': 'Routed to Confluence database based on technical domain topics: project, ai, automation (confidence: 0.92)',
    'topic_scores': {
        'confluence_relevance': 0.85,
        'newhq_relevance': 0.15
    }
}
```

## Error Handling

The integration includes comprehensive error handling:

1. **Topic Agent Unavailable**: Falls back to keyword-based routing
2. **Database Connection Issues**: Graceful degradation with error messages
3. **Azure OpenAI Failures**: Fallback to local keyword analysis
4. **Invalid Queries**: Appropriate error responses

## Performance Considerations

- **Caching**: Topic identification results can be cached for similar queries
- **Parallel Processing**: Topic identification and database initialization run concurrently
- **Resource Management**: Proper cleanup of agents and connections

## Testing

Use the provided test script to validate the integration:

```bash
python test_rag_topic_integration.py
```

This script tests:
- Topic identification accuracy
- Database routing correctness
- End-to-end query processing
- Error handling scenarios

## Monitoring and Logging

All operations are logged through the audit system:

- Topic identification results
- Database selection decisions
- Search performance metrics
- Error conditions and fallbacks

## Future Enhancements

1. **Machine Learning**: Train models on query-database routing patterns
2. **User Feedback**: Incorporate user feedback to improve routing accuracy
3. **Multi-language Support**: Extend topic identification to multiple languages
4. **Performance Optimization**: Implement query result caching
5. **Advanced Analytics**: Query pattern analysis and optimization recommendations

## Troubleshooting

### Common Issues

1. **Topic Agent Not Initializing**
   - Check Azure OpenAI credentials
   - Verify network connectivity
   - Review environment variables

2. **Database Not Found**
   - Verify ChromaDB paths exist
   - Check file permissions
   - Ensure databases are properly populated

3. **Poor Routing Accuracy**
   - Review topic-to-database mapping
   - Adjust confidence thresholds
   - Add domain-specific keywords

### Debug Mode

Enable debug logging to see detailed routing decisions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show:
- Topic identification process
- Database selection reasoning
- Search execution details
- Response generation steps
