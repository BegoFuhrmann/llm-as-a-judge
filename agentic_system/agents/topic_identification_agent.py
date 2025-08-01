"""
Topic Identification Agent (TIA) - Analyzes input queries to identify topics and intent.
"""
import asyncio
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai
from openai import AsyncAzureOpenAI

from ..core.base import BaseAgent, Task, AgentResponse
from ..enums import AgentType, LogLevel
from ..audit.audit_log import audit_logger


class TopicIdentificationAgent(BaseAgent):
    """
    Agent responsible for analyzing input queries to identify topics, intent, and routing information.
    
    This agent serves as the entry point for the system, processing user queries and determining
    the appropriate handling strategy.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, AgentType.TOPIC_IDENTIFICATION, config)
        self.client: Optional[AsyncAzureOpenAI] = None
        self.confidence_threshold = config.get('confidence_threshold', 0.8) if config else 0.8
        self.max_topics = config.get('max_topics', 5) if config else 5
        
        # Topic classification prompts
        self.classification_prompt = """
        You are an expert topic identification system for a corporate knowledge management system. 
        Analyze the given query and provide detailed classification to route it to the appropriate knowledge database.
        
        We have two main knowledge databases:
        1. CONFLUENCE: Contains project documentation, technical information, RPA/AI projects, development processes, BAIA, BegoChat
        2. NEWHQ: Contains office facility information, building details, parking, workplace amenities, headquarters information
        
        Analyze this query: "{query}"
        
        Provide classification for:
        1. Primary topics (up to {max_topics}) - specific topic keywords
        2. Intent classification (information_seeking, task_execution, analysis, compliance_check, etc.)
        3. Complexity level (simple, moderate, complex)
        4. Required capabilities (rag_search, web_search, computation, compliance_verification)
        5. Confidence score (0.0 to 1.0)
        6. Database recommendation (confluence, newhq, both, unclear)
        7. Domain classification (technical, business, compliance, facilities, office, general)
        
        Respond in JSON format:
        {{
            "topics": ["topic1", "topic2", ...],
            "intent": "primary_intent",
            "complexity": "complexity_level",
            "capabilities_needed": ["capability1", "capability2", ...],
            "confidence": 0.95,
            "database_recommendation": "confluence|newhq|both|unclear",
            "routing_strategy": "strategy_description",
            "metadata": {{
                "domain": "identified_domain",
                "urgency": "low|medium|high",
                "estimated_processing_time": "time_estimate",
                "topic_scores": {{
                    "confluence_relevance": 0.0-1.0,
                    "newhq_relevance": 0.0-1.0
                }}
            }}
        }}
        """
    
    async def initialize(self) -> bool:
        """Initialize the Azure OpenAI client and other resources."""
        try:
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            
            self.client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            
            # Test the connection
            await self._test_connection()
            
            self.is_active = True
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "initialize", 
                log_level=LogLevel.INFO
            )
            return True
            
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "initialize_failed",
                log_level=LogLevel.ERROR,
                error=str(e)
            )
            return False
    
    async def _test_connection(self):
        """Test the Azure OpenAI connection."""
        response = await self.client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4"),
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=10
        )
        return response.choices[0].message.content
    
    async def process(self, task: Task) -> AgentResponse:
        """
        Process a query to identify topics and routing strategy.
        
        Args:
            task: Task containing the query to analyze
            
        Returns:
            AgentResponse with topic identification results
        """
        start_time = time.time()
        
        try:
            query = task.input_data.get('query', '')
            if not query:
                return AgentResponse(
                    success=False,
                    error_message="No query provided for topic identification",
                    execution_time=time.time() - start_time
                )
            
            # Perform topic identification
            identification_result = await self._identify_topics(query)
            
            # Validate results
            if identification_result['confidence'] < self.confidence_threshold:
                await audit_logger.log_agent_action(
                    self.agent_id, self.agent_type, "low_confidence_warning",
                    task=task, log_level=LogLevel.WARNING,
                    confidence=identification_result['confidence'],
                    threshold=self.confidence_threshold
                )
            
            execution_time = time.time() - start_time
            
            response = AgentResponse(
                success=True,
                data={
                    'original_query': query,
                    'identification_result': identification_result,
                    'routing_recommendation': self._generate_routing_recommendation(identification_result)
                },
                confidence_score=identification_result['confidence'],
                execution_time=execution_time,
                sources=['azure_openai_analysis', 'internal_classification']
            )
            
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "process_completed",
                task=task, response=response, log_level=LogLevel.INFO
            )
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_response = AgentResponse(
                success=False,
                error_message=f"Topic identification failed: {str(e)}",
                execution_time=execution_time
            )
            
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "process_failed",
                task=task, response=error_response, log_level=LogLevel.ERROR
            )
            
            return error_response
    
    async def _identify_topics(self, query: str) -> Dict[str, Any]:
        """Use Azure OpenAI to identify topics and intent."""
        try:
            prompt = self.classification_prompt.format(
                query=query,
                max_topics=self.max_topics
            )
            
            response = await self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are an expert topic identification system."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse JSON response
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Add additional analysis
            result['query_length'] = len(query)
            result['word_count'] = len(query.split())
            result['analysis_timestamp'] = datetime.now().isoformat()
            
            return result
            
        except json.JSONDecodeError:
            # Fallback analysis if JSON parsing fails
            return self._fallback_analysis(query)
        except Exception as e:
            raise Exception(f"Topic identification failed: {str(e)}")
    
    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Provide basic fallback analysis if main method fails."""
        keywords = query.lower().split()
        
        # Enhanced keyword-based topic identification with database routing
        domain_keywords = {
            'confluence': {
                'keywords': ['project', 'development', 'baia', 'begochat', 'rpa', 'ai', 'automation', 
                           'confluence', 'technical', 'code', 'software', 'system', 'workflow'],
                'domain': 'technical'
            },
            'newhq': {
                'keywords': ['office', 'building', 'parking', 'facilities', 'location', 'space', 
                           'headquarters', 'workplace', 'infrastructure', 'amenities', 'newhq'],
                'domain': 'facilities'
            },
            'compliance': {
                'keywords': ['compliance', 'regulation', 'policy', 'audit', 'legal'],
                'domain': 'compliance'
            },
            'business': {
                'keywords': ['business', 'strategy', 'market', 'sales', 'revenue'],
                'domain': 'business'
            }
        }
        
        identified_topics = []
        confluence_score = 0
        newhq_score = 0
        domain_detected = 'general'
        
        for category, data in domain_keywords.items():
            matches = sum(1 for keyword in data['keywords'] if keyword in keywords)
            if matches > 0:
                identified_topics.extend([kw for kw in data['keywords'] if kw in keywords])
                
                if category == 'confluence':
                    confluence_score += matches
                    domain_detected = data['domain']
                elif category == 'newhq':
                    newhq_score += matches
                    domain_detected = data['domain']
        
        # Determine database recommendation
        if newhq_score > confluence_score:
            database_recommendation = 'newhq'
        elif confluence_score > newhq_score:
            database_recommendation = 'confluence'
        elif confluence_score > 0 and newhq_score > 0:
            database_recommendation = 'both'
        else:
            database_recommendation = 'unclear'
        
        return {
            'topics': list(set(identified_topics[:self.max_topics])),
            'intent': 'information_seeking',
            'complexity': 'moderate',
            'capabilities_needed': ['rag_search'],
            'confidence': 0.6,
            'database_recommendation': database_recommendation,
            'routing_strategy': f'route_to_{database_recommendation}_database',
            'metadata': {
                'domain': domain_detected,
                'urgency': 'medium',
                'estimated_processing_time': '30_seconds',
                'fallback_used': True,
                'topic_scores': {
                    'confluence_relevance': confluence_score / max(len(keywords), 1),
                    'newhq_relevance': newhq_score / max(len(keywords), 1)
                }
            }
        }
    
    def _generate_routing_recommendation(self, identification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate routing recommendations based on identification results."""
        capabilities = identification_result.get('capabilities_needed', [])
        complexity = identification_result.get('complexity', 'moderate')
        confidence = identification_result.get('confidence', 0.5)
        
        # Determine primary agent routing
        if 'compliance_verification' in capabilities:
            primary_agent = AgentType.COMPLIANCE_MONITORING
        elif 'web_search' in capabilities or complexity == 'complex':
            primary_agent = AgentType.AUTONOMOUS
        else:
            primary_agent = AgentType.RAG_BASED
        
        # Determine additional processing needs
        parallel_processing = []
        if confidence < 0.7:
            parallel_processing.append('validation_agent')
        if 'compliance_verification' in capabilities:
            parallel_processing.append('compliance_monitoring')
        
        return {
            'primary_agent': primary_agent.value,
            'parallel_processing': parallel_processing,
            'priority_level': self._determine_priority(identification_result),
            'estimated_resources': self._estimate_resources(complexity, capabilities),
            'validation_required': confidence < 0.8,
            'human_review_needed': complexity == 'complex' and confidence < 0.6
        }
    
    def _determine_priority(self, identification_result: Dict[str, Any]) -> str:
        """Determine processing priority based on identification results."""
        urgency = identification_result.get('metadata', {}).get('urgency', 'medium')
        compliance_related = 'compliance_verification' in identification_result.get('capabilities_needed', [])
        
        if urgency == 'high' or compliance_related:
            return 'high'
        elif urgency == 'low':
            return 'low'
        else:
            return 'medium'
    
    def _estimate_resources(self, complexity: str, capabilities: List[str]) -> Dict[str, Any]:
        """Estimate resource requirements for processing."""
        base_time = {'simple': 10, 'moderate': 30, 'complex': 120}
        capability_multipliers = {
            'web_search': 2.0,
            'compliance_verification': 1.5,
            'computation': 1.3,
            'rag_search': 1.0
        }
        
        estimated_time = base_time.get(complexity, 30)
        for capability in capabilities:
            estimated_time *= capability_multipliers.get(capability, 1.0)
        
        return {
            'estimated_time_seconds': int(estimated_time),
            'memory_usage': 'low' if complexity == 'simple' else 'medium',
            'compute_intensity': 'high' if 'computation' in capabilities else 'medium',
            'external_api_calls': len([c for c in capabilities if c in ['web_search', 'compliance_verification']])
        }
    
    async def shutdown(self) -> bool:
        """Shutdown the agent gracefully."""
        try:
            self.is_active = False
            if self.client:
                await self.client.close()
            
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "shutdown",
                log_level=LogLevel.INFO
            )
            return True
            
        except Exception as e:
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "shutdown_failed",
                log_level=LogLevel.ERROR,
                error=str(e)
            )
            return False
    
    def analyze_query_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple queries in batch for efficiency."""
        # This could be implemented for batch processing optimization
        pass
    
    async def get_database_recommendation(self, query: str) -> Dict[str, Any]:
        """
        Get a direct database recommendation for a query.
        
        Args:
            query: The user query to analyze
            
        Returns:
            Dict containing database recommendation and confidence
        """
        try:
            identification_result = await self._identify_topics(query)
            
            database_rec = identification_result.get('database_recommendation', 'unclear')
            confidence = identification_result.get('confidence', 0.5)
            topic_scores = identification_result.get('metadata', {}).get('topic_scores', {})
            
            return {
                'database': database_rec,
                'confidence': confidence,
                'topic_scores': topic_scores,
                'topics': identification_result.get('topics', []),
                'domain': identification_result.get('metadata', {}).get('domain', 'general'),
                'reasoning': self._generate_routing_reasoning(identification_result)
            }
            
        except Exception as e:
            # Fallback to simple keyword analysis
            fallback_result = self._fallback_analysis(query)
            return {
                'database': fallback_result.get('database_recommendation', 'unclear'),
                'confidence': 0.4,
                'topic_scores': fallback_result.get('metadata', {}).get('topic_scores', {}),
                'topics': fallback_result.get('topics', []),
                'domain': fallback_result.get('metadata', {}).get('domain', 'general'),
                'reasoning': f"Fallback analysis due to error: {str(e)}",
                'error': str(e)
            }
    
    def _generate_routing_reasoning(self, identification_result: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for database routing decision."""
        database_rec = identification_result.get('database_recommendation', 'unclear')
        topics = identification_result.get('topics', [])
        domain = identification_result.get('metadata', {}).get('domain', 'general')
        confidence = identification_result.get('confidence', 0.5)
        
        if database_rec == 'confluence':
            return f"Routed to Confluence database based on {domain} domain topics: {', '.join(topics[:3])} (confidence: {confidence:.2f})"
        elif database_rec == 'newhq':
            return f"Routed to NewHQ database based on {domain} domain topics: {', '.join(topics[:3])} (confidence: {confidence:.2f})"
        elif database_rec == 'both':
            return f"Query spans multiple domains ({domain}), searching both databases for topics: {', '.join(topics[:3])}"
        else:
            return f"Unclear routing for general query, using fallback strategy (confidence: {confidence:.2f})"
