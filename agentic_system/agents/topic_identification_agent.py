"""
Topic Identification Agent (TIA) - Analyzes input queries to identify topics and intent.
"""
import asyncio
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai
from openai import AzureOpenAI
from azure.identity import ClientSecretCredential, get_bearer_token_provider
from dotenv import load_dotenv


# Handle imports for both module and direct execution
try:
    # Try relative imports first (when used as a module)
    from ..core.base import BaseAgent, Task, AgentResponse
    from ..enums import AgentType, LogLevel
    from ..audit.audit_log import audit_logger
except ImportError:
    # Fallback to absolute imports (when run directly)
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    
    from core.base import BaseAgent, Task, AgentResponse
    from enums import AgentType, LogLevel
    from audit.audit_log import audit_logger


class TopicIdentificationAgent(BaseAgent):
    """
    Agent responsible for analyzing input queries to identify topics, intent, and routing information.
    
    This agent serves as the entry point for the system, processing user queries and determining
    the appropriate handling strategy.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, AgentType.TOPIC_IDENTIFICATION, config)
        self.client: Optional[AzureOpenAI] = None
        self.confidence_threshold = config.get('confidence_threshold', 0.6) if config else 0.6
        self.max_topics = config.get('max_topics', 5) if config else 5
        
        # Topic classification prompts
        self.classification_prompt = """
        You are an expert topic identification system for a corporate knowledge management system. 
        Analyze the given query and provide detailed classification to route it to the appropriate knowledge database or agent.
        
        We have three main knowledge sources/agents:
        1. CONFLUENCE: Contains project documentation, technical information, RPA/AI projects, development processes, BAIA, BegoChat. Best for internal, structured data.
        2. NEWHQ: Contains office facility information, building details, parking, workplace amenities, headquarters information. Best for internal, facilities-related data.
        3. AUTONOMOUS_AGENT (Web Search): Can search the public internet for general knowledge, current events, sports, news, or information not available internally.
        
        Analyze this query: "{query}"
        
        Classification Guidelines:
        - SPORTS, NEWS, CURRENT EVENTS, GENERAL KNOWLEDGE → autonomous_agent (web search)
        - OFFICE, FACILITIES, PARKING, BUILDING → newhq 
        - TECHNICAL, PROJECTS, DEVELOPMENT, BAIA → confluence
        
        Provide classification for:
        1. Primary topics (up to {max_topics}) - specific topic keywords
        2. Intent classification (information_seeking, task_execution, analysis, compliance_check, etc.)
        3. Complexity level (simple, moderate, complex)
        4. Required capabilities (rag_search, web_search, computation, compliance_verification)
        5. Confidence score (0.0 to 1.0) - use 0.8+ for clear facility/office questions, 0.7+ for clear technical questions, 0.9+ for sports/news/current events
        6. Database/Agent recommendation (confluence, newhq, autonomous_agent, or a combination)
        7. Domain classification (technical, business, compliance, facilities, office, general, web_search, sports, news)
        
        IMPORTANT: Respond ONLY with valid JSON. No additional text or explanation.
        
        {{
            "topics": ["topic1", "topic2"],
            "intent": "primary_intent",
            "complexity": "complexity_level",
            "capabilities_needed": ["capability1", "capability2"],
            "confidence": 0.85,
            "database_recommendation": "newhq",
            "routing_strategy": "route_to_newhq_database",
            "metadata": {{
                "domain": "facilities",
                "urgency": "medium",
                "estimated_processing_time": "30_seconds",
                "topic_scores": {{
                    "confluence_relevance": 0.1,
                    "newhq_relevance": 0.9,
                    "web_search_relevance": 0.0
                }}
            }}
        }}
        """
    
    async def initialize(self) -> bool:
        """Initialize the Azure OpenAI client and other resources."""
        try:
            # Set NO_PROXY environment variable
            os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
            
            load_dotenv()
            
            # Load all required environment variables using the working pattern
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://begobaiatest.openai.azure.com/")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            
            # Service principal credentials
            tenant_id = os.getenv("AZURE_TENANT_ID")
            client_id = os.getenv("AZURE_CLIENT_ID")
            client_secret = os.getenv("AZURE_CLIENT_SECRET")
            
            # Validate required environment variables
            if not all([tenant_id, client_id, client_secret]):
                raise ValueError("Missing required environment variables: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
            
            # Initialize Azure credential with service principal
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
            
            # Create token provider for Azure OpenAI
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default"
            )
            
            # Initialize Azure OpenAI client with service principal authentication
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
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
        # Convert to async by running in executor since we're using sync client
        import asyncio
        loop = asyncio.get_event_loop()
        
        def sync_test():
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            return response.choices[0].message.content
        
        return await loop.run_in_executor(None, sync_test)
    
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
            
            # Generate final response
            final_response = self._generate_final_response(query, identification_result)
            
            # Validate results
            if identification_result['confidence'] < self.confidence_threshold:
                await audit_logger.log_agent_action(
                    self.agent_id, self.agent_type, "low_confidence_warning",
                    task=task, log_level=LogLevel.WARNING,
                    confidence=identification_result['confidence'],
                    threshold=self.confidence_threshold
                )
                # Override result for low confidence: force autonomous_agent routing
                identification_result.update({
                    'capabilities_needed': ['web_search'],
                    'database_recommendation': 'autonomous_agent',
                    'routing_strategy': 'route_to_autonomous_agent'
                })
                # Update final response for low confidence routing
                final_response = self._generate_final_response(query, identification_result)
            
            execution_time = time.time() - start_time
            
            # Log the final response
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "final_response_generated",
                task=task, log_level=LogLevel.INFO,
                final_response=final_response,
                database_recommendation=identification_result.get('database_recommendation'),
                confidence=identification_result.get('confidence')
            )
            
            response = AgentResponse(
                success=True,
                data={
                    'original_query': query,
                    'identification_result': identification_result,
                    'routing_recommendation': self._generate_routing_recommendation(identification_result),
                    'final_response': final_response
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
        import json  # Move import to top of method
        import asyncio
        
        try:
            prompt = self.classification_prompt.format(
                query=query,
                max_topics=self.max_topics
            )
            
            # Convert to async by running in executor since we're using sync client
            loop = asyncio.get_event_loop()
            
            def sync_chat_completion():
                return self.client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are an expert topic identification system. Always respond with valid JSON only, no additional text."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
            
            response = await loop.run_in_executor(None, sync_chat_completion)
            
            # Parse JSON response
            response_content = response.choices[0].message.content
            
            # Log the raw response for debugging
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "azure_openai_response",
                log_level=LogLevel.DEBUG,
                response_content=response_content[:500]  # First 500 chars
            )
            
            result = json.loads(response_content)
            
            # Add additional analysis
            result['query_length'] = len(query)
            result['word_count'] = len(query.split())
            result['analysis_timestamp'] = datetime.now().isoformat()
            result['fallback_used'] = False
            
            return result
            
        except json.JSONDecodeError as e:
            # Log JSON parsing error
            await audit_logger.log_agent_action(
                self.agent_id, self.agent_type, "json_parse_error",
                log_level=LogLevel.WARNING,
                error=str(e),
                response_content=response.choices[0].message.content[:200] if 'response' in locals() and response.choices else "No response content"
            )
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
                           'headquarters', 'workplace', 'infrastructure', 'amenities', 'newhq', 'new', 'hq'],
                'domain': 'facilities'
            },
            'web_search': {
                'keywords': ['coach', 'sport', 'sports', 'football', 'soccer', 'team', 'player', 'current', 
                           'news', 'weather', 'celebrity', 'movie', 'tv', 'music', 'general', 'public',
                           'hamburger', 'sv', 'hsv', 'bundesliga', 'manager', 'trainer'],
                'domain': 'web_search'
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
        web_search_score = 0
        domain_detected = 'general'
        
        for category, data in domain_keywords.items():
            matches = sum(1 for keyword in data['keywords'] if keyword in keywords)
            if matches > 0:
                identified_topics.extend([kw for kw in data['keywords'] if kw in keywords])
                
                if category == 'confluence':
                    confluence_score += matches * 0.5  # Weighted scoring
                    if matches >= 2:  # Bonus for multiple matches
                        confluence_score += 0.3
                    domain_detected = data['domain']
                elif category == 'newhq':
                    newhq_score += matches * 0.5  # Weighted scoring
                    if matches >= 2:  # Bonus for multiple matches
                        newhq_score += 0.3
                    domain_detected = data['domain']
                elif category == 'web_search':
                    web_search_score += matches * 0.5  # Weighted scoring
                    if matches >= 2:  # Bonus for multiple matches
                        web_search_score += 0.3
                    domain_detected = data['domain']
        
        # Determine database recommendation
        if web_search_score > max(newhq_score, confluence_score):
            database_recommendation = 'autonomous_agent'
        elif newhq_score > confluence_score:
            database_recommendation = 'newhq'
        elif confluence_score > newhq_score:
            database_recommendation = 'confluence'
        elif confluence_score > 0 and newhq_score > 0:
            database_recommendation = 'both'
        else:
            database_recommendation = 'autonomous_agent'  # Default to web search for unclear queries
        
        # Calculate confidence based on best match
        best_score = max(newhq_score, confluence_score, web_search_score, 0.1)
        confidence = min(0.5 + (best_score * 0.2), 0.9)  # Base 0.5, up to 0.9 for strong matches
        
        return {
            'topics': list(set(identified_topics[:self.max_topics])),
            'intent': 'information_seeking',
            'complexity': 'moderate',
            'capabilities_needed': ['web_search'] if database_recommendation == 'autonomous_agent' else ['rag_search'],
            'confidence': confidence,
            'database_recommendation': database_recommendation,
            'routing_strategy': f'route_to_{database_recommendation}_database' if database_recommendation != 'autonomous_agent' else 'route_to_autonomous_agent',
            'metadata': {
                'domain': domain_detected,
                'urgency': 'medium',
                'estimated_processing_time': '30_seconds',
                'fallback_used': True,
                'topic_scores': {
                    'confluence_relevance': confluence_score,
                    'newhq_relevance': newhq_score,
                    'web_search_relevance': web_search_score
                }
            }
        }
    
    def _generate_routing_recommendation(self, identification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate routing recommendations based on identification results."""
        capabilities = identification_result.get('capabilities_needed', [])
        complexity = identification_result.get('complexity', 'moderate')
        confidence = identification_result.get('confidence', 0.5)
        
        # Determine primary agent routing - Force web search for low confidence
        if confidence < self.confidence_threshold:
            # Low confidence always triggers web search through AutonomousAgent
            primary_agent = AgentType.AUTONOMOUS
            if 'web_search' not in capabilities:
                capabilities.append('web_search')
        elif 'compliance_verification' in capabilities:
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
            # Note: sync client doesn't need async close()
            
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
    
    def _generate_final_response(self, query: str, identification_result: Dict[str, Any]) -> str:
        """Generate a comprehensive final response about the query routing decision."""
        database_rec = identification_result.get('database_recommendation', 'unclear')
        topics = identification_result.get('topics', [])
        domain = identification_result.get('metadata', {}).get('domain', 'general')
        confidence = identification_result.get('confidence', 0.5)
        intent = identification_result.get('intent', 'information_seeking')
        complexity = identification_result.get('complexity', 'moderate')
        
        # Build response based on database recommendation
        if database_rec == 'newhq':
            response = f"Query '{query}' has been analyzed and routed to the NewHQ facilities database. "
            response += f"This appears to be a {domain}-related question about {', '.join(topics[:3])} "
            response += f"with {confidence:.1%} confidence. The system will search the NewHQ knowledge base "
            response += f"to provide information about office facilities, building details, or workplace amenities."
            
        elif database_rec == 'confluence':
            response = f"Query '{query}' has been routed to the Confluence technical database. "
            response += f"This appears to be a {domain}-related question about {', '.join(topics[:3])} "
            response += f"with {confidence:.1%} confidence. The system will search project documentation, "
            response += f"technical information, and development processes to answer your question."
            
        elif database_rec == 'autonomous_agent':
            # Check if it's a sports-related query for more specific messaging
            is_sports = any(topic.lower() in ['sports', 'football', 'coach', 'team', 'hamburger sv'] for topic in topics)
            
            if is_sports:
                response = f"Query '{query}' has been analyzed and will be processed by the autonomous web search agent. "
                response += f"This appears to be a sports-related question about {', '.join(topics[:3])} "
                response += f"with {confidence:.1%} confidence. The system will perform a live web search to find "
                response += f"current information, as sports data (team rosters, current coaches, recent transfers) "
                response += f"changes frequently and requires up-to-date external sources. The autonomous agent will "
                response += f"search multiple sports websites, validate the information, and provide you with the most "
                response += f"current answer available."
            else:
                response = f"Query '{query}' will be handled by the autonomous web search agent. "
                response += f"This appears to be a {domain} question about {', '.join(topics[:3])} "
                response += f"with {confidence:.1%} confidence. The system will search the public internet "
                response += f"for current information, as this query likely requires external knowledge not available in internal databases."
            
        elif database_rec == 'both':
            response = f"Query '{query}' spans multiple knowledge domains and will search both internal databases. "
            response += f"This {domain} question about {', '.join(topics[:3])} will be processed by both "
            response += f"Confluence and NewHQ databases to provide comprehensive information."
            
        else:
            response = f"Query '{query}' has unclear routing requirements. "
            response += f"The system identified this as a {domain} question with {confidence:.1%} confidence "
            response += f"but will use fallback routing to ensure you receive an appropriate response."
        
        # Add complexity and processing information
        if complexity == 'complex':
            response += f" Due to the complex nature of this query, additional processing time may be required."
        elif complexity == 'simple':
            response += f" This appears to be a straightforward query that should be resolved quickly."
        
        # Add intent information
        if intent == 'information_seeking':
            response += f" The system will focus on providing informational content to answer your question."
        elif intent == 'task_execution':
            response += f" The system will attempt to help you complete the requested task."
        elif intent == 'compliance_check':
            response += f" The system will review compliance requirements related to your query."
        
        return response
